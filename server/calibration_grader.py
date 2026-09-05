"""
Calibration Grader - the 3x2 matrix. Fully deterministic, no AI, on
purpose: the scoring must be trustworthy and non-gameable, so it is a
plain lookup table, not a model call.

              Correct     Wrong
    HIGH      +1.0        -0.8   <- worst outcome
    MED       +0.6        -0.2
    LOW       +0.1         0.0   <- safe

Also computes the "cost of overconfidence" rupee figure: for every
wrong HIGH-confidence terminal decision, we attribute a liability cost
using LIABILITY_SHARE of the transaction amount (default 20%, chosen to
mirror RBI's proposed bank liability share in its 2026 UPI fraud
compensation pilot - clearly labeled as an illustrative assumption).
"""
from __future__ import annotations
import os
from dataclasses import dataclass

MATRIX = {
    ("HIGH", True): 1.0,
    ("HIGH", False): -0.8,
    ("MED", True): 0.6,
    ("MED", False): -0.2,
    ("LOW", True): 0.1,
    ("LOW", False): 0.0,
}

LIABILITY_SHARE = float(os.getenv("LIABILITY_SHARE", "0.20"))

# maps terminal action -> which ground truth label it implies "correct" for
ACTION_CORRECT_LABEL = {
    "flag_fraud": "fraud",
    "allow_transaction": "legitimate",
    "escalate_to_review": None,  # escalation is graded as "correct" if the case was genuinely ambiguous/fraud-leaning
}


@dataclass
class ScoreResult:
    matrix_cell: str
    reward: float
    is_correct: bool
    cost_of_overconfidence_inr: float
    explanation: str


def grade_decision(action: str, confidence: str, ground_truth_label: str, amount: float) -> ScoreResult:
    if confidence not in ("HIGH", "MED", "LOW"):
        raise ValueError(f"Invalid confidence level: {confidence}")
    if action not in ACTION_CORRECT_LABEL:
        raise ValueError(f"Invalid terminal action: {action}")

    if action == "escalate_to_review":
        # Escalation is treated as "correct" whenever the case truly was
        # fraud (a human should look at real fraud) or when the case was
        # a friendly-fraud style judgment call. It is treated as a soft
        # miss (still scored, but never the worst cell) on clean legit cases.
        is_correct = ground_truth_label == "fraud"
    else:
        is_correct = ACTION_CORRECT_LABEL[action] == ground_truth_label

    reward = MATRIX[(confidence, is_correct)]
    cell = f"{confidence}/{'CORRECT' if is_correct else 'WRONG'}"

    cost = 0.0
    if confidence == "HIGH" and not is_correct:
        cost = round(amount * LIABILITY_SHARE, 2)

    explanation = (
        f"Action '{action}' at {confidence} confidence vs ground truth '{ground_truth_label}' "
        f"-> {'CORRECT' if is_correct else 'WRONG'} -> matrix cell [{cell}] -> reward {reward:+.2f}."
    )
    if cost:
        explanation += f" Cost of overconfidence attributed: Rs {cost:,.2f} ({LIABILITY_SHARE*100:.0f}% of transaction amount)."

    return ScoreResult(matrix_cell=cell, reward=reward, is_correct=is_correct,
                        cost_of_overconfidence_inr=cost, explanation=explanation)
