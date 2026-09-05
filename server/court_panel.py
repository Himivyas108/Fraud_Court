"""
Court Panel: Prosecutor / Defender / Judge.

This is the project's headline mechanic. Three role-scoped calls run in a
fixed sequence over the SAME evidence base (no new information leakage).
Implemented as plain Python + direct LLM calls with a deterministic
heuristic fallback - no agent framework needed for a fixed 3-step debate.
"""
from __future__ import annotations
import json
from dataclasses import dataclass, field
from server.llm_client import call_llm_json, LLMFailure

PROSECUTOR_SYSTEM = (
    "You are the Prosecutor in an adversarial fraud-review panel. You argue, using ONLY "
    "the evidence provided, that this transaction/dispute IS fraud. Be specific and cite "
    "the evidence. Do not invent evidence not given to you. "
    'Return JSON: {"argument": str, "cited_evidence": [str], "strength": "STRONG"|"MODERATE"|"WEAK"}'
)
DEFENDER_SYSTEM = (
    "You are the Defender in an adversarial fraud-review panel. You argue, using ONLY the "
    "evidence provided, that this transaction/dispute is LEGITIMATE. Be specific and cite "
    "the evidence. Do not invent evidence not given to you. "
    'Return JSON: {"argument": str, "cited_evidence": [str], "strength": "STRONG"|"MODERATE"|"WEAK"}'
)
JUDGE_SYSTEM = (
    "You are the Judge. Weigh the Prosecutor's and Defender's arguments, grounded only in "
    "the evidence log, and produce a recommendation. You do not have to average the two "
    "sides' strengths - explain your actual reasoning, including if you override what a "
    "naive strength-average would suggest. "
    'Return JSON: {"recommendation": "fraud"|"legitimate"|"uncertain", "reasoning": str, '
    '"confidence_leaning": "HIGH"|"MED"|"LOW"}'
)


@dataclass
class DebateRecord:
    prosecutor_argument: dict
    defender_argument: dict
    verdict: dict
    mode: str  # "llm" | "heuristic_fallback"
    fallback_reason: str | None = None


def _format_evidence(evidence_log: list[dict]) -> str:
    lines = []
    for e in evidence_log:
        lines.append(f"- [{e['tool']}] signal={e['signal']}: {e['result']}")
    return "\n".join(lines) if lines else "(no evidence gathered yet)"


def _heuristic_argument(evidence_log: list[dict], side: str) -> dict:
    matching = [e for e in evidence_log if e["signal"] == side]
    if side == "fraud":
        strength = "STRONG" if len(matching) >= 3 else ("MODERATE" if matching else "WEAK")
        if matching:
            argument = "Multiple independent signals point to fraud: " + "; ".join(m["result"] for m in matching)
        else:
            argument = "No direct fraud-indicating evidence was found in the investigation so far."
    else:
        strength = "STRONG" if len(matching) >= 3 else ("MODERATE" if matching else "WEAK")
        if matching:
            argument = "The evidence is consistent with a legitimate transaction: " + "; ".join(m["result"] for m in matching)
        else:
            argument = "No benign explanation is directly supported by the evidence gathered so far."
    return {
        "argument": argument,
        "cited_evidence": [m["tool"] for m in matching],
        "strength": strength,
    }


def _heuristic_verdict(pros: dict, defn: dict, evidence_log: list[dict]) -> dict:
    strength_score = {"WEAK": 1, "MODERATE": 2, "STRONG": 3}
    p_score = strength_score[pros["strength"]]
    d_score = strength_score[defn["strength"]]
    fraud_signals = sum(1 for e in evidence_log if e["signal"] == "fraud")
    benign_signals = sum(1 for e in evidence_log if e["signal"] == "benign")

    if p_score > d_score and fraud_signals >= 2:
        rec, leaning = "fraud", "HIGH" if p_score == 3 and d_score == 1 else "MED"
    elif d_score > p_score and benign_signals >= 2:
        rec, leaning = "legitimate", "HIGH" if d_score == 3 and p_score == 1 else "MED"
    elif fraud_signals == 0 and benign_signals == 0:
        rec, leaning = "uncertain", "LOW"
    else:
        rec, leaning = "uncertain", "LOW"

    reasoning = (
        f"Prosecutor strength={pros['strength']} ({fraud_signals} fraud-signal evidence items); "
        f"Defender strength={defn['strength']} ({benign_signals} benign-signal evidence items). "
        f"Heuristic judge leans '{rec}' at {leaning} confidence based on which side's evidence "
        f"count and argument strength dominate."
    )
    return {"recommendation": rec, "reasoning": reasoning, "confidence_leaning": leaning}


def run_court_panel(evidence_log: list[dict], simulate_failure: bool = False) -> DebateRecord:
    evidence_text = _format_evidence(evidence_log)

    try:
        pros = call_llm_json(PROSECUTOR_SYSTEM, f"Evidence log:\n{evidence_text}", simulate_failure=simulate_failure)
        defn = call_llm_json(DEFENDER_SYSTEM, f"Evidence log:\n{evidence_text}", simulate_failure=simulate_failure)
        verdict = call_llm_json(
            JUDGE_SYSTEM,
            f"Evidence log:\n{evidence_text}\n\nProsecutor argument: {json.dumps(pros)}\n\nDefender argument: {json.dumps(defn)}",
            simulate_failure=simulate_failure,
        )
        return DebateRecord(prosecutor_argument=pros, defender_argument=defn, verdict=verdict, mode="llm")
    except LLMFailure as e:
        pros = _heuristic_argument(evidence_log, "fraud")
        defn = _heuristic_argument(evidence_log, "benign")
        verdict = _heuristic_verdict(pros, defn, evidence_log)
        return DebateRecord(
            prosecutor_argument=pros, defender_argument=defn, verdict=verdict,
            mode="heuristic_fallback", fallback_reason=str(e),
        )
