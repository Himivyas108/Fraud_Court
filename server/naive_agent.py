"""
Naive single-shot baseline agent - deliberately shallow, used ONLY for
the ablation comparison (see /run_ablation). It looks at the case's
visible metadata alone (no investigation, no debate) and immediately
commits to a terminal decision at HIGH confidence, the way a naive
"prompt an LLM once and take its answer" classifier would behave.

This exists to make the "why does this need the extra machinery"
argument empirical instead of architectural: run the same held-out
batch through this and through the full investigate->debate->calibrate
pipeline, and compare calibration scores directly.
"""
from __future__ import annotations
from server.case_generator import Case


def naive_decide(case: Case) -> dict:
    """Zero investigation. Pattern-matches on surface metadata only, and is
    always maximally confident - this is the overconfidence failure mode
    the whole project exists to fix, reproduced on purpose as a baseline."""
    reason = case.dispute_reason_code.lower()
    suspicious_keywords = ("fraud", "not received", "dispute")
    looks_suspicious = any(k in reason for k in suspicious_keywords) or case.amount > 50000

    if looks_suspicious:
        return {"action": "flag_fraud", "confidence": "HIGH", "rationale": "Naive baseline: surface metadata looked suspicious, single-shot HIGH-confidence flag with no investigation."}
    return {"action": "allow_transaction", "confidence": "HIGH", "rationale": "Naive baseline: surface metadata looked clean, single-shot HIGH-confidence allow with no investigation."}
