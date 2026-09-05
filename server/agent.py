"""
The investigating/deciding agent's autopilot policy.

Two modes:
  - "llm":       each next-action choice and the final terminal decision
                 are produced by an LLM call, constrained to the bounded
                 action set and validated before being applied.
  - "heuristic": a deterministic evidence-scoring policy used whenever no
                 LLM key is configured, or whenever an LLM call fails
                 validation - this is what keeps the whole app runnable
                 and demoable with zero external credentials.

Malformed LLM output (action outside the bounded set, missing confidence
on a terminal action, unparseable JSON) is REJECTED and the step falls
back to the heuristic policy for that single step - the environment
never lets an agent "invent" an action, and a bad LLM turn never crashes
the episode.
"""
from __future__ import annotations
from server.case_generator import EVIDENCE_TOOLS
from server.llm_client import call_llm_json, LLMFailure
from server import audit_memory

TERMINAL_ACTIONS = ["flag_fraud", "allow_transaction", "escalate_to_review"]
CONFIDENCE_LEVELS = ["HIGH", "MED", "LOW"]

AGENT_SYSTEM = (
    "You are a fraud/chargeback investigation agent. You will be given the case so far and "
    "the evidence gathered. Choose exactly ONE next action from the allowed set. Investigate "
    "before deciding; use convene_debate_panel when evidence conflicts; only take a terminal "
    "action once you have gathered enough evidence (ideally after using convene_debate_panel "
    "when evidence is mixed). "
    'Return JSON: {"action": str, "confidence": "HIGH"|"MED"|"LOW"|null, "rationale": str}. '
    "confidence is REQUIRED (non-null) if and only if action is a terminal action "
    "(flag_fraud, allow_transaction, escalate_to_review); otherwise it must be null."
)


def _heuristic_next_action(case_public: dict, evidence_log: list[dict], panel_used: bool, debate_record: dict | None = None) -> dict:
    called_tools = {e["tool"] for e in evidence_log}
    remaining = [t for t in EVIDENCE_TOOLS if t not in called_tools]

    fraud_signals = sum(1 for e in evidence_log if e["signal"] == "fraud")
    benign_signals = sum(1 for e in evidence_log if e["signal"] == "benign")

    # deliberately thorough: investigate every available tool before
    # deciding, full stop. A single early red-herring signal (or an early
    # run of benign-looking tools) must never end the investigation
    # prematurely - this is exactly what the friendly-fraud case needs,
    # since its one real tell is not always among the first tools called.
    if remaining:
        return {"action": remaining[0], "confidence": None, "rationale": "Heuristic policy: gathering evidence before deciding (investigates all available tools)."}

    mixed = fraud_signals > 0 and benign_signals > 0
    if mixed and not panel_used:
        return {"action": "convene_debate_panel", "confidence": None, "rationale": "Heuristic policy: evidence is mixed, convening Court Panel before deciding."}

    # If the Court Panel was convened, its verdict - not a raw evidence
    # count - drives the terminal call. This is the point of the panel:
    # it can surface that a single, well-cited fraud signal outweighs
    # several benign ones (exactly the friendly-fraud shape), which a
    # naive majority-count would miss entirely.
    if panel_used and debate_record is not None:
        verdict = debate_record["verdict"]
        rec = verdict.get("recommendation")
        leaning = verdict.get("confidence_leaning", "MED")
        if rec == "fraud":
            return {"action": "flag_fraud", "confidence": leaning, "rationale": f"Heuristic policy: following Court Panel verdict ({verdict.get('reasoning','')})"}
        if rec == "legitimate":
            return {"action": "allow_transaction", "confidence": leaning, "rationale": f"Heuristic policy: following Court Panel verdict ({verdict.get('reasoning','')})"}
        return {"action": "escalate_to_review", "confidence": "LOW", "rationale": f"Heuristic policy: Court Panel verdict was uncertain ({verdict.get('reasoning','')}); escalating."}

    # decide (no panel was needed - evidence wasn't mixed)
    if fraud_signals >= 3 and fraud_signals > benign_signals:
        confidence = "HIGH" if fraud_signals >= 4 and benign_signals == 0 else "MED"
        return {"action": "flag_fraud", "confidence": confidence, "rationale": f"{fraud_signals} fraud-signal evidence items outweigh {benign_signals} benign ones."}
    if benign_signals >= 3 and benign_signals > fraud_signals:
        confidence = "HIGH" if benign_signals >= 4 and fraud_signals == 0 else "MED"
        return {"action": "allow_transaction", "confidence": confidence, "rationale": f"{benign_signals} benign-signal evidence items outweigh {fraud_signals} fraud ones."}
    if fraud_signals == benign_signals == 0:
        return {"action": "escalate_to_review", "confidence": "LOW", "rationale": "No decisive evidence surfaced; escalating for human review."}

    return {"action": "escalate_to_review", "confidence": "MED", "rationale": f"Evidence genuinely mixed ({fraud_signals} fraud vs {benign_signals} benign signals); escalating rather than guessing."}


def _validate_llm_action(parsed: dict) -> bool:
    action = parsed.get("action")
    confidence = parsed.get("confidence")
    valid_actions = EVIDENCE_TOOLS + ["convene_debate_panel"] + TERMINAL_ACTIONS
    if action not in valid_actions:
        return False
    if action in TERMINAL_ACTIONS:
        return confidence in CONFIDENCE_LEVELS
    return confidence is None


def choose_next_action(case_public: dict, evidence_log: list[dict], panel_used: bool, debate_record: dict | None = None, simulate_failure: bool = False) -> dict:
    try:
        context = {
            "case": case_public,
            "evidence_gathered_so_far": evidence_log,
            "court_panel_already_used": panel_used,
            "court_panel_verdict": debate_record["verdict"] if debate_record else None,
        }
        parsed = call_llm_json(AGENT_SYSTEM, str(context), simulate_failure=simulate_failure)
        if not _validate_llm_action(parsed):
            audit_memory.log_failure(
                case_public.get("id"), "agent.choose_next_action",
                f"LLM returned an invalid/malformed action: {parsed}",
                "Rejected the malformed step and fell back to the deterministic heuristic policy for this turn.",
                "Recovered - episode continued without interruption.",
            )
            return {**_heuristic_next_action(case_public, evidence_log, panel_used, debate_record), "mode": "heuristic_fallback"}
        return {**parsed, "mode": "llm"}
    except LLMFailure as e:
        # a missing API key is "demo mode", not an incident - only log real
        # failures (network errors, bad JSON) or explicitly injected ones
        # to the failures ledger, otherwise every heuristic step would spam it
        is_real_incident = simulate_failure or "No GEMINI_API_KEY" not in str(e)
        if is_real_incident:
            audit_memory.log_failure(
                case_public.get("id"), "agent.choose_next_action",
                f"LLM call failed: {e}",
                "Fell back to the deterministic heuristic policy for this turn.",
                "Recovered - episode continued without interruption.",
            )
        return {**_heuristic_next_action(case_public, evidence_log, panel_used, debate_record), "mode": "heuristic_fallback"}


CORRECTION_SUMMARY_SYSTEM = (
    "Summarize this auditor correction into one crisp sentence a future risk analyst could "
    "scan in under 3 seconds, capturing WHY the auditor overturned or confirmed the decision. "
    'Return JSON: {"summary": str}'
)


def summarize_correction(reason_text: str, simulate_failure: bool = False) -> str:
    """
    Second, distinct AI use: turning free-text auditor reasoning into a
    compact, reusable summary for the Knowledge Base. Falls back to a
    simple truncation heuristic if the LLM is unavailable.
    """
    try:
        parsed = call_llm_json(CORRECTION_SUMMARY_SYSTEM, reason_text, simulate_failure=simulate_failure)
        summary = parsed.get("summary")
        if isinstance(summary, str) and summary.strip():
            return summary.strip()
        raise LLMFailure("LLM returned empty summary")
    except LLMFailure:
        return (reason_text[:140] + "...") if len(reason_text) > 140 else reason_text
