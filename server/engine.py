"""
Core environment engine: reset / step / terminal-decision logic, shared
by the FastAPI app (app/main.py) and the offline batch runner
(scripts/run_batch.py) so both paths exercise exactly the same code.
"""
from __future__ import annotations
import time
from server.case_generator import generate_case, Case, golden_trap_cases
from server import episode_store, tools, court_panel, calibration_grader, anti_gaming, audit_memory, agent


def reset_episode(seed: int, force_fraud_type: str | None = None, force_label: str | None = None) -> episode_store.Episode:
    case = generate_case(seed, force_fraud_type=force_fraud_type, force_label=force_label)
    ep = episode_store.create_episode(case)
    ep.trace.append({"type": "reset", "message": f"Case {case.id} loaded (fraud_type={case.fraud_type}, category={case.category}).", "ts": time.time()})
    return ep


def step_investigate(ep: episode_store.Episode, tool_name: str) -> dict:
    if ep.status != "active":
        raise ValueError("Episode is already terminal.")
    already = {e["tool"]: e for e in ep.evidence_log}
    if tool_name in already:
        # idempotent: return the same shape as a fresh call would
        cached = already[tool_name]
        return {"result": cached["result"], "signal": cached["signal"]}
    result = tools.call_tool(ep.case, tool_name, already)
    ep.evidence_log.append({"tool": tool_name, "result": result["result"], "signal": result["signal"], "timestamp": time.time()})
    ep.trace.append({"type": "investigate", "tool": tool_name, "result": result["result"], "ts": time.time()})
    return result


def step_convene_panel(ep: episode_store.Episode, simulate_failure: bool = False) -> dict:
    if ep.status != "active":
        raise ValueError("Episode is already terminal.")
    record = court_panel.run_court_panel(ep.evidence_log, simulate_failure=simulate_failure)
    ep.debate_record = {
        "prosecutor_argument": record.prosecutor_argument,
        "defender_argument": record.defender_argument,
        "verdict": record.verdict,
        "mode": record.mode,
        "fallback_reason": record.fallback_reason,
    }
    ep.trace.append({"type": "court_panel", "record": ep.debate_record, "ts": time.time()})
    if record.mode == "heuristic_fallback":
        is_real_incident = simulate_failure or (record.fallback_reason and "No GEMINI_API_KEY" not in record.fallback_reason)
        if is_real_incident:
            audit_memory.log_failure(
                ep.case.id, "court_panel",
                f"LLM unavailable for Court Panel: {record.fallback_reason}",
                "Fell back to deterministic evidence-strength heuristic for Prosecutor/Defender/Judge.",
                "Recovered - panel verdict still produced, episode continued.",
            )
    return ep.debate_record


def step_terminal(ep: episode_store.Episode, action: str, confidence: str) -> dict:
    if ep.status != "active":
        raise ValueError("Episode is already terminal.")
    if action not in ("flag_fraud", "allow_transaction", "escalate_to_review"):
        raise ValueError(f"Invalid terminal action: {action}")
    if confidence not in ("HIGH", "MED", "LOW"):
        raise ValueError(f"Invalid confidence: {confidence}")

    ep.decision = {"action": action, "confidence": confidence, "timestamp": time.time()}

    result = calibration_grader.grade_decision(action, confidence, ep.case.hidden_ground_truth_label, ep.case.amount)
    gaming = anti_gaming.GLOBAL_DETECTOR.record_and_check(confidence)
    adjusted_reward = round(result.reward * gaming.penalty_multiplier, 4)

    evidence_sig = audit_memory.build_evidence_signature(ep.case.fraud_type, ep.evidence_log)
    precedent = audit_memory.find_precedent(evidence_sig)

    ep.score = {
        "matrix_cell": result.matrix_cell,
        "raw_reward": result.reward,
        "adjusted_reward": adjusted_reward,
        "is_correct": result.is_correct,
        "ground_truth_label": ep.case.hidden_ground_truth_label,
        "cost_of_overconfidence_inr": result.cost_of_overconfidence_inr,
        "explanation": result.explanation,
        "anti_gaming": {
            "triggered": gaming.triggered, "low_rate": round(gaming.low_rate, 3),
            "penalty_multiplier": gaming.penalty_multiplier, "window_size": gaming.window_size,
        },
        "evidence_signature": evidence_sig,
        "precedent_matches": precedent,
    }
    ep.status = "terminal"
    ep.trace.append({"type": "terminal_decision", "action": action, "confidence": confidence, "score": ep.score, "ts": time.time()})

    # auto-queue for human audit: every escalation, plus every HIGH-confidence
    # flag_fraud (the highest-stakes call worth spot-checking)
    ep.needs_audit = action == "escalate_to_review" or (action == "flag_fraud" and confidence == "HIGH")
    return ep.score


def run_autopilot(ep: episode_store.Episode, max_steps: int = 12, simulate_failure: bool = False) -> episode_store.Episode:
    """Drives an episode end-to-end using the agent policy (LLM or heuristic fallback)."""
    panel_used = False
    for _ in range(max_steps):
        if ep.status != "active":
            break
        decision = agent.choose_next_action(ep.case.public_dict(), ep.evidence_log, panel_used, debate_record=ep.debate_record, simulate_failure=simulate_failure)
        action = decision["action"]
        if action in ("flag_fraud", "allow_transaction", "escalate_to_review"):
            step_terminal(ep, action, decision["confidence"])
        elif action == "convene_debate_panel":
            step_convene_panel(ep, simulate_failure=simulate_failure)
            panel_used = True
        else:
            step_investigate(ep, action)
    if ep.status == "active":
        # forced close: ran out of steps without deciding -> escalate, never leave it hanging
        step_terminal(ep, "escalate_to_review", "LOW")
        ep.trace.append({"type": "forced_escalation", "message": "Max steps reached without a decision; auto-escalated at LOW confidence.", "ts": time.time()})
    return ep
