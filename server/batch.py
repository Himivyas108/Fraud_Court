"""
Held-out batch evaluation. Runs a fixed, seed-reproducible set of cases
through the full pipeline and aggregates precision, recall, calibration
score, false-positive cost by confidence tier, and total cost of
overconfidence avoided/incurred. Writes committed JSON - no hand-edits;
this same function backs both the /run_batch API endpoint and the CLI
script, so there's exactly one code path producing these numbers.
"""
from __future__ import annotations
import time
import uuid
from server import engine, episode_store, calibration_grader, anti_gaming
from server.case_generator import golden_trap_cases
from server.naive_agent import naive_decide


def run_held_out_batch(n: int = 40, seed_start: int = 1000, use_golden_trap: bool = False, policy: str = "full") -> dict:
    """
    policy: "full"  -> the actual FraudCourt pipeline (investigate -> debate -> calibrate)
            "naive"  -> single-shot baseline, zero investigation, always HIGH confidence
                        (see server/naive_agent.py). Used by /run_ablation for the
                        empirical AI-judgment comparison.
    """
    run_id = f"run_{uuid.uuid4().hex[:8]}"
    started = time.time()
    anti_gaming.GLOBAL_DETECTOR.reset()  # isolate each batch run's gaming stats

    episodes: list[episode_store.Episode] = []

    if use_golden_trap:
        cases = golden_trap_cases()
        for c in cases:
            ep = episode_store.create_episode(c)
            episodes.append(ep)
    else:
        for i in range(n):
            ep = engine.reset_episode(seed=seed_start + i)
            episodes.append(ep)

    if policy == "naive":
        for ep in episodes:
            decision = naive_decide(ep.case)
            engine.step_terminal(ep, decision["action"], decision["confidence"])
    else:
        for ep in episodes:
            engine.run_autopilot(ep)

    tp = fp = tn = fn = 0
    calibration_correct_by_conf = {"HIGH": [0, 0], "MED": [0, 0], "LOW": [0, 0]}  # [correct_count, total_count]
    cost_by_tier = {"HIGH": 0.0, "MED": 0.0, "LOW": 0.0}
    total_cost_of_overconfidence = 0.0
    llm_mode_count = 0
    heuristic_mode_count = 0

    for ep in episodes:
        gt = ep.case.hidden_ground_truth_label
        action = ep.decision["action"]
        conf = ep.decision["confidence"]
        is_correct = ep.score["is_correct"]

        predicted_fraud = action == "flag_fraud"
        actual_fraud = gt == "fraud"
        if predicted_fraud and actual_fraud:
            tp += 1
        elif predicted_fraud and not actual_fraud:
            fp += 1
        elif not predicted_fraud and not actual_fraud:
            tn += 1
        elif not predicted_fraud and actual_fraud:
            fn += 1

        calibration_correct_by_conf[conf][1] += 1
        if is_correct:
            calibration_correct_by_conf[conf][0] += 1

        cost_by_tier[conf] += ep.score["cost_of_overconfidence_inr"]
        total_cost_of_overconfidence += ep.score["cost_of_overconfidence_inr"]

        for step in ep.trace:
            if step.get("type") == "court_panel":
                if step["record"]["mode"] == "llm":
                    llm_mode_count += 1
                else:
                    heuristic_mode_count += 1

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    # Calibration score: for each confidence tier, the fraction correct
    # SHOULD roughly match the "meaning" of that tier (HIGH should be
    # correct nearly always, LOW is allowed to be uncertain). We report
    # the overall weighted correctness-consistency as a single 0-1 score:
    # 1.0 = HIGH decisions are correct ~100% of the time, MED ~60-80%,
    # LOW can be anything (it's the "safe hedge" tier by design).
    def _tier_rate(tier):
        correct, total = calibration_correct_by_conf[tier]
        return (correct / total) if total else None

    high_rate = _tier_rate("HIGH")
    med_rate = _tier_rate("MED")
    calibration_components = [r for r in (high_rate, med_rate) if r is not None]
    calibration_score = round(sum(calibration_components) / len(calibration_components), 3) if calibration_components else 0.0
    if high_rate is not None:
        # HIGH being wrong often is penalised harder in the composite score
        calibration_score = round((high_rate * 0.7 + (med_rate or 0) * 0.3) if med_rate is not None else high_rate, 3)

    report = {
        "run_id": run_id,
        "policy": policy,
        "n_cases": len(episodes),
        "mode": "golden_trap" if use_golden_trap else "random_held_out",
        "seed_start": None if use_golden_trap else seed_start,
        "generated_at": time.time(),
        "duration_seconds": round(time.time() - started, 2),
        "confusion_matrix": {"true_positive": tp, "false_positive": fp, "true_negative": tn, "false_negative": fn},
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "f1": round(f1, 3),
        "calibration_score": calibration_score,
        "calibration_by_tier": {
            tier: {"correct": calibration_correct_by_conf[tier][0], "total": calibration_correct_by_conf[tier][1],
                   "rate": _tier_rate(tier)}
            for tier in ("HIGH", "MED", "LOW")
        },
        "false_positive_cost_by_tier_inr": {k: round(v, 2) for k, v in cost_by_tier.items()},
        "total_cost_of_overconfidence_inr": round(total_cost_of_overconfidence, 2),
        "court_panel_llm_calls": llm_mode_count,
        "court_panel_heuristic_fallback_calls": heuristic_mode_count,
        "case_ids": [ep.case.id for ep in episodes],
    }
    return report


def run_ablation(n: int = 30, seed_start: int = 2000) -> dict:
    """
    Runs the SAME held-out seed range through both policies and returns
    both reports side by side - the empirical answer to "why does this
    need this much AI machinery, specifically."
    """
    naive_report = run_held_out_batch(n=n, seed_start=seed_start, policy="naive")
    full_report = run_held_out_batch(n=n, seed_start=seed_start, policy="full")
    return {
        "n_cases": n,
        "seed_start": seed_start,
        "naive_single_shot": naive_report,
        "full_pipeline": full_report,
        "calibration_score_delta": round(full_report["calibration_score"] - naive_report["calibration_score"], 3),
        "cost_of_overconfidence_delta_inr": round(
            naive_report["total_cost_of_overconfidence_inr"] - full_report["total_cost_of_overconfidence_inr"], 2
        ),
    }
