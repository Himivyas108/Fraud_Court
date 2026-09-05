from server import engine, episode_store, anti_gaming


def test_reset_creates_active_episode():
    ep = engine.reset_episode(seed=10)
    assert ep.status == "active"
    assert ep.case.hidden_ground_truth_label in ("fraud", "legitimate")


def test_investigate_reveals_evidence_and_is_idempotent():
    ep = engine.reset_episode(seed=11)
    r1 = engine.step_investigate(ep, "check_velocity")
    assert len(ep.evidence_log) == 1
    r2 = engine.step_investigate(ep, "check_velocity")  # duplicate call
    assert len(ep.evidence_log) == 1  # no-op, not appended twice
    assert r1 == r2


def test_terminal_decision_scores_and_closes_episode():
    ep = engine.reset_episode(seed=12, force_fraud_type="stolen_card", force_label="fraud")
    for tool in ["check_velocity", "check_device_fingerprint", "query_transaction_history"]:
        engine.step_investigate(ep, tool)
    engine.step_terminal(ep, "flag_fraud", "HIGH")
    assert ep.status == "terminal"
    assert ep.score is not None
    assert ep.score["ground_truth_label"] == "fraud"


def test_cannot_step_a_terminal_episode():
    ep = engine.reset_episode(seed=13)
    engine.step_terminal(ep, "escalate_to_review", "LOW")
    import pytest
    with pytest.raises(ValueError):
        engine.step_investigate(ep, "check_velocity")


def test_autopilot_always_reaches_a_terminal_decision():
    anti_gaming.GLOBAL_DETECTOR.reset()
    ep = engine.reset_episode(seed=14)
    engine.run_autopilot(ep)
    assert ep.status == "terminal"
    assert ep.decision["confidence"] in ("HIGH", "MED", "LOW")


def test_golden_trap_cases_all_reach_a_decision():
    from server.case_generator import golden_trap_cases
    anti_gaming.GLOBAL_DETECTOR.reset()
    for case in golden_trap_cases():
        ep = episode_store.create_episode(case)
        engine.run_autopilot(ep)
        assert ep.status == "terminal"
        assert ep.score["matrix_cell"] in (
            "HIGH/CORRECT", "HIGH/WRONG", "MED/CORRECT", "MED/WRONG", "LOW/CORRECT", "LOW/WRONG"
        )
