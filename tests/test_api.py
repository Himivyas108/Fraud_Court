from fastapi.testclient import TestClient
from app.main import app


def test_full_flow_reset_step_terminal():
    with TestClient(app) as client:
        r = client.get("/health")
        assert r.status_code == 200

        r = client.get("/schema")
        assert r.status_code == 200
        assert "reward_matrix" in r.json()

        r = client.get("/tasks")
        assert r.status_code == 200
        assert len(r.json()["tasks"]) >= 1

        r = client.post("/reset", json={"task_id": "clear_stolen_card"})
        assert r.status_code == 200
        ep_id = r.json()["episode_id"]

        r = client.post("/step", json={"episode_id": ep_id, "action": "check_velocity"})
        assert r.status_code == 200
        assert len(r.json()["evidence_log"]) == 1

        r = client.post("/step", json={"episode_id": ep_id, "action": "flag_fraud", "confidence": "HIGH"})
        assert r.status_code == 200
        assert r.json()["status"] == "terminal"


def test_terminal_action_requires_confidence():
    with TestClient(app) as client:
        r = client.post("/reset", json={})
        ep_id = r.json()["episode_id"]
        r = client.post("/step", json={"episode_id": ep_id, "action": "flag_fraud"})
        assert r.status_code == 400


def test_unknown_episode_returns_404():
    with TestClient(app) as client:
        r = client.get("/state", params={"episode_id": "does_not_exist"})
        assert r.status_code == 404


def test_run_batch_and_report_roundtrip():
    with TestClient(app) as client:
        r = client.post("/run_batch", json={"n": 8, "seed_start": 5000})
        assert r.status_code == 200
        assert r.json()["n_cases"] == 8
        r2 = client.get("/report")
        assert r2.status_code == 200
        assert r2.json()["n_cases"] == 8


def test_run_ablation_returns_both_policies():
    with TestClient(app) as client:
        r = client.get("/run_ablation", params={"n": 6, "seed_start": 9000})
        assert r.status_code == 200
        body = r.json()
        assert "naive_single_shot" in body and "full_pipeline" in body


def test_audit_feedback_requires_reason():
    with TestClient(app) as client:
        r = client.post("/run_autopilot_episode", json={"task_id": "contradictory_dispute"})
        ep_id = r.json()["episode_id"]
        r2 = client.post(f"/cases/{ep_id}/audit_feedback", json={"auditor_decision": "flag_fraud", "reason": "   "})
        assert r2.status_code == 400


def test_break_it_button_recovers_live():
    with TestClient(app) as client:
        r = client.post("/reset", json={"task_id": "clear_legitimate"})
        ep_id = r.json()["episode_id"]
        r2 = client.post("/debug/inject_failure", json={"episode_id": ep_id, "component": "court_panel"})
        assert r2.status_code == 200
        assert r2.json()["failure_record"]["outcome"].startswith("Recovered")
        r3 = client.get("/failures")
        assert len(r3.json()["failures"]) >= 1
