"""
FraudCourt FastAPI server.

Implements an OpenEnv-inspired surface (/reset /step /state /tasks
/health /schema) plus the product-facing endpoints needed for the
dashboard: held-out batch evaluation, human-in-the-loop audit feedback,
the Tier-A knowledge base, a failures ledger, and a "Break It" debug
endpoint that lets a judge inject a live failure and watch the system
recover.
"""
from __future__ import annotations
import os
import time
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse

from app.schemas import ResetRequest, StepRequest, AuditFeedbackRequest, RunBatchRequest, InjectFailureRequest
from server import engine, episode_store, audit_memory, agent, court_panel
from server.db import init_db
from server.case_generator import FRAUD_TYPES, CATEGORIES, EVIDENCE_TOOLS
from server.llm_client import LLM_ENABLED, GEMINI_MODEL

FRONTEND_DIR = os.path.join(os.path.dirname(__file__), "..", "frontend")

from contextlib import asynccontextmanager


@asynccontextmanager
async def _lifespan(app: FastAPI):
    init_db()
    yield


app = FastAPI(
    title="FraudCourt",
    description="Calibrated fraud & chargeback adjudication environment - AI Risk Manager track.",
    version="1.0.0",
    lifespan=_lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


TASKS = {
    "contradictory_dispute": {"description": "Mixed evidence, forces the Court Panel to fire.", "force_fraud_type": "friendly_fraud", "force_label": "fraud"},
    "friendly_fraud_showcase": {"description": "Real card, real customer, disputed intent - the case a rules engine cannot catch.", "force_fraud_type": "friendly_fraud", "force_label": None},
    "clear_stolen_card": {"description": "Strong, consistent fraud signals across tools.", "force_fraud_type": "stolen_card", "force_label": "fraud"},
    "clear_legitimate": {"description": "Strong, consistent benign signals across tools.", "force_fraud_type": None, "force_label": "legitimate"},
    "random": {"description": "Fully random procedurally generated case.", "force_fraud_type": None, "force_label": None},
}


def _episode_or_404(episode_id: str) -> episode_store.Episode:
    ep = episode_store.get_episode(episode_id)
    if ep is None:
        raise HTTPException(404, f"Unknown episode_id: {episode_id}")
    return ep


def _episode_public_state(ep: episode_store.Episode) -> dict:
    return {
        "episode_id": ep.episode_id,
        "case": ep.case.public_dict(),
        "evidence_log": ep.evidence_log,
        "debate_record": ep.debate_record,
        "decision": ep.decision,
        "score": ep.score,
        "status": ep.status,
        "trace": ep.trace,
        "needs_audit": getattr(ep, "needs_audit", False),
    }


# --- OpenEnv-standard surface -------------------------------------------

@app.get("/health")
def health():
    return {"status": "ok", "llm_enabled": LLM_ENABLED, "model": GEMINI_MODEL if LLM_ENABLED else "heuristic-fallback-mode"}


@app.get("/schema")
def schema():
    return {
        "spec_version": 1,
        "environment": "FraudCourt",
        "investigative_tools": EVIDENCE_TOOLS,
        "terminal_actions": ["flag_fraud", "allow_transaction", "escalate_to_review"],
        "confidence_levels": ["HIGH", "MED", "LOW"],
        "fraud_types": FRAUD_TYPES,
        "categories": CATEGORIES,
        "reward_matrix": {
            "HIGH/correct": 1.0, "HIGH/wrong": -0.8,
            "MED/correct": 0.6, "MED/wrong": -0.2,
            "LOW/correct": 0.1, "LOW/wrong": 0.0,
        },
    }


@app.get("/tasks")
def tasks():
    return {"tasks": [{"task_id": k, **v} for k, v in TASKS.items()]}


@app.post("/reset")
def reset(req: ResetRequest):
    task = TASKS.get(req.task_id) if req.task_id else None
    seed = req.seed if req.seed is not None else int(time.time() * 1000) % 1_000_000
    force_fraud_type = task["force_fraud_type"] if task else None
    force_label = task["force_label"] if task else None
    ep = engine.reset_episode(seed=seed, force_fraud_type=force_fraud_type, force_label=force_label)
    return _episode_public_state(ep)


@app.post("/step")
def step(req: StepRequest):
    ep = _episode_or_404(req.episode_id)
    try:
        if req.action in ("flag_fraud", "allow_transaction", "escalate_to_review"):
            if not req.confidence:
                raise HTTPException(400, "confidence (HIGH|MED|LOW) is required for a terminal action")
            engine.step_terminal(ep, req.action, req.confidence)
        elif req.action == "convene_debate_panel":
            engine.step_convene_panel(ep, simulate_failure=req.simulate_failure)
        else:
            engine.step_investigate(ep, req.action)
    except ValueError as e:
        raise HTTPException(400, str(e))
    return _episode_public_state(ep)


@app.get("/state")
def state(episode_id: str):
    ep = _episode_or_404(episode_id)
    return _episode_public_state(ep)


# --- Autopilot (agent-driven full run, for the "Run Episode" button) ---

@app.post("/episodes/{episode_id}/autopilot")
def autopilot(episode_id: str, simulate_failure: bool = False):
    ep = _episode_or_404(episode_id)
    if ep.status != "active":
        return _episode_public_state(ep)
    engine.run_autopilot(ep, simulate_failure=simulate_failure)
    return _episode_public_state(ep)


@app.post("/run_autopilot_episode")
def run_autopilot_episode(req: ResetRequest, simulate_failure: bool = False):
    """Convenience: reset + autopilot in one call, for the live demo runner."""
    task = TASKS.get(req.task_id) if req.task_id else None
    seed = req.seed if req.seed is not None else int(time.time() * 1000) % 1_000_000
    force_fraud_type = task["force_fraud_type"] if task else None
    force_label = task["force_label"] if task else None
    ep = engine.reset_episode(seed=seed, force_fraud_type=force_fraud_type, force_label=force_label)
    engine.run_autopilot(ep, simulate_failure=simulate_failure)
    return _episode_public_state(ep)


# --- Held-out batch evaluation -------------------------------------------

@app.post("/run_batch")
def run_batch(req: RunBatchRequest):
    from server.batch import run_held_out_batch
    report = run_held_out_batch(n=req.n, seed_start=req.seed_start, use_golden_trap=req.use_golden_trap)
    audit_memory.save_batch_report(report["run_id"], report)
    reports_dir = os.path.join(os.path.dirname(__file__), "..", "reports")
    os.makedirs(reports_dir, exist_ok=True)
    import json
    fname = "golden_trap_summary.json" if req.use_golden_trap else "component_shift_summary.json"
    with open(os.path.join(reports_dir, fname), "w") as f:
        json.dump(report, f, indent=2)
    return report


@app.get("/run_ablation")
def run_ablation(n: int = 30, seed_start: int = 2000):
    """Naive single-shot baseline vs. the full pipeline, same seeds - see server/batch.py."""
    from server.batch import run_ablation as _run_ablation
    result = _run_ablation(n=n, seed_start=seed_start)
    import json
    reports_dir = os.path.join(os.path.dirname(__file__), "..", "reports")
    os.makedirs(reports_dir, exist_ok=True)
    with open(os.path.join(reports_dir, "ablation_summary.json"), "w") as f:
        json.dump(result, f, indent=2)
    return result


@app.get("/report")
def report():
    r = audit_memory.latest_batch_report()
    if r is None:
        raise HTTPException(404, "No batch report has been generated yet. POST /run_batch first.")
    return r


# --- Human-in-the-loop audit -------------------------------------------

@app.get("/audit_queue")
def audit_queue():
    pending = [
        _episode_public_state(ep)
        for ep in episode_store.all_terminal_episodes()
        if getattr(ep, "needs_audit", False)
    ]
    return {"pending": pending, "count": len(pending)}


@app.post("/cases/{episode_id}/audit_feedback")
def audit_feedback(episode_id: str, req: AuditFeedbackRequest):
    ep = _episode_or_404(episode_id)
    if ep.status != "terminal":
        raise HTTPException(400, "Cannot audit a case that has not reached a terminal decision yet.")
    if not req.reason or not req.reason.strip():
        raise HTTPException(400, "A reason is required for every auditor decision.")

    evidence_sig = ep.score["evidence_signature"]
    record = audit_memory.record_correction(
        case_id=ep.case.id,
        original_decision=ep.decision["action"],
        original_confidence=ep.decision["confidence"],
        auditor_decision=req.auditor_decision,
        reason_text=req.reason.strip(),
        evidence_signature=evidence_sig,
    )
    summary = agent.summarize_correction(req.reason.strip())
    record["summary"] = summary
    ep.needs_audit = False
    ep.trace.append({"type": "audit_feedback", "record": record, "ts": time.time()})
    return record


@app.get("/knowledge_base")
def knowledge_base():
    return {"entries": audit_memory.knowledge_base_entries()}


@app.get("/corrections")
def corrections():
    return {"corrections": audit_memory.all_corrections()}


# --- Failures ledger + live "Break It" demo button ----------------------

@app.get("/failures")
def failures():
    return {"failures": audit_memory.all_failures()}


@app.post("/debug/inject_failure")
def inject_failure(req: InjectFailureRequest):
    """
    Lets a judge press a button and watch the system recover live: forces
    the next Court Panel or agent-decision call for this episode's
    remaining flow to simulate an LLM failure, so the fallback path fires
    visibly, and logs an auto-generated incident record.
    """
    ep = _episode_or_404(req.episode_id)
    if req.component == "court_panel":
        result = engine.step_convene_panel(ep, simulate_failure=True) if ep.status == "active" else None
    else:
        decision = agent.choose_next_action(ep.case.public_dict(), ep.evidence_log, ep.debate_record is not None, simulate_failure=True)
        result = decision
    failure_record = audit_memory.log_failure(
        ep.case.id, req.component,
        "Manually injected failure via /debug/inject_failure (live demo).",
        "System caught the failure and used the documented deterministic fallback instead of crashing or guessing.",
        "Recovered live - episode continued.",
    )
    return {"episode": _episode_public_state(ep), "injected_result": result, "failure_record": failure_record}


# --- Frontend static hosting ---------------------------------------------

@app.get("/")
def root():
    index_path = os.path.join(FRONTEND_DIR, "index.html")
    return FileResponse(index_path)


app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")
