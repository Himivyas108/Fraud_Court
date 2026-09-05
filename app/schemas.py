from __future__ import annotations
from pydantic import BaseModel
from typing import Optional, Literal


class ResetRequest(BaseModel):
    seed: Optional[int] = None
    task_id: Optional[str] = None   # e.g. "contradictory_dispute" / "friendly_fraud_showcase"


class StepRequest(BaseModel):
    episode_id: str
    action: str
    confidence: Optional[Literal["HIGH", "MED", "LOW"]] = None
    simulate_failure: bool = False


class AuditFeedbackRequest(BaseModel):
    auditor_decision: Literal["flag_fraud", "allow_transaction", "escalate_to_review"]
    reason: str


class RunBatchRequest(BaseModel):
    n: int = 40
    seed_start: int = 1000
    use_golden_trap: bool = False


class InjectFailureRequest(BaseModel):
    episode_id: str
    component: Literal["agent", "court_panel"] = "court_panel"
