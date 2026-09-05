"""
In-memory store for active episodes (OpenEnv-style session state).
Kept in-process for simplicity - documented limitation: episodes do not
survive a server restart. Durable artifacts (corrections, failures,
batch reports) live in SQLite via audit_memory.py / db.py instead.
"""
from __future__ import annotations
import time
import uuid
from dataclasses import dataclass, field
from server.case_generator import Case

_EPISODES: dict[str, "Episode"] = {}


@dataclass
class Episode:
    episode_id: str
    case: Case
    evidence_log: list = field(default_factory=list)   # [{tool, result, signal, timestamp}]
    debate_record: dict | None = None
    decision: dict | None = None       # {action, confidence, timestamp}
    score: dict | None = None
    status: str = "active"             # active | terminal
    trace: list = field(default_factory=list)          # human-readable step log for the UI
    created_at: float = field(default_factory=time.time)


def create_episode(case: Case) -> Episode:
    ep = Episode(episode_id=f"ep_{uuid.uuid4().hex[:12]}", case=case)
    _EPISODES[ep.episode_id] = ep
    return ep


def get_episode(episode_id: str) -> Episode | None:
    return _EPISODES.get(episode_id)


def all_active_episodes() -> list[Episode]:
    return [e for e in _EPISODES.values() if e.status == "active"]


def all_terminal_episodes() -> list[Episode]:
    return [e for e in _EPISODES.values() if e.status == "terminal"]
