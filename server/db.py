"""
Lightweight SQLite persistence layer. No ORM needed for this scope -
plain sqlite3 with a couple of small tables. This is the durable store
for: human-in-the-loop corrections, the Tier-A structured-key-match
knowledge base, and the failures ledger (auto-postmortems).
"""
from __future__ import annotations
import sqlite3
import os
import threading

DB_PATH = os.getenv("FRAUDCOURT_DB_PATH", os.path.join(os.path.dirname(__file__), "..", "data", "fraudcourt.db"))
os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

_local = threading.local()


def get_conn() -> sqlite3.Connection:
    if not hasattr(_local, "conn"):
        _local.conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        _local.conn.row_factory = sqlite3.Row
    return _local.conn


def init_db():
    conn = get_conn()
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS corrections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            case_id TEXT NOT NULL,
            original_decision TEXT,
            original_confidence TEXT,
            auditor_decision TEXT NOT NULL,
            reason_text TEXT NOT NULL,
            evidence_signature TEXT,
            was_overturned INTEGER NOT NULL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS failures (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            case_id TEXT,
            component TEXT NOT NULL,
            what_broke TEXT NOT NULL,
            what_we_did TEXT NOT NULL,
            outcome TEXT NOT NULL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS batch_reports (
            run_id TEXT PRIMARY KEY,
            report_json TEXT NOT NULL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        );
        """
    )
    conn.commit()
