"""
Human-in-the-loop auditor feedback + Tier-A Knowledge Base (structured
key-match precedent lookup) + failures ledger.

Design choices, stated explicitly (per the project's "boring and
explainable over fancy and opaque" instinct):
  - The auditor override itself is NEVER mediated by AI. It is a hard
    human decision, unmediated by a model, or it isn't really HITL.
  - Precedent lookup is a deterministic string/key match over an
    `evidence_signature`, not a vector embedding store. This is Tier A
    from the design doc: zero dependencies, fully explainable to a judge
    who asks "how does it find similar cases?" - you can show them the
    exact match, no black box. (Tier B / ChromaDB is a documented,
    optional stretch upgrade - see README.)
"""
from __future__ import annotations
import json
from dataclasses import dataclass
from server.db import get_conn


def build_evidence_signature(fraud_type: str, evidence_log: list[dict]) -> str:
    """Compact, reusable key: fraud_type + which signal-bearing tools fired."""
    fraud_tools = sorted(e["tool"] for e in evidence_log if e["signal"] == "fraud")
    benign_tools = sorted(e["tool"] for e in evidence_log if e["signal"] == "benign")
    return f"{fraud_type}|fraud:{','.join(fraud_tools) or 'none'}|benign:{','.join(benign_tools) or 'none'}"


def record_correction(case_id: str, original_decision: str, original_confidence: str,
                       auditor_decision: str, reason_text: str, evidence_signature: str) -> dict:
    was_overturned = int(auditor_decision != original_decision)
    conn = get_conn()
    cur = conn.execute(
        """INSERT INTO corrections
           (case_id, original_decision, original_confidence, auditor_decision,
            reason_text, evidence_signature, was_overturned)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (case_id, original_decision, original_confidence, auditor_decision,
         reason_text, evidence_signature, was_overturned),
    )
    conn.commit()
    return {
        "id": cur.lastrowid, "case_id": case_id, "original_decision": original_decision,
        "original_confidence": original_confidence, "auditor_decision": auditor_decision,
        "reason_text": reason_text, "evidence_signature": evidence_signature,
        "was_overturned": bool(was_overturned),
    }


def find_precedent(evidence_signature: str, limit: int = 5) -> list[dict]:
    """Exact-key + fuzzy (shared fraud_type prefix) precedent lookup."""
    conn = get_conn()
    rows = conn.execute(
        "SELECT * FROM corrections WHERE evidence_signature = ? ORDER BY created_at DESC LIMIT ?",
        (evidence_signature, limit),
    ).fetchall()
    if not rows:
        fraud_type = evidence_signature.split("|", 1)[0]
        rows = conn.execute(
            "SELECT * FROM corrections WHERE evidence_signature LIKE ? ORDER BY created_at DESC LIMIT ?",
            (f"{fraud_type}|%", limit),
        ).fetchall()
    return [dict(r) for r in rows]


def overturn_rate_for_signature(evidence_signature: str) -> dict:
    precedents = find_precedent(evidence_signature, limit=1000)
    if not precedents:
        return {"n": 0, "overturn_rate": None}
    n = len(precedents)
    overturned = sum(1 for p in precedents if p["was_overturned"])
    return {"n": n, "overturn_rate": round(overturned / n, 3)}


def all_corrections(limit: int = 200) -> list[dict]:
    conn = get_conn()
    rows = conn.execute("SELECT * FROM corrections ORDER BY created_at DESC LIMIT ?", (limit,)).fetchall()
    return [dict(r) for r in rows]


def knowledge_base_entries(limit: int = 200) -> list[dict]:
    """Group corrections by evidence_signature -> a browsable KB view."""
    conn = get_conn()
    rows = conn.execute(
        """SELECT evidence_signature, COUNT(*) as n,
                  SUM(was_overturned) as overturned,
                  MAX(created_at) as last_seen
           FROM corrections GROUP BY evidence_signature ORDER BY last_seen DESC LIMIT ?""",
        (limit,),
    ).fetchall()
    out = []
    for r in rows:
        d = dict(r)
        d["overturn_rate"] = round((d["overturned"] or 0) / d["n"], 3) if d["n"] else 0.0
        out.append(d)
    return out


def log_failure(case_id: str | None, component: str, what_broke: str, what_we_did: str, outcome: str) -> dict:
    conn = get_conn()
    cur = conn.execute(
        "INSERT INTO failures (case_id, component, what_broke, what_we_did, outcome) VALUES (?, ?, ?, ?, ?)",
        (case_id, component, what_broke, what_we_did, outcome),
    )
    conn.commit()
    return {"id": cur.lastrowid, "case_id": case_id, "component": component,
             "what_broke": what_broke, "what_we_did": what_we_did, "outcome": outcome}


def all_failures(limit: int = 200) -> list[dict]:
    conn = get_conn()
    rows = conn.execute("SELECT * FROM failures ORDER BY created_at DESC LIMIT ?", (limit,)).fetchall()
    return [dict(r) for r in rows]


def save_batch_report(run_id: str, report: dict):
    conn = get_conn()
    conn.execute(
        "INSERT OR REPLACE INTO batch_reports (run_id, report_json) VALUES (?, ?)",
        (run_id, json.dumps(report)),
    )
    conn.commit()


def latest_batch_report() -> dict | None:
    conn = get_conn()
    row = conn.execute("SELECT * FROM batch_reports ORDER BY created_at DESC LIMIT 1").fetchone()
    return json.loads(row["report_json"]) if row else None
