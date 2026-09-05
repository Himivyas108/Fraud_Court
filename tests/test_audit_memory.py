from server.audit_memory import (
    build_evidence_signature, record_correction, find_precedent,
    overturn_rate_for_signature, knowledge_base_entries, log_failure, all_failures,
)


def test_evidence_signature_is_deterministic():
    log = [
        {"tool": "check_velocity", "signal": "fraud", "result": "x"},
        {"tool": "check_device_fingerprint", "signal": "benign", "result": "y"},
    ]
    sig1 = build_evidence_signature("stolen_card", log)
    sig2 = build_evidence_signature("stolen_card", log)
    assert sig1 == sig2
    assert "stolen_card" in sig1


def test_record_and_find_precedent_roundtrip():
    sig = build_evidence_signature("refund_abuse", [{"tool": "check_velocity", "signal": "fraud", "result": "x"}])
    record_correction("case_1", "flag_fraud", "HIGH", "allow_transaction", "Customer had a valid reason.", sig)
    matches = find_precedent(sig)
    assert len(matches) == 1
    assert bool(matches[0]["was_overturned"]) is True


def test_overturn_rate_computation():
    sig = build_evidence_signature("account_takeover", [])
    record_correction("case_a", "flag_fraud", "HIGH", "flag_fraud", "Confirmed.", sig)
    record_correction("case_b", "flag_fraud", "HIGH", "allow_transaction", "Overturned.", sig)
    stats = overturn_rate_for_signature(sig)
    assert stats["n"] == 2
    assert stats["overturn_rate"] == 0.5


def test_knowledge_base_groups_by_signature():
    sig = build_evidence_signature("merchant_collusion", [])
    record_correction("c1", "flag_fraud", "MED", "flag_fraud", "ok", sig)
    entries = knowledge_base_entries()
    assert any(e["evidence_signature"] == sig for e in entries)


def test_failures_ledger_records_and_reads():
    log_failure("case_x", "court_panel", "LLM timed out", "fell back to heuristic", "recovered")
    failures = all_failures()
    assert len(failures) >= 1
    assert failures[0]["component"] == "court_panel"
