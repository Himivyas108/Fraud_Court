from server.case_generator import generate_case, golden_trap_cases, EVIDENCE_TOOLS, FRAUD_TYPES


def test_determinism_same_seed_identical_case():
    """Same seed must produce a byte-identical case, every time - this is
    what makes the held-out batch reproducible for reviewers."""
    a = generate_case(seed=777)
    b = generate_case(seed=777)
    assert a == b


def test_different_seeds_usually_differ():
    a = generate_case(seed=1)
    b = generate_case(seed=2)
    assert a.id != b.id


def test_forced_fraud_type_and_label_are_respected():
    c = generate_case(seed=55, force_fraud_type="friendly_fraud", force_label="fraud")
    assert c.fraud_type == "friendly_fraud"
    assert c.hidden_ground_truth_label == "fraud"


def test_public_dict_never_leaks_hidden_fields():
    c = generate_case(seed=9)
    pub = c.public_dict()
    assert "hidden_ground_truth_label" not in pub
    assert "hidden_evidence" not in pub


def test_all_evidence_tools_present_for_every_case():
    c = generate_case(seed=3)
    assert set(c.hidden_evidence.keys()) == set(EVIDENCE_TOOLS)


def test_golden_trap_suite_size_and_labels():
    cases = golden_trap_cases()
    assert len(cases) == 15
    fraud_cases = [c for c in cases if c.hidden_ground_truth_label == "fraud"]
    legit_cases = [c for c in cases if c.hidden_ground_truth_label == "legitimate"]
    assert len(fraud_cases) > 0 and len(legit_cases) > 0
    for c in cases:
        assert c.fraud_type in FRAUD_TYPES
