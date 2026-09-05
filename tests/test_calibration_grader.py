import pytest
from server.calibration_grader import grade_decision, MATRIX


def test_matrix_values_match_spec():
    assert MATRIX[("HIGH", True)] == 1.0
    assert MATRIX[("HIGH", False)] == -0.8
    assert MATRIX[("MED", True)] == 0.6
    assert MATRIX[("MED", False)] == -0.2
    assert MATRIX[("LOW", True)] == 0.1
    assert MATRIX[("LOW", False)] == 0.0


def test_high_confidence_correct_flag():
    r = grade_decision("flag_fraud", "HIGH", "fraud", amount=10000)
    assert r.is_correct is True
    assert r.reward == 1.0
    assert r.cost_of_overconfidence_inr == 0.0


def test_high_confidence_wrong_flag_incurs_cost():
    r = grade_decision("flag_fraud", "HIGH", "legitimate", amount=10000)
    assert r.is_correct is False
    assert r.reward == -0.8
    assert r.cost_of_overconfidence_inr == pytest.approx(2000.0)  # 20% of 10000


def test_low_confidence_wrong_is_the_safe_zero_cell():
    r = grade_decision("allow_transaction", "LOW", "fraud", amount=99999)
    assert r.is_correct is False
    assert r.reward == 0.0
    assert r.cost_of_overconfidence_inr == 0.0  # only HIGH-wrong incurs cost


def test_escalation_correct_when_case_is_fraud():
    r = grade_decision("escalate_to_review", "MED", "fraud", amount=5000)
    assert r.is_correct is True


def test_invalid_confidence_rejected():
    with pytest.raises(ValueError):
        grade_decision("flag_fraud", "SUPER_HIGH", "fraud", amount=100)


def test_invalid_action_rejected():
    with pytest.raises(ValueError):
        grade_decision("deny_and_ban", "HIGH", "fraud", amount=100)
