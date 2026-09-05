from server.anti_gaming import AntiGamingDetector


def test_no_trigger_under_threshold():
    d = AntiGamingDetector()
    for _ in range(10):
        result = d.record_and_check("MED")
    assert result.triggered is False
    assert result.penalty_multiplier == 1.0


def test_triggers_when_low_rate_exceeds_threshold():
    d = AntiGamingDetector()
    result = None
    for _ in range(15):
        result = d.record_and_check("LOW")
    assert result.triggered is True
    assert result.penalty_multiplier < 1.0
    assert result.low_rate > 0.70


def test_does_not_trigger_before_minimum_decision_count():
    d = AntiGamingDetector()
    result = None
    for _ in range(5):
        result = d.record_and_check("LOW")
    # only 5 decisions so far - below MIN_DECISIONS_TO_TRIGGER
    assert result.triggered is False


def test_reset_clears_history():
    d = AntiGamingDetector()
    for _ in range(15):
        d.record_and_check("LOW")
    d.reset()
    result = d.record_and_check("LOW")
    assert result.window_size == 1
    assert result.triggered is False
