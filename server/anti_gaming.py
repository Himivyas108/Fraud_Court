"""
Anti-gaming detector. Deterministic, stateful, no AI. Prevents the trivial
exploit of "always say LOW to avoid the -0.8 worst case": if LOW-rate
exceeds 70% across a rolling window of 10+ decisions, a progressive
penalty multiplier is applied to subsequent rewards.
"""
from __future__ import annotations
from collections import deque
from dataclasses import dataclass

WINDOW_SIZE = 50
MIN_DECISIONS_TO_TRIGGER = 10
LOW_RATE_THRESHOLD = 0.70


@dataclass
class GamingCheckResult:
    triggered: bool
    low_rate: float
    penalty_multiplier: float
    window_size: int


class AntiGamingDetector:
    def __init__(self):
        self._history: deque[str] = deque(maxlen=WINDOW_SIZE)

    def record_and_check(self, confidence: str) -> GamingCheckResult:
        self._history.append(confidence)
        n = len(self._history)
        low_count = sum(1 for c in self._history if c == "LOW")
        low_rate = low_count / n if n else 0.0

        if n >= MIN_DECISIONS_TO_TRIGGER and low_rate > LOW_RATE_THRESHOLD:
            # progressive penalty: scales from 0.9x down to 0.5x as the
            # LOW-rate climbs from the threshold toward 1.0
            overshoot = min((low_rate - LOW_RATE_THRESHOLD) / (1.0 - LOW_RATE_THRESHOLD), 1.0)
            multiplier = 0.9 - 0.4 * overshoot
            return GamingCheckResult(True, low_rate, round(multiplier, 3), n)

        return GamingCheckResult(False, low_rate, 1.0, n)

    def reset(self):
        self._history.clear()


# module-level singleton shared across episodes in this process, matching
# the "rolling window across an agent's recent decisions" design in the spec
GLOBAL_DETECTOR = AntiGamingDetector()
