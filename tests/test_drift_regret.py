"""Tests for post-drift regret metric."""

from __future__ import annotations

from sparc.harness.drift_regret import post_drift_eval_window, regret_under_drift
from sparc.harness.metrics import EvalMetrics


def test_post_drift_window_bounds() -> None:
    start, end = post_drift_eval_window(500_000)
    assert start == 250_000
    assert end == 300_000


def test_regret_under_drift_computed() -> None:
    train = 500_000
    start, _ = post_drift_eval_window(train)
    history = [
        EvalMetrics(start, 10.0, 50.0, 100),
        EvalMetrics(start + 10_000, 12.0, 40.0, 120),
        EvalMetrics(start + 20_000, 11.0, 45.0, 140),
    ]
    regret = regret_under_drift(history, train)
    assert regret is not None
    assert regret == (0.0 + 10.0 + 5.0) / 3


def test_regret_none_without_post_drift_evals() -> None:
    history = [EvalMetrics(1000, 1.0, 1.0, 0)]
    assert regret_under_drift(history, 500_000) is None
