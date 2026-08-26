"""Tests for SyntheticAnnotator (EXPERIMENTS.md)."""

from __future__ import annotations

import numpy as np

from sparc.annotator_sim import (
    DRIFT1_ALPHA,
    OPERATOR_ALPHAS,
    SyntheticAnnotator,
)
from sparc.env.dataset import PreferenceQuery
from sparc.env.segments import TrajectorySegment


def _seg(speed: float, clearance: float = 1.0, action_scale: float = 0.1) -> TrajectorySegment:
    states = np.zeros((4, 8), dtype=np.float64)
    states[:, 3] = speed
    states[:, 5:8] = clearance
    actions = np.ones((4, 2), dtype=np.float64) * action_scale
    return TrajectorySegment(states=states, actions=actions)


def test_operator_rotation() -> None:
    ann = SyntheticAnnotator(seed=0)
    assert ann.operator_for_window(0) == 1
    assert ann.operator_for_window(1) == 2
    assert ann.operator_for_window(2) == 3
    assert ann.operator_for_window(3) == 1


def test_op1_prefers_speed() -> None:
    ann = SyntheticAnnotator(p_mistake=0.0, p_skip=0.0, seed=0)
    fast = _seg(speed=3.0)
    slow = _seg(speed=0.5)
    label = ann.label_pair(fast, slow, operator_id=1, env_step=0)
    assert label == 0  # prefer segment_0 (fast)


def test_op2_prefers_clearance() -> None:
    ann = SyntheticAnnotator(p_mistake=0.0, p_skip=0.0, seed=0)
    wide = _seg(speed=1.0, clearance=3.0)
    tight = _seg(speed=1.0, clearance=0.2)
    label = ann.label_pair(wide, tight, operator_id=2, env_step=0)
    assert label == 0


def test_drift1_changes_op2_alpha() -> None:
    train_steps = 750_000
    ann = SyntheticAnnotator.stress_regime(train_steps=train_steps, seed=0)
    before = ann.alpha_for(2, env_step=0)
    assert np.allclose(before, OPERATOR_ALPHAS[2])
    after = ann.alpha_for(2, env_step=int(0.5 * train_steps))
    assert np.allclose(after, DRIFT1_ALPHA)


def test_easy_regime_single_operator_no_noise() -> None:
    ann = SyntheticAnnotator.easy_regime(seed=42)
    assert ann.num_operators == 1
    assert ann.p_mistake == 0.0
    assert ann.p_skip == 0.0


def test_label_query_integration() -> None:
    ann = SyntheticAnnotator(p_mistake=0.0, p_skip=0.0, seed=1)
    q = PreferenceQuery(
        segment_0=_seg(2.0),
        segment_1=_seg(0.5),
        operator_id=1,
        window_id=5,
        env_step=1000,
    )
    assert ann.label_query(q) == 0


def test_skip_rate_high() -> None:
    ann = SyntheticAnnotator(p_mistake=0.0, p_skip=1.0, seed=0)
    label = ann.label_pair(_seg(1.0), _seg(2.0), operator_id=1, env_step=0)
    assert label is None
