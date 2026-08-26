"""Tests for held-out reward-model accuracy metric."""

from __future__ import annotations

import numpy as np

from sparc.annotator_sim.synthetic import SyntheticAnnotator
from sparc.env.segments import TrajectorySegment
from sparc.harness.rm_accuracy import held_out_pairwise_accuracy, oracle_preference_label
from sparc.reward_model import RewardEnsemble
from sparc.reward_model.config import RewardModelConfig


def _segment(val: float) -> TrajectorySegment:
    states = np.full((4, 8), val, dtype=np.float64)
    actions = np.zeros((4, 2), dtype=np.float64)
    return TrajectorySegment(states=states, actions=actions)


def test_oracle_label_follows_utility_order() -> None:
    ann = SyntheticAnnotator.easy_regime(seed=0)
    ann.feature_layout = "rover"
    low = _segment(0.1)
    high = _segment(2.0)
    assert oracle_preference_label(high, low, ann, 2, 100) == 0
    assert oracle_preference_label(low, high, ann, 2, 100) == 1


def test_held_out_accuracy_runs() -> None:
    ann = SyntheticAnnotator.stress_regime(train_steps=10_000, seed=0)
    ann.feature_layout = "rover"
    ensemble = RewardEnsemble.create(8, 2, 4, config=RewardModelConfig(ensemble_size=2), seed=0)
    segments = [_segment(float(i)) for i in range(10)]
    acc = held_out_pairwise_accuracy(
        ensemble, ann, segments, operator_id=1, env_step=100, n_pairs=20, seed=0
    )
    assert 0.0 <= acc <= 1.0
