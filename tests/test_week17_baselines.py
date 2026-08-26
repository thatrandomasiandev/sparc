"""Week 17 baseline tests — RUNE, SURF, APReL."""

from __future__ import annotations

import numpy as np

from sparc.baselines.aprel_sampling import greedy_medoid_indices
from sparc.baselines.config import BaselineConfig
from sparc.baselines.pebble_reward import PebbleRewardModel
from sparc.harness.runner import run_baseline


def test_greedy_medoid_batch_diversifies_selection() -> None:
    sa1 = np.random.default_rng(0).normal(size=(20, 4, 3)).astype(np.float32)
    sa2 = np.random.default_rng(1).normal(size=(20, 4, 3)).astype(np.float32)
    scores = np.linspace(1.0, 0.0, 20)
    picked = greedy_medoid_indices(sa1, sa2, scores, batch_size=5)
    assert len(picked) == 5
    assert picked[0] == 0  # highest disagreement first
    assert len(set(picked)) == 5


def test_aprel_feed_type_samples_labels() -> None:
    rm = PebbleRewardModel(obs_dim=4, action_dim=1, segment_length=4, mb_size=4, seed=0)
    for traj in range(4):
        rm.start_trajectory()
        for t in range(30):
            obs = np.full(4, traj + t * 0.01, dtype=np.float32)
            rm.add_transition(obs, np.array([0.1], dtype=np.float32), float(t))
    before = rm.total_labels
    rm.sample_preferences(feed_type=2)
    assert rm.total_labels > before


def test_rune_smoke() -> None:
    cfg = BaselineConfig(
        method="rune",
        env_id="Pendulum-v1",
        seed=0,
        num_train_steps=1500,
        num_seed_steps=100,
        num_unsup_steps=300,
        num_interact=200,
        max_feedback=16,
        eval_frequency=1000,
        num_eval_episodes=1,
        segment_length=8,
        reward_batch=8,
        reward_update=10,
        large_batch=3,
        learning_starts=50,
        buffer_size=3000,
        rune_beta=0.05,
    )
    result = run_baseline(cfg, method="rune")
    assert result.method == "rune"
    assert result.total_queries > 0
    assert result.total_env_steps >= 1200


def test_surf_smoke() -> None:
    cfg = BaselineConfig(
        method="surf",
        env_id="Pendulum-v1",
        seed=1,
        num_train_steps=1500,
        num_seed_steps=100,
        num_unsup_steps=300,
        num_interact=200,
        max_feedback=16,
        eval_frequency=1000,
        num_eval_episodes=1,
        segment_length=8,
        reward_batch=8,
        reward_update=10,
        large_batch=3,
        learning_starts=50,
        buffer_size=3000,
    )
    result = run_baseline(cfg, method="surf")
    assert result.method == "surf"
    assert result.total_queries > 0


def test_aprel_smoke() -> None:
    cfg = BaselineConfig(
        method="aprel",
        env_id="Pendulum-v1",
        seed=2,
        num_train_steps=1500,
        num_seed_steps=100,
        num_unsup_steps=300,
        num_interact=200,
        max_feedback=16,
        eval_frequency=1000,
        num_eval_episodes=1,
        segment_length=8,
        reward_batch=8,
        reward_update=10,
        large_batch=3,
        learning_starts=50,
        buffer_size=3000,
    )
    result = run_baseline(cfg, method="aprel")
    assert result.method == "aprel"
    assert result.total_queries > 0
