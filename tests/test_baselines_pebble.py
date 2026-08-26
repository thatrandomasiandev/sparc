"""Tests for PEBBLE-style baseline reward model and replay relabeling."""

from __future__ import annotations

import gymnasium as gym
import numpy as np
from stable_baselines3 import SAC

from sparc.baselines.loop import LearnedRewardWrapper
from sparc.baselines.config import BaselineConfig
from sparc.baselines.loop import _maybe_query
from sparc.baselines.pebble_reward import PebbleRewardModel
from sparc.baselines.relabel import relabel_sac_replay_buffer


def test_pebble_reward_r_hat_and_train() -> None:
    rm = PebbleRewardModel(obs_dim=4, action_dim=1, segment_length=4, mb_size=4, seed=0)
    obs = np.zeros(4, dtype=np.float32)
    action = np.array([0.5], dtype=np.float32)
    for traj in range(3):
        rm.start_trajectory()
        for t in range(30):
            obs = np.full(4, traj + t * 0.01, dtype=np.float32)
            rm.add_transition(obs, np.array([0.1], dtype=np.float32), float(t))
    labeled = rm.uniform_sampling()
    assert labeled >= 0
    acc = rm.train()
    assert 0.0 <= acc <= 1.0
    r = rm.r_hat(obs, action)
    assert isinstance(r, float)


def test_disagreement_sampling_respects_budget() -> None:
    rm = PebbleRewardModel(
        obs_dim=4,
        action_dim=1,
        segment_length=4,
        mb_size=4,
        large_batch=3,
        seed=1,
    )
    for traj in range(4):
        rm.start_trajectory()
        for t in range(30):
            obs = np.full(4, traj + t * 0.01, dtype=np.float32)
            rm.add_transition(obs, np.array([0.1], dtype=np.float32), float(t))
    before = rm.total_labels
    rm.disagreement_sampling()
    assert rm.total_labels >= before


def test_relabel_sac_replay_buffer_updates_stale_rewards() -> None:
    env = gym.make("Pendulum-v1")
    rm = PebbleRewardModel(obs_dim=3, action_dim=1, segment_length=4, seed=0)
    wrapped = LearnedRewardWrapper(env, rm)
    model = SAC(
        "MlpPolicy",
        wrapped,
        buffer_size=500,
        learning_starts=10,
        batch_size=32,
        verbose=0,
        seed=0,
    )
    model.learn(total_timesteps=64, reset_num_timesteps=True)
    assert model.replay_buffer.size() > 0

    stale = np.full_like(model.replay_buffer.rewards, -999.0)
    model.replay_buffer.rewards[:] = stale

    n = relabel_sac_replay_buffer(model, rm, env)
    assert n == model.replay_buffer.size()
    assert not np.allclose(model.replay_buffer.rewards, -999.0)
    env.close()


def test_maybe_query_fires_after_unsup_with_chunked_steps() -> None:
    """First preference batch must fire when effective_step crosses unsup_end (not ==)."""
    from sparc.logging.jsonl import JsonlLogger

    rm = PebbleRewardModel(obs_dim=4, action_dim=1, segment_length=4, mb_size=4, seed=0)
    for traj in range(4):
        rm.start_trajectory()
        for t in range(30):
            obs = np.full(4, traj + t * 0.01, dtype=np.float32)
            rm.add_transition(obs, np.array([0.1], dtype=np.float32), float(t))
    cfg = BaselineConfig(
        num_seed_steps=200,
        num_unsup_steps=400,
        max_feedback=100,
        num_interact=5000,
        feed_type=1,
    )
    logger = JsonlLogger(path=None)
    unsup_end = cfg.num_seed_steps + cfg.num_unsup_steps
    interact, total_feedback, acc, done = _maybe_query(
        effective_step=640,
        cfg=cfg,
        reward_model=rm,
        unsup_end=unsup_end,
        interact=0,
        total_feedback=0,
        first_query_done=False,
        logger=logger,
    )
    assert done is True
    assert rm.total_labels > 0
    assert acc >= 0.0
