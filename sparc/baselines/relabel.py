"""Relabel off-policy replay buffers after preference reward-model updates (B-Pref PEBBLE)."""

from __future__ import annotations

import gymnasium as gym
import numpy as np
from stable_baselines3 import SAC

from sparc.baselines.pebble_reward import PebbleRewardModel


def _as_action(env: gym.Env, action: np.ndarray | int) -> np.ndarray:
    if isinstance(env.action_space, gym.spaces.Discrete):
        return np.array([float(int(action))], dtype=np.float32)
    return np.asarray(action, dtype=np.float32)


def relabel_sac_replay_buffer(
    model: SAC,
    reward_model: PebbleRewardModel,
    env: gym.Env,
) -> int:
    """
    Overwrite stored SAC transition rewards with the current ensemble mean.

    B-Pref PEBBLE retrains the reward model periodically; without relabeling,
    stale rewards in the replay buffer bias off-policy learning.
    """
    buffer = model.replay_buffer
    if buffer is None:
        return 0
    n = int(buffer.size())
    if n == 0:
        return 0

    for idx in range(n):
        obs = np.asarray(buffer.observations[idx, 0], dtype=np.float32)
        action = np.asarray(buffer.actions[idx, 0])
        buffer.rewards[idx, 0] = reward_model.r_hat(obs, _as_action(env, action))

    return n
