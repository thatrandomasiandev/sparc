"""Collect trajectory segments from a Gymnasium environment."""

from __future__ import annotations

from typing import TYPE_CHECKING

import gymnasium as gym
import numpy as np

from sparc.env.buffer import SegmentBuffer

if TYPE_CHECKING:
    from stable_baselines3.common.base_class import BaseAlgorithm


def _flat_obs(obs: np.ndarray | dict) -> np.ndarray:
    if isinstance(obs, dict):
        parts = [np.asarray(v, dtype=np.float64).reshape(-1) for v in obs.values()]
        return np.concatenate(parts)
    return np.asarray(obs, dtype=np.float64).reshape(-1)


def _flat_action(action: np.ndarray | int) -> np.ndarray:
    return np.atleast_1d(np.asarray(action, dtype=np.float64)).reshape(-1)


def infer_env_dims(env: gym.Env) -> tuple[int, int]:
    obs, _ = env.reset()
    obs_dim = int(_flat_obs(obs).shape[0])
    space = env.action_space
    if hasattr(space, "shape") and space.shape is not None and len(space.shape) > 0:
        action_dim = int(space.shape[0])
    else:
        action_dim = 1
    return obs_dim, action_dim


def evaluate_random_policy(
    env: gym.Env,
    num_episodes: int,
    seed: int,
) -> tuple[float, float]:
    """Mean learned-proxy and true return under random actions (pre-policy SPARC eval)."""
    returns: list[float] = []
    true_returns: list[float] = []
    for ep in range(num_episodes):
        obs, _ = env.reset(seed=seed + ep)
        done = False
        ep_r = 0.0
        ep_true = 0.0
        while not done:
            action = env.action_space.sample()
            obs, reward, term, trunc, _ = env.step(action)
            ep_r += float(reward)
            ep_true += float(reward)
            done = term or trunc
        returns.append(ep_r)
        true_returns.append(ep_true)
    return float(np.mean(returns)), float(np.mean(true_returns))


def collect_rollouts(
    env: gym.Env,
    buffer: SegmentBuffer,
    rng: np.random.Generator,
    total_steps: int,
    policy: BaseAlgorithm | None = None,
) -> int:
    """Fill segment buffer; optional SB3 policy instead of random actions."""
    obs, _ = env.reset(seed=int(rng.integers(0, 2**31)))
    state_list: list[np.ndarray] = []
    action_list: list[np.ndarray] = []
    steps = 0

    while steps < total_steps:
        if policy is None:
            action = env.action_space.sample()
        else:
            action, _ = policy.predict(obs, deterministic=False)
        next_obs, _reward, terminated, truncated, _ = env.step(action)
        state_list.append(_flat_obs(obs))
        action_list.append(_flat_action(action))
        obs = next_obs
        steps += 1

        if terminated or truncated:
            if len(state_list) >= buffer.horizon:
                buffer.add_trajectory(np.stack(state_list), np.stack(action_list))
            state_list.clear()
            action_list.clear()
            obs, _ = env.reset()
        elif len(state_list) >= buffer.horizon * 2:
            buffer.add_trajectory(
                np.stack(state_list[: buffer.horizon * 2]),
                np.stack(action_list[: buffer.horizon * 2]),
            )
            state_list = state_list[buffer.horizon :]
            action_list = action_list[buffer.horizon :]

    if len(state_list) >= buffer.horizon:
        buffer.add_trajectory(np.stack(state_list), np.stack(action_list))
    return steps
