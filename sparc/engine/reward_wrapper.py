"""Gym wrapper: SPARC ensemble + confidence gate step rewards."""

from __future__ import annotations

from collections import deque
from typing import Any

import gymnasium as gym
import numpy as np

from sparc.env.segments import TrajectorySegment
from sparc.policy.gate import ConfidenceGate
from sparc.reward_model.ensemble import RewardEnsemble


def _flat_obs(obs: np.ndarray | dict) -> np.ndarray:
    if isinstance(obs, dict):
        return np.concatenate([np.asarray(v, dtype=np.float64).reshape(-1) for v in obs.values()])
    return np.asarray(obs, dtype=np.float64).reshape(-1)


def _flat_action(action: np.ndarray | int) -> np.ndarray:
    return np.atleast_1d(np.asarray(action, dtype=np.float64)).reshape(-1)


def _segment_from_rollout(
    states: deque[np.ndarray],
    actions: deque[np.ndarray],
    horizon: int,
) -> TrajectorySegment:
    s_list = list(states)[-horizon:]
    a_list = list(actions)[-horizon:]
    state_dim = s_list[-1].shape[0]
    action_dim = a_list[-1].shape[0]
    if len(s_list) < horizon:
        pad = horizon - len(s_list)
        s_arr = np.vstack([np.zeros((pad, state_dim), dtype=np.float64), np.stack(s_list)])
        a_arr = np.vstack([np.zeros((pad, action_dim), dtype=np.float64), np.stack(a_list)])
    else:
        s_arr = np.stack(s_list)
        a_arr = np.stack(a_list)
    return TrajectorySegment(states=s_arr, actions=a_arr)


class SparcRewardWrapper(gym.Wrapper):
    """
    Step rewards from SPARC ensemble + Pillar 3 bounding over a rolling H-step segment.
    """

    def __init__(
        self,
        env: gym.Env,
        ensemble: RewardEnsemble,
        gate: ConfidenceGate,
        horizon: int,
        operator_id: int = 1,
    ) -> None:
        super().__init__(env)
        self.ensemble = ensemble
        self.gate = gate
        self.horizon = horizon
        self.operator_id = operator_id
        self._states: deque[np.ndarray] = deque(maxlen=horizon)
        self._actions: deque[np.ndarray] = deque(maxlen=horizon)
        self._last_obs: np.ndarray | None = None

    def set_operator(self, operator_id: int) -> None:
        self.operator_id = operator_id
        self.gate.operator_id = operator_id

    def reset(self, **kwargs: Any) -> tuple[np.ndarray, dict[str, Any]]:
        obs, info = self.env.reset(**kwargs)
        self._states.clear()
        self._actions.clear()
        self._last_obs = _flat_obs(obs)
        return obs, info

    def _bounded_reward(self) -> float:
        segment = _segment_from_rollout(self._states, self._actions, self.horizon)
        r_hat = self.ensemble.pooled_reward(segment, self.operator_id)
        bound = self.gate.bound_reward(segment, r_hat)
        if bound.skip_transition:
            return 0.0
        return bound.reward

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if self._last_obs is None:
            raise RuntimeError("reset() must precede step()")
        act = _flat_action(action)
        self._states.append(self._last_obs.copy())
        self._actions.append(act.copy())
        obs, true_r, term, trunc, info = self.env.step(action)
        r_hat = self._bounded_reward()
        self._last_obs = _flat_obs(obs)
        info["true_reward"] = true_r
        info["learned_reward"] = r_hat
        return obs, r_hat, term, trunc, info
