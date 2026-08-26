"""RUNE baseline — PEBBLE + reward-ensemble exploration bonus (Liang et al. 2022)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import numpy as np

from sparc.baselines.config import BaselineConfig
from sparc.baselines.loop import LearnedRewardWrapper, _as_action, run_preference_baseline
from sparc.baselines.pebble_reward import PebbleRewardModel
from sparc.harness.metrics import BaselineRunResult


class RuneRewardWrapper(LearnedRewardWrapper):
    """Adds β · std(r̂_ensemble) exploration bonus to learned step rewards."""

    def __init__(
        self,
        env: gym.Env,
        reward_model: PebbleRewardModel,
        beta: float = 0.05,
    ) -> None:
        super().__init__(env, reward_model)
        self.beta = beta

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if self._last_obs is None:
            raise RuntimeError("reset() must precede step()")
        obs, true_r, term, trunc, info = self.env.step(action)
        action_arr = _as_action(self.env, action)
        self.reward_model.add_transition(self._last_obs, action_arr, float(true_r))
        r_hat = self.reward_model.r_hat(self._last_obs, action_arr)
        bonus = self.beta * self.reward_model.step_disagreement(self._last_obs, action_arr)
        self._last_obs = obs
        info["true_reward"] = true_r
        info["rune_bonus"] = bonus
        return obs, r_hat + bonus, term, trunc, info


@dataclass
class RuneTrainer:
    """PEBBLE + RUNE unified exploration on ensemble reward disagreement."""

    config: BaselineConfig

    def run(self) -> BaselineRunResult:
        return run_preference_baseline(
            self.config,
            method="rune",
            wrapper_cls=RuneRewardWrapper,
            wrapper_kwargs={"beta": self.config.rune_beta},
        )
