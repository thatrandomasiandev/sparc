"""SB3 SAC policy training on SPARC learned rewards."""

from __future__ import annotations

import gymnasium as gym
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.utils import set_random_seed

from sparc.engine.config import TrainConfig
from sparc.engine.reward_wrapper import SparcRewardWrapper
from sparc.policy.gate import ConfidenceGate
from sparc.reward_model.ensemble import RewardEnsemble


class SparcPolicyTrainer:
    """Trains SAC against SPARC bounded rewards (mirrors PEBBLE policy loop)."""

    def __init__(
        self,
        base_env: gym.Env,
        ensemble: RewardEnsemble,
        gate: ConfidenceGate,
        config: TrainConfig,
    ) -> None:
        self.base_env = base_env
        self.config = config
        self.wrapper = SparcRewardWrapper(
            base_env,
            ensemble,
            gate,
            horizon=config.horizon,
            operator_id=config.operator_id,
        )
        set_random_seed(config.seed)
        self.model = SAC(
            "MlpPolicy",
            self.wrapper,
            learning_rate=3e-4,
            buffer_size=max(config.policy_steps_per_window * config.num_windows * 2, 10_000),
            learning_starts=min(config.policy_learning_starts, config.policy_steps_per_window),
            batch_size=config.policy_batch_size,
            gamma=0.99,
            seed=config.seed,
            verbose=0,
        )

    def set_operator(self, operator_id: int) -> None:
        self.wrapper.set_operator(operator_id)

    def train(self, timesteps: int) -> None:
        if timesteps <= 0:
            return
        self.model.learn(total_timesteps=timesteps, reset_num_timesteps=False)

    def evaluate(
        self,
        num_episodes: int,
        seed: int,
    ) -> tuple[float, float]:
        """Return (mean learned return, mean true env return) under deterministic policy."""
        learned: list[float] = []
        true_rets: list[float] = []
        for ep in range(num_episodes):
            obs, _ = self.base_env.reset(seed=seed + ep)
            self.wrapper.reset()
            done = False
            ep_learned = 0.0
            ep_true = 0.0
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, true_r, term, trunc, info = self.wrapper.step(action)
                ep_learned += float(info.get("learned_reward", 0.0))
                ep_true += float(info.get("true_reward", true_r))
                done = term or trunc
            learned.append(ep_learned)
            true_rets.append(ep_true)
        return float(np.mean(learned)), float(np.mean(true_rets))
