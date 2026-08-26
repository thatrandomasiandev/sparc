"""Tests for SPARC reward wrapper and policy trainer."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import gymnasium as gym
import numpy as np

from sparc.engine.config import TrainConfig
from sparc.engine.reward_wrapper import SparcRewardWrapper
from sparc.harness.compare import compare_summaries
from sparc.policy import BoundingConfig, ConfidenceGate
from sparc.policy.config import BoundingMode
from sparc.reward_model import RewardEnsemble


class _TinyEnv(gym.Env):
    def __init__(self) -> None:
        super().__init__()
        self.observation_space = gym.spaces.Box(-1, 1, shape=(4,), dtype=np.float32)
        self.action_space = gym.spaces.Box(-1, 1, shape=(2,), dtype=np.float32)
        self._step = 0

    def reset(self, *, seed=None, options=None):
        self._step = 0
        return np.zeros(4, dtype=np.float32), {}

    def step(self, action):
        self._step += 1
        obs = np.full(4, self._step * 0.01, dtype=np.float32)
        return obs, 1.0, self._step >= 20, False, {}


def test_sparc_reward_wrapper_bounded_step() -> None:
    env = _TinyEnv()
    cfg = TrainConfig(state_dim=4, action_dim=2, horizon=4)
    ensemble = RewardEnsemble.create(4, 2, 4, config=cfg.reward, seed=0)
    gate = ConfidenceGate(
        ensemble,
        BoundingConfig(mode=BoundingMode.SHRINK),
        operator_id=2,
    )
    wrapped = SparcRewardWrapper(env, ensemble, gate, horizon=4, operator_id=2)
    obs, _ = wrapped.reset()
    total = 0.0
    for _ in range(10):
        action = wrapped.action_space.sample()
        obs, reward, term, trunc, info = wrapped.step(action)
        total += reward
        assert "true_reward" in info
        if term or trunc:
            break
    assert isinstance(total, float)


def test_compare_summaries() -> None:
    summary_a = {
        "method": "pebble",
        "env_id": "walker-walk",
        "runs": [{"total_queries": 1000, "eval_history": [{"mean_return": 900.0}]}],
        "summary": {"mean_asymptotic_return": 900.0, "mean_total_queries": 1000.0},
    }
    summary_b = {
        "method": "sparc",
        "env_id": "walker-walk",
        "runs": [{"total_queries": 180, "eval_history": [{"mean_return": 880.0}]}],
        "summary": {"mean_asymptotic_return": 880.0, "mean_total_queries": 180.0},
    }
    with tempfile.TemporaryDirectory() as tmp:
        pa, pb = Path(tmp) / "a.json", Path(tmp) / "b.json"
        pa.write_text(json.dumps(summary_a))
        pb.write_text(json.dumps(summary_b))
        report = compare_summaries(pa, pb)
        assert report.mean_queries_a == 1000.0
        assert report.mean_queries_b == 180.0
        assert report.query_reduction_pct == 82.0
