"""SURF baseline trainer — PEBBLE + semi-supervised pseudo-labels."""

from __future__ import annotations

from dataclasses import dataclass

from sparc.baselines.config import BaselineConfig
from sparc.baselines.loop import run_preference_baseline
from sparc.baselines.pebble_reward import PebbleRewardModel
from sparc.baselines.surf_reward import SurfRewardModel
from sparc.env.specs import get_env_spec
from sparc.harness.metrics import BaselineRunResult


def _make_surf_reward(cfg: BaselineConfig, obs_dim: int, action_dim: int) -> PebbleRewardModel:
    spec = get_env_spec(cfg.env_id)
    seg_len = spec.segment_horizon if spec else cfg.segment_length
    if cfg.env_id == "CartPole-v1":
        seg_len = min(seg_len, 10)
    return SurfRewardModel(
        obs_dim=obs_dim,
        action_dim=action_dim,
        segment_length=seg_len,
        ensemble_size=cfg.ensemble_size,
        lr=cfg.reward_lr,
        mb_size=min(cfg.reward_batch, 32),
        large_batch=cfg.large_batch,
        activation=cfg.activation,
        reward_update=cfg.reward_update,
        teacher_gamma=cfg.teacher_gamma,
        teacher_eps_mistake=cfg.teacher_eps_mistake,
        teacher_eps_skip=cfg.teacher_eps_skip,
        teacher_eps_equal=cfg.teacher_eps_equal,
        seed=cfg.seed,
        surf_threshold=cfg.surf_threshold,
        surf_noise_std=cfg.surf_noise_std,
        surf_unlabeled_batch=cfg.surf_unlabeled_batch,
        surf_pseudo_weight=cfg.surf_pseudo_weight,
    )


@dataclass
class SurfTrainer:
    """Runs SURF: PEBBLE query schedule + semi-supervised reward training."""

    config: BaselineConfig

    def run(self) -> BaselineRunResult:
        return run_preference_baseline(
            self.config,
            method="surf",
            reward_model_factory=_make_surf_reward,
        )
