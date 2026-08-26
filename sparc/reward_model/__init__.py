"""Pillar 1 — per-operator latent reward model. See DESIGN.md § Pillar 1."""

from sparc.reward_model.config import RewardModelConfig
from sparc.reward_model.encoder import TrajectoryEncoder
from sparc.reward_model.ensemble import RewardEnsemble
from sparc.reward_model.latent import infer_operator_latent
from sparc.reward_model.losses import bradley_terry_loss

__all__ = [
    "RewardModelConfig",
    "TrajectoryEncoder",
    "RewardEnsemble",
    "infer_operator_latent",
    "bradley_terry_loss",
]
