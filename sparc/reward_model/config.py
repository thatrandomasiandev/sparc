"""Default hyperparameters for Pillar 1 (DESIGN.md)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RewardModelConfig:
    embed_dim: int = 128
    latent_dim: int = 16
    hidden_dim: int = 256
    num_operators: int = 3
    ensemble_size: int = 7
    sigma_z: float = 1.0
    map_steps: int = 100
    map_lr: float = 0.05
    train_lr: float = 3e-4
