"""Hyperparameters for PEBBLE / PrefPPO baselines (B-Pref defaults, EXPERIMENTS.md)."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class BaselineConfig:
    """
    Shared baseline training config.

    Defaults follow B-Pref `train_PEBBLE.yaml` scaled for smoke runs.
    """

    method: str = "pebble"  # pebble | prefppo | surf | rune | aprel
    env_id: str = "CartPole-v1"
    seed: int = 0
    num_train_steps: int = 500_000
    num_seed_steps: int = 1000
    num_unsup_steps: int = 5000
    num_interact: int = 5000
    max_feedback: int = 1000
    eval_frequency: int = 10_000
    num_eval_episodes: int = 10
    log_path: str | None = None

    # reward model (B-Pref)
    segment_length: int = 50
    ensemble_size: int = 3
    reward_lr: float = 3e-4
    reward_batch: int = 128
    reward_update: int = 200
    feed_type: int = 1  # 0 uniform, 1 disagreement (PEBBLE), 2 APReL medoids
    large_batch: int = 10
    activation: str = "tanh"
    label_margin: float = 0.0

    # annotator (easy regime oracle for baseline parity)
    use_oracle_labels: bool = True
    teacher_gamma: float = 1.0
    teacher_eps_mistake: float = 0.0
    teacher_eps_skip: float = 0.0
    teacher_eps_equal: float = 0.0

    # policy (SB3)
    policy_lr: float = 3e-4
    batch_size: int = 256
    buffer_size: int = 1_000_000
    learning_starts: int = 1000
    train_freq: int = 1
    gradient_steps: int = 1
    gamma: float = 0.99

    # RUNE (Liang et al. 2022)
    rune_beta: float = 0.05

    # SURF (Park et al. 2022)
    surf_threshold: float = 0.95
    surf_noise_std: float = 0.1
    surf_unlabeled_batch: int = 128
    surf_pseudo_weight: float = 1.0

    @classmethod
    def from_json_file(cls, path: Path | str) -> BaselineConfig:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(**data)

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json_file(self, path: Path | str) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
