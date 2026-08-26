"""Training engine configuration."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from sparc.annotator_sim.defaults import NUM_OPERATORS
from sparc.policy.config import BoundingMode
from sparc.query_optimizer.config import QueryOptimizerConfig
from sparc.reward_model.config import RewardModelConfig


@dataclass
class TrainConfig:
    """Minimal integrated training config for Week 14 smoke / toy runs."""

    seed: int = 0
    num_windows: int = 5
    env_steps_per_window: int = 500
    state_dim: int = 8
    action_dim: int = 2
    horizon: int = 4
    train_epochs_per_window: int = 1
    regime: str = "easy"  # "easy" | "stress"
    train_steps_total: int = 750_000
    log_path: str | None = None
    env_id: str | None = None
    operator_id: int = 1
    num_operators: int = NUM_OPERATORS
    reward: RewardModelConfig = field(default_factory=RewardModelConfig)
    query: QueryOptimizerConfig = field(default_factory=QueryOptimizerConfig)
    bounding_mode: BoundingMode = BoundingMode.SHRINK
    eval_every_windows: int = 1
    num_eval_episodes: int = 10
    train_policy: bool = True
    policy_steps_per_window: int = 500
    policy_batch_size: int = 256
    policy_learning_starts: int = 100
    max_queries: int | None = None  # EXPERIMENTS.md N_SPARC=200 under stress
    auto_stress_schedule: bool = False  # derive num_windows / ΔT from env spec
    # Week 25 ablations (ROADMAP): each flag disables one pillar independently
    share_operator_latents: bool = False  # True → SPARC − Pillar 1 (pooled labels → op 1)
    enable_drift_detector: bool = True  # False → SPARC − Pillar 4
    # Pillar 2 ablation: query.selection_mode = "random"
    # Pillar 3 ablation: bounding_mode = "pass-through"

    @classmethod
    def from_json_file(cls, path: Path | str) -> TrainConfig:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict) -> TrainConfig:
        reward = RewardModelConfig(**data.pop("reward", {}))
        query = QueryOptimizerConfig(**data.pop("query", {}))
        mode_str = data.pop("bounding_mode", "shrink")
        bounding_mode = BoundingMode(mode_str)
        return cls(reward=reward, query=query, bounding_mode=bounding_mode, **data)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["bounding_mode"] = self.bounding_mode.value
        return d

    def to_json_file(self, path: Path | str) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
