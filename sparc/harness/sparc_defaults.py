"""Apply EXPERIMENTS.md env registry defaults to SPARC TrainConfig."""

from __future__ import annotations

from dataclasses import replace

from sparc.engine.config import TrainConfig
from sparc.env.specs import get_env_spec


def apply_sparc_env_defaults(config: TrainConfig) -> TrainConfig:
    if not config.env_id:
        return config
    spec = get_env_spec(config.env_id)
    if spec is None:
        return config
    updates: dict = {"train_steps_total": spec.train_steps}
    if config.horizon == 4:
        updates["horizon"] = spec.segment_horizon
    return replace(config, **updates)
