"""Apply EXPERIMENTS.md defaults from env registry to baseline configs."""

from __future__ import annotations

from dataclasses import replace

from sparc.baselines.config import BaselineConfig
from sparc.env.specs import get_env_spec


def apply_env_defaults(config: BaselineConfig) -> BaselineConfig:
    """Apply segment horizon (and train steps when unset) from EXPERIMENTS env registry."""
    spec = get_env_spec(config.env_id)
    if spec is None:
        return config
    updates: dict = {"segment_length": spec.segment_horizon}
    if config.num_train_steps == 500_000 and spec.train_steps != 500_000:
        updates["num_train_steps"] = spec.train_steps
    return replace(config, **updates)
