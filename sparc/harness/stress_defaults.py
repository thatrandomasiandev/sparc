"""EXPERIMENTS.md stress-regime comms windows and query budgets."""

from __future__ import annotations

import math
from dataclasses import replace

from sparc.engine.config import TrainConfig
from sparc.env.specs import EnvSpec, get_env_spec
from sparc.harness.sparc_defaults import apply_sparc_env_defaults
from sparc.query_optimizer.config import QueryOptimizerConfig

# Locked in PROBLEM.md / EXPERIMENTS.md
COMMS_DELTA_T_SIM = 900.0  # seconds of simulated mission time per window
N_SPARC_QUERIES = 200  # ≤20% of PEBBLE reference budget
N_PEBBLE_QUERIES = 1000
QUERY_BATCH_SIZE = 8


def env_steps_per_comms_window(spec: EnvSpec) -> int:
    """Env steps spanning one comms window (ΔT sim mission time)."""
    return int(COMMS_DELTA_T_SIM / spec.sim_dt)


def num_comms_windows(spec: EnvSpec) -> int:
    """Number of comms windows to cover train_steps (ceil division)."""
    steps = env_steps_per_comms_window(spec)
    return max(1, math.ceil(spec.train_steps / steps))


def max_queries_without_cap(spec: EnvSpec, batch_size: int = QUERY_BATCH_SIZE) -> int:
    """Total queries if every window issues a full batch."""
    return num_comms_windows(spec) * batch_size


def stress_schedule_for_env(env_id: str) -> dict:
    """
    Compute stress-regime training schedule from EXPERIMENTS.md env registry.

    Returns keys suitable for TrainConfig.from_dict / dataclass replace.
    """
    spec = get_env_spec(env_id)
    if spec is None:
        raise ValueError(f"unknown env_id for stress schedule: {env_id}")

    steps_per_window = env_steps_per_comms_window(spec)
    windows = num_comms_windows(spec)
    policy_steps = max(1000, spec.train_steps // windows)

    return {
        "env_id": env_id,
        "train_steps_total": spec.train_steps,
        "horizon": spec.segment_horizon,
        "num_windows": windows,
        "env_steps_per_window": steps_per_window,
        "max_queries": N_SPARC_QUERIES,
        "policy_steps_per_window": policy_steps,
        "query": QueryOptimizerConfig(batch_size=QUERY_BATCH_SIZE),
    }


def apply_stress_defaults(config: TrainConfig) -> TrainConfig:
    """
    Apply env registry + stress comms schedule when requested.

    Order: env spec defaults, then stress schedule if ``auto_stress_schedule``.
    """
    config = apply_sparc_env_defaults(config)
    if not config.auto_stress_schedule or config.regime != "stress":
        return config
    if not config.env_id:
        return config

    schedule = stress_schedule_for_env(config.env_id)
    query_defaults = schedule.pop("query")
    schedule.pop("env_id", None)
    # Preserve ablation / custom query fields (e.g. selection_mode) from the loaded config
    query = replace(
        query_defaults,
        selection_mode=config.query.selection_mode,
        lambda_div=config.query.lambda_div,
        eta_similarity=config.query.eta_similarity,
        newton_prior_sigma=config.query.newton_prior_sigma,
        wall_clock_budget_sec=config.query.wall_clock_budget_sec,
        max_candidates=config.query.max_candidates,
    )
    return replace(config, query=query, **schedule)
