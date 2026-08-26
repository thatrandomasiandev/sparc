"""EXPERIMENTS.md stress-regime schedule and query budget tests."""

from __future__ import annotations

from sparc.engine.config import TrainConfig
from sparc.harness.stress_defaults import (
    N_SPARC_QUERIES,
    QUERY_BATCH_SIZE,
    apply_stress_defaults,
    env_steps_per_comms_window,
    max_queries_without_cap,
    num_comms_windows,
    stress_schedule_for_env,
)
from sparc.env.specs import get_env_spec


def test_rover_comms_window_steps() -> None:
    spec = get_env_spec("sparc-rover-nav-v0")
    assert spec is not None
    assert env_steps_per_comms_window(spec) == 18_000  # 900s / 0.05s
    assert num_comms_windows(spec) == 42  # ceil(750k / 18k)
    assert max_queries_without_cap(spec) == 42 * QUERY_BATCH_SIZE


def test_walker_comms_window_steps() -> None:
    spec = get_env_spec("walker-walk")
    assert spec is not None
    assert env_steps_per_comms_window(spec) == 36_000  # 900s / 0.025s
    assert num_comms_windows(spec) == 14  # ceil(500k / 36k)
    assert max_queries_without_cap(spec) == 14 * QUERY_BATCH_SIZE


def test_apply_stress_schedule_rover() -> None:
    cfg = TrainConfig.from_dict(
        {
            "env_id": "sparc-rover-nav-v0",
            "regime": "stress",
            "auto_stress_schedule": True,
        }
    )
    out = apply_stress_defaults(cfg)
    assert out.num_windows == 42
    assert out.env_steps_per_window == 18_000
    assert out.max_queries == N_SPARC_QUERIES
    assert out.query.batch_size == QUERY_BATCH_SIZE
    assert out.train_steps_total == 750_000
    assert out.horizon == 50


def test_apply_stress_schedule_walker() -> None:
    schedule = stress_schedule_for_env("walker-walk")
    assert schedule["num_windows"] == 14
    assert schedule["env_steps_per_window"] == 36_000
    assert schedule["max_queries"] == 200


def test_max_queries_cap_in_trainer() -> None:
    from sparc.engine.trainer import SparcTrainer

    cfg = TrainConfig.from_dict(
        {
            "seed": 0,
            "num_windows": 4,
            "env_steps_per_window": 100,
            "horizon": 4,
            "regime": "easy",
            "max_queries": 5,
            "query": {"batch_size": 8},
        }
    )
    result = SparcTrainer(cfg).run()
    assert result.total_queries <= 5
