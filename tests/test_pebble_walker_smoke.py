"""PEBBLE smoke on walker-walk (requires MuJoCo / [dmc] extra)."""

from __future__ import annotations

import time

import pytest

from sparc.baselines.config import BaselineConfig
from sparc.baselines.pebble import PebbleTrainer
from sparc.env.dmc import is_dmc_available

pytestmark = pytest.mark.skipif(not is_dmc_available(), reason="shimmy/dm-control not installed")


@pytest.mark.timeout(180)
def test_pebble_walker_walk_smoke() -> None:
    cfg = BaselineConfig(
        env_id="walker-walk",
        seed=0,
        num_train_steps=2500,
        num_seed_steps=300,
        num_unsup_steps=600,
        num_interact=400,
        max_feedback=16,
        eval_frequency=1200,
        num_eval_episodes=1,
        reward_batch=8,
        reward_update=10,
        large_batch=3,
        learning_starts=100,
        buffer_size=5000,
        batch_size=64,
    )
    t0 = time.perf_counter()
    result = PebbleTrainer(cfg).run()
    elapsed = time.perf_counter() - t0
    assert result.total_env_steps >= 2000
    assert result.total_queries >= 0
    assert elapsed < 180.0
