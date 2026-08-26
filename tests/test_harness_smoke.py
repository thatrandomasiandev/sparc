"""End-to-end smoke tests for Phase 3 baseline harness."""

from __future__ import annotations

import json
import tempfile
import time
from pathlib import Path

from sparc.baselines.config import BaselineConfig
from sparc.baselines.pebble import PebbleTrainer
from sparc.cli.main import main
from sparc.harness.runner import run_baseline


def test_pebble_cartpole_smoke() -> None:
    cfg = BaselineConfig(
        env_id="Pendulum-v1",
        seed=0,
        num_train_steps=1500,
        num_seed_steps=100,
        num_unsup_steps=300,
        num_interact=200,
        max_feedback=20,
        eval_frequency=1000,
        num_eval_episodes=1,
        segment_length=8,
        reward_batch=8,
        reward_update=10,
        large_batch=3,
        learning_starts=50,
        buffer_size=3000,
    )
    t0 = time.perf_counter()
    result = PebbleTrainer(cfg).run()
    elapsed = time.perf_counter() - t0
    assert result.total_env_steps >= cfg.num_train_steps - 200
    assert elapsed < 120.0


def test_cli_benchmark_pebble_smoke() -> None:
    config = {
        "env_id": "Pendulum-v1",
        "seed": 1,
        "num_train_steps": 1200,
        "num_seed_steps": 80,
        "num_unsup_steps": 200,
        "num_interact": 150,
        "max_feedback": 16,
        "eval_frequency": 800,
        "num_eval_episodes": 1,
        "segment_length": 8,
        "reward_batch": 8,
        "reward_update": 8,
        "large_batch": 3,
        "learning_starts": 40,
        "buffer_size": 2000,
    }
    with tempfile.TemporaryDirectory() as tmp:
        config_path = Path(tmp) / "pebble.json"
        log_path = Path(tmp) / "bench.jsonl"
        config_path.write_text(json.dumps(config), encoding="utf-8")
        t0 = time.perf_counter()
        code = main(
            [
                "benchmark",
                "--method",
                "pebble",
                "--config",
                str(config_path),
                "--log",
                str(log_path),
            ]
        )
        elapsed = time.perf_counter() - t0
        assert code == 0
        assert elapsed < 120.0
        assert log_path.exists()
        assert log_path.stat().st_size > 0


def test_run_baseline_prefppo_smoke() -> None:
    cfg = BaselineConfig(
        method="prefppo",
        env_id="Pendulum-v1",
        seed=2,
        num_train_steps=1200,
        num_seed_steps=80,
        num_unsup_steps=200,
        num_interact=150,
        max_feedback=12,
        eval_frequency=900,
        num_eval_episodes=1,
        segment_length=8,
        reward_batch=8,
        reward_update=8,
        large_batch=3,
    )
    result = run_baseline(cfg, method="prefppo")
    assert result.method == "prefppo"
    assert result.total_env_steps > 0
