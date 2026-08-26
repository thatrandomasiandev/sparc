"""End-to-end smoke test — full pipeline under 2 minutes (ROADMAP Week 14)."""

from __future__ import annotations

import json
import tempfile
import time
from pathlib import Path

import pytest

pytest.importorskip("torch")

from sparc.cli.main import main
from sparc.engine import SparcTrainer, TrainConfig
from sparc.query_optimizer import QueryOptimizerConfig
from sparc.reward_model import RewardModelConfig


def _smoke_config(log_path: str | None = None) -> TrainConfig:
    return TrainConfig(
        seed=0,
        num_windows=3,
        env_steps_per_window=100,
        reward=RewardModelConfig(
            embed_dim=16,
            latent_dim=4,
            hidden_dim=32,
            ensemble_size=3,
        ),
        query=QueryOptimizerConfig(batch_size=4),
        log_path=log_path,
    )


def test_sparc_trainer_e2e_smoke() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        log_path = Path(tmp) / "train.jsonl"
        cfg = _smoke_config(str(log_path))
        t0 = time.perf_counter()
        result = SparcTrainer(cfg).run()
        elapsed = time.perf_counter() - t0

        assert result.windows_completed == 3
        assert result.total_queries == 12
        assert result.total_labels > 0
        assert elapsed < 120.0

        lines = log_path.read_text(encoding="utf-8").strip().splitlines()
        events = [json.loads(line)["event"] for line in lines]
        assert "train_start" in events
        assert "window_open" in events
        assert "label_received" in events
        assert "reward_train" in events
        assert "gate_update" in events
        assert "bounding_applied" in events
        assert "train_end" in events


def test_cli_train_smoke_config() -> None:
    config_path = Path(__file__).resolve().parents[1] / "experiments/configs/smoke_easy.json"
    if not config_path.exists():
        pytest.skip("smoke config missing")

    with tempfile.TemporaryDirectory() as tmp:
        log_path = Path(tmp) / "cli.jsonl"
        t0 = time.perf_counter()
        code = main(["train", "--config", str(config_path), "--log", str(log_path)])
        elapsed = time.perf_counter() - t0

        assert code == 0
        assert elapsed < 120.0
        assert log_path.exists()
        assert log_path.stat().st_size > 0
