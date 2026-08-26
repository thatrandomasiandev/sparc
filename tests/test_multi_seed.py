"""Tests for multi-seed harness and statistics."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from sparc.baselines.config import BaselineConfig
from sparc.harness.multi_seed import run_multi_seed_baseline
from sparc.harness.seeds import MASTER_SEEDS, parse_seeds
from sparc.harness.stats import confidence_interval_95, summarize_seed_runs, welch_ttest


def test_parse_seeds() -> None:
    assert parse_seeds(None, default=[7]) == [7]
    assert parse_seeds("all") == MASTER_SEEDS
    assert parse_seeds("0,2,4") == [0, 2, 4]


def test_confidence_interval_95() -> None:
    mean, lo, hi = confidence_interval_95([10.0, 12.0, 11.0, 13.0])
    assert lo <= mean <= hi
    assert mean == pytest.approx(11.5)


def test_summarize_seed_runs() -> None:
    summary = summarize_seed_runs([100.0, 110.0], [120.0, 130.0], [50, 60], [1.0, 2.0])
    assert summary.n_seeds == 2
    assert summary.mean_asymptotic_return == pytest.approx(105.0)
    assert summary.mean_total_queries == pytest.approx(55.0)


def test_welch_ttest_with_scipy() -> None:
    t, p = welch_ttest([1.0, 2.0, 3.0, 4.0], [1.1, 2.1, 2.9, 4.2])
    if t == t:  # not NaN
        assert 0.0 <= p <= 1.0


def test_run_multi_seed_baseline_smoke() -> None:
    cfg = BaselineConfig(
        env_id="Pendulum-v1",
        seed=0,
        num_train_steps=1000,
        num_seed_steps=80,
        num_unsup_steps=150,
        num_interact=120,
        max_feedback=8,
        eval_frequency=500,
        num_eval_episodes=1,
        reward_batch=8,
        reward_update=5,
        large_batch=3,
        learning_starts=40,
        buffer_size=2000,
        batch_size=64,
    )
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        multi = run_multi_seed_baseline(cfg, method="pebble", seeds=[0, 1], output_dir=out)
        assert multi.summary is not None
        assert multi.summary.n_seeds == 2
        assert len(multi.runs) == 2
        assert (out / "summary.json").exists()
        data = json.loads((out / "summary.json").read_text())
        assert data["seeds"] == [0, 1]
        assert (out / "seed_0.jsonl").exists()
