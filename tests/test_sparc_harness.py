"""SPARC harness integration tests."""

from __future__ import annotations

import tempfile
from pathlib import Path

from sparc.engine.config import TrainConfig
from sparc.harness.multi_seed import run_multi_seed_baseline
from sparc.harness.sparc_runner import run_sparc


def _rover_smoke_config(seed: int = 0) -> TrainConfig:
    return TrainConfig.from_dict(
        {
            "seed": seed,
            "env_id": "sparc-rover-nav-v0",
            "num_windows": 2,
            "env_steps_per_window": 200,
            "horizon": 8,
            "regime": "stress",
            "train_steps_total": 10000,
            "eval_every_windows": 1,
            "num_eval_episodes": 1,
            "train_policy": True,
            "policy_steps_per_window": 100,
            "policy_learning_starts": 20,
            "reward": {"embed_dim": 32, "latent_dim": 8, "hidden_dim": 64, "ensemble_size": 3},
            "query": {"batch_size": 4},
        }
    )


def test_run_sparc_rover_stress_smoke() -> None:
    result = run_sparc(_rover_smoke_config())
    assert result.method == "sparc"
    assert result.env_id == "sparc-rover-nav-v0"
    assert result.total_queries >= 0
    assert result.total_env_steps > 0


def test_multi_seed_sparc_smoke() -> None:
    cfg = _rover_smoke_config()
    with tempfile.TemporaryDirectory() as tmp:
        multi = run_multi_seed_baseline(cfg, method="sparc", seeds=[0, 1], output_dir=Path(tmp))
        assert multi.summary is not None
        assert multi.summary.n_seeds == 2
        assert (Path(tmp) / "summary.json").exists()
