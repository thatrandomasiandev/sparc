"""Tests for Week 25 pillar ablation wiring."""

from __future__ import annotations

from sparc.engine.config import TrainConfig
from sparc.harness.stress_defaults import apply_stress_defaults
from sparc.query_optimizer.batch import select_batch_random
from sparc.query_optimizer.config import QueryOptimizerConfig
from sparc.query_optimizer.eig import QueryCandidate
from sparc.env.segments import TrajectorySegment
import numpy as np


def _toy_candidates(n: int = 10) -> list[QueryCandidate]:
    segs = [
        TrajectorySegment(
            states=np.zeros((4, 2), dtype=np.float32),
            actions=np.zeros((4, 1), dtype=np.float32),
        )
        for _ in range(n + 1)
    ]
    return [QueryCandidate(segment_0=segs[i], segment_1=segs[i + 1]) for i in range(n)]


def test_select_batch_random_size() -> None:
    selected = select_batch_random(_toy_candidates(10), batch_size=4, rng=np.random.default_rng(0))
    assert len(selected) == 4


def test_ablation_flags_roundtrip_json() -> None:
    cfg = TrainConfig.from_dict(
        {
            "share_operator_latents": True,
            "enable_drift_detector": False,
            "bounding_mode": "pass-through",
            "query": {"selection_mode": "random", "batch_size": 4},
        }
    )
    assert cfg.share_operator_latents is True
    assert cfg.enable_drift_detector is False
    assert cfg.bounding_mode.value == "pass-through"
    assert cfg.query.selection_mode == "random"


def test_stress_defaults_preserve_selection_mode() -> None:
    cfg = TrainConfig.from_dict(
        {
            "env_id": "walker-walk",
            "regime": "stress",
            "auto_stress_schedule": True,
            "query": {"selection_mode": "random"},
        }
    )
    cfg = apply_stress_defaults(cfg)
    assert cfg.query.selection_mode == "random"
    assert cfg.query.batch_size == 8
