"""Tests for Pillar 4 SPRT drift detector and re-query wiring."""

from __future__ import annotations

import math

import numpy as np
import pytest

pytest.importorskip("torch")

from sparc.drift_detector import (
    DriftDetector,
    DriftDetectorConfig,
    log_likelihood_ratio,
)
from sparc.env.dataset import PreferenceQuery, PreferenceRecord
from sparc.env.segments import TrajectorySegment
from sparc.query_optimizer import CommsWindowOptimizer, QueryCandidate, QueryOptimizerConfig
from sparc.query_optimizer.requery import select_batch_with_requery
from sparc.reward_model import RewardEnsemble, RewardModelConfig


def _seg(speed: float) -> TrajectorySegment:
    states = np.zeros((4, 8), dtype=np.float64)
    states[:, 3] = speed
    states[:, 5:8] = 1.0
    actions = np.zeros((4, 2), dtype=np.float64)
    return TrajectorySegment(states=states, actions=actions)


def _trained_ensemble(seed: int = 0) -> RewardEnsemble:
    cfg = RewardModelConfig(embed_dim=16, latent_dim=4, hidden_dim=32, ensemble_size=3)
    model = RewardEnsemble.create(8, 2, 4, config=cfg, seed=seed)
    return model


def test_sprt_boundaries() -> None:
    cfg = DriftDetectorConfig(alpha=0.05, beta=0.10)
    assert cfg.boundary_upper == pytest.approx(math.log((1.0 - cfg.beta) / cfg.alpha))
    assert cfg.boundary_lower == pytest.approx(math.log(cfg.beta / (1.0 - cfg.alpha)))


def test_log_lik_ratio_positive_when_label_surprising_under_h0() -> None:
    """y=0 but H0 prefers seg1 (delta_h0 < 0); H1 shift raises P(y=0)."""
    lr = log_likelihood_ratio(y=0, delta_h0=-3.0, eta=2.0)
    assert lr > 0.0


def test_drift_trigger_on_consecutive_inconsistent_labels() -> None:
    from unittest.mock import patch

    ensemble = _trained_ensemble()
    cfg = DriftDetectorConfig(alpha=0.5, beta=0.5, eta=2.0)
    detector = DriftDetector(ensemble=ensemble, config=cfg)

    records: list[PreferenceRecord] = []
    for _ in range(10):
        q = PreferenceQuery(_seg(3.0), _seg(0.1), operator_id=2, window_id=1, env_step=100)
        records.append(PreferenceRecord(query=q, label=0))

    with patch(
        "sparc.drift_detector.sprt.mean_preference_delta",
        return_value=-3.0,
    ):
        events = detector.update_with_labels(records)

    assert len(events) >= 1
    assert detector.states[2].requery is True
    assert detector.states[2].regime >= 1


def test_requery_cleared_after_window_open() -> None:
    ensemble = _trained_ensemble()
    drift_cfg = DriftDetectorConfig(eta=2.0)
    detector = DriftDetector(ensemble=ensemble, config=drift_cfg)
    detector.states[2].requery = True

    cands = [
        QueryCandidate(_seg(1.0 + i * 0.1), _seg(0.5)) for i in range(20)
    ]
    detector.record_window_queries(cands[:8], window_id=0, operator_id=2)

    opt = CommsWindowOptimizer(
        ensemble=ensemble,
        config=QueryOptimizerConfig(batch_size=8),
        drift_detector=detector,
        num_operators=3,
    )
    result = opt.on_window_open(window_id=1, candidates=cands, env_step=500, operator_id=2)
    assert len(result.queries) == 8
    assert detector.states[2].requery is False


def test_select_batch_with_requery_includes_history() -> None:
    ensemble = _trained_ensemble()
    detector = DriftDetector(ensemble=ensemble, config=DriftDetectorConfig())
    high = QueryCandidate(_seg(5.0), _seg(0.1))
    low = QueryCandidate(_seg(1.0), _seg(0.9))
    detector.record_window_queries([high, low], window_id=0, operator_id=2)

    pool = [QueryCandidate(_seg(1.0 + i * 0.05), _seg(0.5)) for i in range(15)]
    selected = select_batch_with_requery(
        pool,
        batch_size=8,
        operator_id=2,
        ensemble=ensemble,
        drift_detector=detector,
        requery_operator_id=2,
    )
    assert len(selected) == 8
    assert any(id(s) == id(high) for s in selected)
