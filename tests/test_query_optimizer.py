"""Tests for Pillar 2 batch query optimizer."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("torch")

from sparc.annotator_sim import SyntheticAnnotator
from sparc.env.dataset import PreferenceDataset, PreferenceQuery
from sparc.env.segments import TrajectorySegment
from sparc.query_optimizer import (
    CommsWindowOptimizer,
    QueryCandidate,
    QueryOptimizerConfig,
    benchmark_batch_selection,
    operator_for_window,
    score_eig,
    select_batch,
)
from sparc.reward_model import RewardEnsemble, RewardModelConfig


def _seg(speed: float, tag: float = 0.0) -> TrajectorySegment:
    states = np.zeros((4, 8), dtype=np.float64)
    states[:, 3] = speed
    states[:, 5:8] = 1.0 + tag
    actions = np.zeros((4, 2), dtype=np.float64)
    return TrajectorySegment(states=states, actions=actions)


def _candidates(n: int) -> list[QueryCandidate]:
    out: list[QueryCandidate] = []
    for k in range(n):
        out.append(
            QueryCandidate(
                segment_0=_seg(1.0 + k * 0.05, tag=k * 0.01),
                segment_1=_seg(0.5, tag=k * 0.02),
            )
        )
    return out


def _trained_ensemble(seed: int = 0) -> RewardEnsemble:
    ds = PreferenceDataset()
    ann = SyntheticAnnotator(p_mistake=0.0, p_skip=0.0, seed=seed)
    for k in range(12):
        q = PreferenceQuery(
            _seg(1.0 + k * 0.1),
            _seg(0.5),
            operator_id=1 + (k % 3),
            window_id=k,
            env_step=k * 50,
        )
        ds.add_query(q, ann.label_query(q))
    cfg = RewardModelConfig(embed_dim=16, latent_dim=4, hidden_dim=32, ensemble_size=3)
    model = RewardEnsemble.create(8, 2, 4, config=cfg, seed=seed)
    model.train_epoch(ds)
    return model


def test_operator_for_window_rotation() -> None:
    assert operator_for_window(0) == 1
    assert operator_for_window(2) == 3
    assert operator_for_window(3) == 1


def test_score_eig_nonnegative() -> None:
    model = _trained_ensemble()
    c = _candidates(1)[0]
    eig = score_eig(c, operator_id=1, ensemble=model)
    assert eig >= -1e-6


def test_select_batch_size_and_unique() -> None:
    model = _trained_ensemble()
    cands = _candidates(20)
    selected = select_batch(cands, batch_size=8, operator_id=1, ensemble=model)
    assert len(selected) == 8
    assert len({id(s) for s in selected}) == 8


def test_comms_window_outstanding_guard() -> None:
    model = _trained_ensemble()
    opt = CommsWindowOptimizer(
        ensemble=model,
        config=QueryOptimizerConfig(batch_size=4),
    )
    cands = _candidates(15)
    opt.on_window_open(window_id=0, candidates=cands, env_step=0)
    with pytest.raises(RuntimeError, match="outstanding"):
        opt.on_window_open(window_id=1, candidates=cands, env_step=100)
    opt.on_labels_received()
    result = opt.on_window_open(window_id=1, candidates=cands, env_step=100)
    assert len(result.queries) == 4


def test_benchmark_under_wall_clock_budget() -> None:
    """ROADMAP Week 11: batch selection must finish << comms window wall-clock."""
    cfg_rm = RewardModelConfig(
        embed_dim=128,
        latent_dim=16,
        hidden_dim=256,
        ensemble_size=7,
    )
    model = RewardEnsemble.create(8, 2, 4, config=cfg_rm, seed=7)
    ds = PreferenceDataset()
    ann = SyntheticAnnotator(p_mistake=0.0, p_skip=0.0, seed=7)
    for k in range(20):
        q = PreferenceQuery(_seg(1 + k * 0.05), _seg(0.5), 1, 0, k)
        ds.add_query(q, ann.label_query(q))
    model.train_epoch(ds)

    cands = _candidates(500)
    opt_cfg = QueryOptimizerConfig(batch_size=8, max_candidates=500)
    elapsed = benchmark_batch_selection(model, cands, opt_cfg, operator_id=1)
    assert elapsed < opt_cfg.wall_clock_budget_sec, (
        f"SelectBatch took {elapsed:.3f}s, budget {opt_cfg.wall_clock_budget_sec}s"
    )
