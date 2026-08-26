"""Tests for Pillar 3 confidence-gated policy bounding (toy analytic cases)."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("torch")

from sparc.env.segments import TrajectorySegment
from sparc.policy import (
    BoundingConfig,
    BoundingMode,
    ConfidenceGate,
    UpdateAction,
    apply_bounding,
    apply_minibatch_bounding,
    compute_disagreement_percentile,
    compute_threshold,
    effective_rewards,
)
from sparc.reward_model import RewardEnsemble, RewardModelConfig


def test_passthrough_when_disagreement_below_tau() -> None:
    res = apply_bounding(r_hat=5.0, disagreement=0.1, tau=0.5, mode=BoundingMode.SHRINK)
    assert res.action == UpdateAction.APPLY
    assert res.reward == 5.0


def test_shrink_formula() -> None:
    """Analytic: alpha = tau/D, r' = alpha * r_hat."""
    tau, d, r = 0.5, 2.0, 10.0
    res = apply_bounding(r, d, tau, BoundingMode.SHRINK)
    assert res.action == UpdateAction.APPLY
    assert res.reward == pytest.approx((tau / d) * r)


def test_freeze_skips_transition() -> None:
    res = apply_bounding(r_hat=3.0, disagreement=1.0, tau=0.2, mode=BoundingMode.FREEZE)
    assert res.action == UpdateAction.SKIP_TRANSITION
    assert res.skip_transition


def test_pass_through_mode_when_above_tau() -> None:
    res = apply_bounding(r_hat=7.0, disagreement=2.0, tau=0.5, mode=BoundingMode.PASS_THROUGH)
    assert res.reward == 7.0


def test_minibatch_skip_when_gated_fraction_exceeds_f_max() -> None:
    cfg = BoundingConfig(f_max=0.5, mode=BoundingMode.SHRINK)
    rewards = [1.0, 2.0, 3.0, 4.0]
    disagreements = [0.1, 2.0, 2.5, 3.0]  # 3/4 > tau=0.5
    tau = 0.5
    results, skip_all = apply_minibatch_bounding(rewards, disagreements, tau, cfg)
    assert skip_all
    assert all(r.action == UpdateAction.SKIP_MINIBATCH for r in results)


def test_minibatch_per_transition_shrink() -> None:
    cfg = BoundingConfig(f_max=0.5, mode=BoundingMode.SHRINK)
    rewards = [10.0, 10.0]
    disagreements = [0.1, 2.0]
    tau = 0.5
    results, skip_all = apply_minibatch_bounding(rewards, disagreements, tau, cfg)
    assert not skip_all
    assert results[0].reward == 10.0
    assert results[1].reward == pytest.approx((tau / 2.0) * 10.0)


def test_q90_and_threshold() -> None:
    d_vals = [0.0, 0.1, 0.2, 0.5, 1.0, 2.0]
    q90 = compute_disagreement_percentile(d_vals, percentile=90.0)
    assert q90 == pytest.approx(float(np.percentile(d_vals, 90.0)))
    cfg = BoundingConfig(tau_0=0.01, kappa=1.0)
    tau = compute_threshold(q90, cfg)
    assert tau == pytest.approx(0.01 + q90)


def _seg(speed: float) -> TrajectorySegment:
    states = np.zeros((4, 8), dtype=np.float64)
    states[:, 3] = speed
    states[:, 5:8] = 1.0
    actions = np.zeros((4, 2), dtype=np.float64)
    return TrajectorySegment(states=states, actions=actions)


def test_confidence_gate_window_update() -> None:
    cfg_rm = RewardModelConfig(embed_dim=8, latent_dim=4, hidden_dim=16, ensemble_size=3)
    ensemble = RewardEnsemble.create(8, 2, 4, config=cfg_rm, seed=0)
    gate = ConfidenceGate(ensemble=ensemble, config=BoundingConfig(tau_0=0.0, kappa=1.0))
    segments = [_seg(1.0), _seg(2.0), _seg(3.0)]
    tau = gate.update_threshold(segments)
    assert tau >= 0.0
    assert gate.q90 >= 0.0
    results, skip = gate.bound_minibatch(segments, [1.0, 1.0, 1.0])
    assert len(results) == 3
    assert isinstance(skip, bool)


def test_effective_rewards_marks_skips() -> None:
    cfg = BoundingConfig(mode=BoundingMode.FREEZE)
    rewards = [1.0, 2.0]
    disagreements = [0.1, 5.0]
    results, _ = apply_minibatch_bounding(rewards, disagreements, tau=0.5, config=cfg)
    eff = effective_rewards(results)
    assert eff[0] == 1.0
    assert eff[1] is None
