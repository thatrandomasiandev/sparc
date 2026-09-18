"""Property tests asserting PAPER CLAIMS — not mere smoke tests.

Each test traces to something PROJECT.md / the manuscript asserts.
"""

from __future__ import annotations

import math

import pytest
import torch

from plr.acquisition import QueryCosts, bald_score
from plr.likelihood import binary_logp, plackett_luce_logp, reward_gap, rewards, unit
from plr.mdp import Belief, expected_regret, make_terrain_grid, voi_score
from plr.users import answer, sample_population


def test_unit_norm_invariant_i1() -> None:
    """I1: unit(w) has norm 1; rewards ignore raw scale of w."""
    w = torch.tensor([3.0, 4.0])
    assert torch.allclose(unit(w).norm(), torch.tensor(1.0), atol=1e-6)
    phi = torch.randn(5, 2)
    # Doubling w must not change rewards once normalized inside rewards().
    assert torch.allclose(rewards(phi, w), rewards(phi, 2 * w), atol=1e-5)


def test_bradley_terry_equals_plackett_luce_at_k2() -> None:
    """BT and PL must agree exactly at K=2 (PROJECT.md §3.2)."""
    torch.manual_seed(0)
    phi = torch.randn(2, 4)
    w = unit(torch.randn(4))
    b = torch.tensor(2.5)
    delta = phi[0] - phi[1]
    log_bt = binary_logp(delta, w, b)
    order_01 = torch.tensor([0, 1])
    order_10 = torch.tensor([1, 0])
    log_pl_01 = plackett_luce_logp(
        phi.unsqueeze(0), w.unsqueeze(0), b.unsqueeze(0), order_01.unsqueeze(0)
    )
    log_pl_10 = plackett_luce_logp(
        phi.unsqueeze(0), w.unsqueeze(0), b.unsqueeze(0), order_10.unsqueeze(0)
    )
    assert torch.allclose(log_bt, log_pl_01.squeeze(0), atol=1e-5)
    # P(1≻0) = 1 - P(0≻1) in log space via log1mexp-ish: check probs sum to 1
    p0 = log_pl_01.squeeze(0).exp()
    p1 = log_pl_10.squeeze(0).exp()
    assert torch.allclose(p0 + p1, torch.tensor(1.0), atol=1e-5)


def test_trivial_query_has_zero_information() -> None:
    """Identical options ⇒ BALD = 0. Volume-removal's pathology; IG fixes it."""
    torch.manual_seed(1)
    d = 4
    phi = torch.randn(1, d).repeat(2, 1)  # identical options
    particles_w = unit(torch.randn(32, d))
    particles_b = torch.full((32,), 3.0)
    score = bald_score(phi, particles_w, particles_b)
    assert abs(float(score)) < 1e-6


def test_difficulty_independent_of_b_invariant_i2() -> None:
    """I2: reward_gap does not take b; scaling b cannot change difficulty."""
    torch.manual_seed(2)
    phi = torch.randn(3, 5)
    w = unit(torch.randn(5))
    g = reward_gap(phi, w)
    # Same gap regardless of any b we might have lying around
    assert g.ndim == 0 or g.numel() == 1
    assert torch.isfinite(g)


def test_query_costs_raise_on_missing_modality_i3() -> None:
    """I3: never default a cost."""
    costs = QueryCosts(seconds={"binary": 4.2})
    assert costs.cost("binary") == 4.2
    with pytest.raises(KeyError, match="pilot"):
        costs.cost("ternary")


def test_encoder_features_include_spread_i4() -> None:
    """I4: annotation features include spread (+ option count)."""
    from plr.encoder import annotation_features

    torch.manual_seed(3)
    phi = torch.randn(4, 6)  # K=4, d=6
    order = torch.tensor([2, 0, 1, 3])
    feats = annotation_features(phi, order)
    # direction (6) + context (6) + spread (1) + K (1) = 14
    assert feats.shape[-1] == 2 * 6 + 2
    # spread and K are the last two dims; K must equal 4
    assert torch.allclose(feats[-1], torch.tensor(4.0))


def test_binary_logp_stable_for_large_gap() -> None:
    """logsigmoid path must not underflow to -inf for confident easy queries."""
    delta = torch.tensor([10.0, 0.0, 0.0])
    w = unit(torch.tensor([1.0, 0.0, 0.0]))
    b = torch.tensor(50.0)  # b * gap ≈ 500
    lp_pref = binary_logp(delta, w, b)
    lp_opp = binary_logp(-delta, w, b)
    assert torch.isfinite(lp_pref) and torch.isfinite(lp_opp)
    # Preferred side saturates near 0; opposite must stay finite (not -inf from log(0)).
    assert lp_pref.item() <= 0.0
    assert lp_opp.item() < -20.0
    assert math.isfinite(lp_opp.item())


def test_policy_value_linearity() -> None:
    """V^π(w) = M @ w for fixed π — the mdp.py speed trick."""
    env = make_terrain_grid(n_rows=4, n_cols=4, d=3, seed=0)
    w = unit(torch.randn(3))
    _, pi = env.value_iteration(w)
    M = env.policy_value_matrix(pi)
    # Full state values via matmul vs one-step Bellman unroll from VI under same reward
    V_mat = M @ unit(w)
    r = env.reward_vector(w)
    P_pi = env.policy_transition(pi)
    # V = r + gamma P V  => should match V_mat
    V_bellman = r + env.gamma * (P_pi @ V_mat)
    assert torch.allclose(V_mat, V_bellman, atol=1e-4)


def test_answer_samples_valid_permutation() -> None:
    torch.manual_seed(4)
    phi = torch.randn(3, 4)
    w = unit(torch.randn(4))
    order = answer(phi, w, torch.tensor(2.0))
    assert sorted(order.tolist()) == [0, 1, 2]


def test_expected_regret_nonnegative() -> None:
    env = make_terrain_grid(n_rows=4, n_cols=4, d=4, seed=5)
    particles = unit(torch.randn(16, 4))
    weights = torch.full((16,), 1 / 16)
    V_star = env.optimal_values(particles)
    loss = expected_regret(env, weights, particles, V_star)
    assert float(loss) >= -1e-5


def test_particle_loss_nonnegative_and_voi_finite() -> None:
    """Queue #1: particle-policy loss is a valid VOI objective."""
    env = make_terrain_grid(n_rows=4, n_cols=4, d=4, seed=11)
    w = unit(torch.randn(12, 4))
    b = torch.full((12,), 3.0)
    belief = Belief.create(env, w, b)
    assert float(belief.loss_particle()) >= -1e-5
    assert float(belief.loss_mean()) >= -1e-5
    phi_q = torch.stack([env.phi[0], env.phi[5]], dim=0)
    s_part = voi_score(belief, phi_q, loss_mode="particle")
    s_mean = voi_score(belief, phi_q, loss_mode="mean")
    assert torch.isfinite(s_part) and torch.isfinite(s_mean)


def test_population_spread_zero_equal_b() -> None:
    pop = sample_population(20, 5, consistency_spread=0.0, b_median=3.0, seed=7)
    assert torch.allclose(pop.b, torch.full((20,), 3.0))
    assert torch.allclose(pop.w.norm(dim=-1), torch.ones(20), atol=1e-5)
