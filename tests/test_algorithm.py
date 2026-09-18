"""Tests for the POP-VOI algorithm (paper method)."""

from __future__ import annotations

import torch

from plr.algorithm import POPVOI, Acquisition, build_population_prior, default_mode_centers
from plr.mdp import make_terrain_grid
from plr.users import answer, sample_gaussian_mixture_population


def _toy_setup(seed: int = 0):
    torch.manual_seed(seed)
    d = 4
    env = make_terrain_grid(n_rows=4, n_cols=4, d=d, seed=seed)
    centers = default_mode_centers(d, seed=seed)
    pw, pb = build_population_prior(16, d, centers, seed=seed)
    g = torch.Generator().manual_seed(seed)
    idx = torch.randint(0, env.S, (12, 2), generator=g)
    candidates = env.phi[idx]
    user = sample_gaussian_mixture_population(1, d, mode_centers=centers, seed=seed + 1)
    return env, pw, pb, candidates, user.w[0], user.b[0]


def test_popvoi_selects_and_updates() -> None:
    env, pw, pb, cand, w_true, b_true = _toy_setup(0)
    algo = POPVOI(env, pw, pb, cand, acquisition=Acquisition.VOI, seed=0)
    q, score = algo.select_query()
    assert 0 <= q < cand.shape[0]
    assert score == score  # not NaN
    order = answer(cand[q], w_true, b_true)
    before = algo.belief.weights.clone()
    algo.update(q, order)
    assert not torch.allclose(before, algo.belief.weights)
    assert torch.allclose(algo.belief.weights.sum(), torch.tensor(1.0), atol=1e-5)


def test_popvoi_run_reduces_or_tracks_regret() -> None:
    """Smoke: algorithm runs; oracle ends ≤ random on this seed (not a paper claim)."""
    env, pw, pb, cand, w_true, b_true = _toy_setup(1)

    def make_answer(w, b):
        def fn(phi):
            return answer(phi, w, b)

        return fn

    regrets = {}
    for acq in (Acquisition.RANDOM, Acquisition.VOI, Acquisition.ORACLE):
        algo = POPVOI(env, pw, pb, cand, acquisition=acq, seed=1)
        log = algo.run(
            make_answer(w_true, b_true),
            n_queries=5,
            w_true=w_true,
            b_true=b_true,
        )
        assert len(log) == 5
        regrets[acq] = log[-1].true_regret_after

    assert regrets[Acquisition.ORACLE] is not None
    # Soft check: oracle should not be wildly worse than random on this toy
    assert regrets[Acquisition.ORACLE] <= regrets[Acquisition.RANDOM] + 1.0


def test_decide_w_is_unit() -> None:
    env, pw, pb, cand, _, _ = _toy_setup(2)
    algo = POPVOI(env, pw, pb, cand, seed=2)
    w = algo.decide_w()
    assert torch.allclose(w.norm(), torch.tensor(1.0), atol=1e-5)


def test_oracle_requires_truth() -> None:
    env, pw, pb, cand, _, _ = _toy_setup(3)
    algo = POPVOI(env, pw, pb, cand, acquisition=Acquisition.ORACLE, seed=3)
    try:
        algo.select_query()
        raised = False
    except ValueError:
        raised = True
    assert raised
