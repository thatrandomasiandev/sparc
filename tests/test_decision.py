"""Decision-rule helpers used by E-act (C-mech-4)."""

import torch

from plr.algorithm import POPVOI, Acquisition, build_population_prior, default_mode_centers
from plr.mdp import Belief, decide_w, make_terrain_grid
from plr.users import answer


def test_decide_w_rules_unit_norm():
    env = make_terrain_grid(n_rows=4, n_cols=4, d=3, seed=0)
    centers = default_mode_centers(3, seed=0)
    pw, pb = build_population_prior(12, 3, centers, seed=1)
    belief = Belief.create(env, pw, pb)
    for rule in ("mean", "map", "sample", "softminimax"):
        w = decide_w(belief, rule)
        assert w.shape == (3,)
        assert abs(float(w.norm()) - 1.0) < 1e-5


def test_popvoi_decision_rule_affects_regret_path():
    """Different rules must be pluggable without breaking the select/update loop."""
    env = make_terrain_grid(n_rows=4, n_cols=4, d=3, seed=0)
    centers = default_mode_centers(3, seed=0)
    pw, pb = build_population_prior(8, 3, centers, seed=2)
    g = torch.Generator().manual_seed(0)
    idx = torch.randint(0, env.S, (6, 2), generator=g)
    candidates = env.phi[idx]
    w_true, b_true = pw[0], pb[0]
    regrets = {}
    for rule in ("mean", "softminimax"):
        algo = POPVOI(
            env, pw, pb, candidates,
            acquisition=Acquisition.VOI,
            decision_rule=rule,
            seed=0,
        )
        log = algo.run(lambda phi: answer(phi, w_true, b_true), 3, w_true=w_true)
        regrets[rule] = [e.true_regret_after for e in log]
        assert len(log) == 3
    # Soft sanity: both produce finite regrets
    assert all(r == r for curve in regrets.values() for r in curve)
