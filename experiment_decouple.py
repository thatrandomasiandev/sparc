#!/usr/bin/env python3
"""E-decouple — structural irrelevance ρ vs BALD−VOI gap (Claim C-mech-1).

Domain: task features on reachable cells; decoy features only on unreachable cells.
Prediction: paired (BALD − VOI) final regret **increases** with ρ.
Kill: flat or decreasing.

Pre-specified primary: final-query regret; report paired BALD−VOI vs ρ.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.nn.functional import softmax

from plr.acquisition import bald_score
from plr.domains import make_decoupled_grid
from plr.likelihood import plackett_luce_logp, unit
from plr.mdp import (
    Belief,
    oracle_true_regret_score,
    posterior_mean_decision,
    voi_score,
)
from plr.users import answer, sample_gaussian_mixture_population


def _candidates(env, reachable, unreachable, n: int, g: torch.Generator) -> torch.Tensor:
    """Mix reachable and unreachable cells so decoy dims appear in the query pool."""
    pool = list(reachable) + list(unreachable)
    if not pool:
        pool = list(range(env.S))
    idx = torch.tensor(pool, dtype=torch.long)
    pick = idx[torch.randint(0, len(idx), (n, 2), generator=g)]
    return env.phi[pick]


def _bayes_update(belief: Belief, phi, order) -> Belief:
    P = belief.particles_w.shape[0]
    logp = plackett_luce_logp(
        phi.unsqueeze(0).expand(P, -1, -1),
        belief.particles_w,
        belief.particles_b,
        order.unsqueeze(0).expand(P, -1),
    )
    new_w = softmax(torch.log(belief.weights.clamp_min(1e-12)) + logp, dim=0)
    return belief.with_weights(new_w)


def run_user(
    belief0: Belief,
    candidates: torch.Tensor,
    w_true: torch.Tensor,
    b_true: torch.Tensor,
    strategy: str,
    n_queries: int,
    g: torch.Generator,
) -> dict:
    belief = belief0.with_weights(torch.full_like(belief0.weights, 1.0 / belief0.weights.numel()))
    regrets = []
    for _t in range(n_queries):
        if strategy == "random":
            q = int(torch.randint(0, candidates.shape[0], (1,), generator=g).item())
        else:
            scores = []
            for qi in range(candidates.shape[0]):
                phi = candidates[qi]
                if strategy == "bald":
                    s = bald_score(
                        phi, belief.particles_w, belief.particles_b, belief.weights
                    )
                elif strategy == "voi":
                    s = voi_score(belief, phi, loss_mode="particle")
                elif strategy == "oracle":
                    s = oracle_true_regret_score(belief, phi, w_true, b_true)
                else:
                    raise ValueError(strategy)
                scores.append(float(s))
            q = int(torch.tensor(scores).argmax().item())
        phi = candidates[q]
        order = answer(phi, w_true, b_true, generator=g)
        belief = _bayes_update(belief, phi, order)
        decision = posterior_mean_decision(belief.weights, belief.particles_w)
        regrets.append(float(belief.env.regret(w_true, decision)))
    return {
        "strategy": strategy,
        "final_regret": regrets[-1],
        "regret_curve": regrets,
    }


def run_rho(
    rho: float,
    seed: int,
    *,
    d_total: int = 8,
    n_users: int = 8,
    n_queries: int = 8,
    n_particles: int = 32,
    n_candidates: int = 16,
) -> list[dict]:
    d_decoy = int(round(rho * d_total))
    d_task = d_total - d_decoy
    if d_task < 1:
        d_task, d_decoy = 1, d_total - 1

    env, meta = make_decoupled_grid(
        d_task=d_task, d_decoy=d_decoy, seed=seed, n_rows=6, n_cols=6
    )
    d = d_task + d_decoy
    # Modes live in full R^d but only task dims affect π*; decoy still affects answers
    # when queries touch unreachable cells.
    centers = unit(torch.randn(3, d, generator=torch.Generator().manual_seed(seed)))
    pop = sample_gaussian_mixture_population(
        n_users, d, mode_centers=centers, consistency_spread=0.3, seed=seed
    )
    particles = sample_gaussian_mixture_population(
        n_particles, d, mode_centers=centers, consistency_spread=0.3, seed=seed + 999
    )
    g = torch.Generator().manual_seed(seed)
    candidates = _candidates(
        env, meta["reachable"], meta["unreachable"], n_candidates, g
    )
    belief0 = Belief.create(env, particles.w, particles.b)

    rows = []
    for u in range(n_users):
        for strategy in ("random", "bald", "voi", "oracle"):
            g_u = torch.Generator().manual_seed(seed * 1_000_003 + u * 97)
            result = run_user(
                belief0, candidates, pop.w[u], pop.b[u], strategy, n_queries, g_u
            )
            result.update(
                {
                    "seed": seed,
                    "user": u,
                    "rho": meta["rho"],
                    "d_task": d_task,
                    "d_decoy": d_decoy,
                    "d": d,
                }
            )
            rows.append(result)
            print(
                json.dumps(
                    {
                        k: result[k]
                        for k in (
                            "seed",
                            "user",
                            "rho",
                            "strategy",
                            "final_regret",
                        )
                    }
                ),
                flush=True,
            )
    return rows


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--rhos", type=float, nargs="+", default=[0.0, 0.25, 0.5, 0.75])
    p.add_argument("--n-users", type=int, default=8)
    p.add_argument("--n-queries", type=int, default=8)
    p.add_argument("--n-particles", type=int, default=32)
    p.add_argument("--n-candidates", type=int, default=16)
    p.add_argument("--d-total", type=int, default=8)
    p.add_argument(
        "--out", type=Path, default=Path("experiments/runs/exp_decouple.jsonl")
    )
    args = p.parse_args()

    # Pre-specify in RESULTS before launch — primary = final_regret.
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        for rho in args.rhos:
            for seed in args.seeds:
                for row in run_rho(
                    rho,
                    seed,
                    d_total=args.d_total,
                    n_users=args.n_users,
                    n_queries=args.n_queries,
                    n_particles=args.n_particles,
                    n_candidates=args.n_candidates,
                ):
                    f.write(json.dumps(row) + "\n")
                    f.flush()


if __name__ == "__main__":
    main()
