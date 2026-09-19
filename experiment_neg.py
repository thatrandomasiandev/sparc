#!/usr/bin/env python3
"""E-neg — negative control (Claim C-mech-7).

Labels come from a reward orthogonal to the task reward (same ‖w‖, w_label ⟂ w_task
in ambient space after projection). Prediction: VOI ≉ better than random on *task*
final regret (random−voi CI contains 0 or is negative).

If VOI still beats random here, the acquisition is chasing label likelihood that is
decision-irrelevant — a failure mode we must document.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import Tensor

from plr.algorithm import Acquisition, POPVOI, build_population_prior, default_mode_centers
from plr.likelihood import unit
from plr.mdp import make_terrain_grid
from plr.users import answer, sample_gaussian_mixture_population

STRATEGIES = ("random", "bald", "voi")
_ACQ = {
    "random": Acquisition.RANDOM,
    "bald": Acquisition.BALD,
    "voi": Acquisition.VOI,
}


def _orthogonal(w: Tensor, g: torch.Generator) -> Tensor:
    """Unit vector in the orthogonal complement of ``w`` (ambient R^d)."""
    d = w.shape[0]
    if d < 2:
        raise ValueError("need d>=2 for orthogonal negative control")
    # Householder-style: pick random v, subtract projection onto w
    v = torch.randn(d, generator=g)
    v = v - (v @ w) * w
    if float(v.norm()) < 1e-6:
        # Degenerate draw — use a coordinate axis not aligned with w
        e = torch.zeros(d)
        e[int(w.abs().argmin().item())] = 1.0
        v = e - (e @ w) * w
    return unit(v)


def _candidate_queries(env, n: int, K: int, g: torch.Generator) -> Tensor:
    S, _d = env.phi.shape
    idx = torch.randint(0, S, (n, K), generator=g)
    return env.phi[idx]


def run_seed(
    seed: int,
    *,
    d: int,
    n_users: int,
    n_queries: int,
    n_particles: int,
    n_candidates: int,
    out_file,
) -> None:
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)
    env = make_terrain_grid(n_rows=6, n_cols=6, d=d, seed=seed)
    centers = default_mode_centers(d, seed=seed)
    users = sample_gaussian_mixture_population(
        n_users, d, mode_centers=centers, consistency_spread=0.3, seed=seed
    )
    particles_w, particles_b = build_population_prior(
        n_particles, d, centers, consistency_spread=0.3, seed=seed + 10_000
    )
    candidates = _candidate_queries(env, n_candidates, K=2, g=g)

    for u in range(n_users):
        g_u = torch.Generator().manual_seed(seed * 999 + u)
        w_task = users.w[u]
        w_label = _orthogonal(w_task, g_u)
        b = users.b[u]
        for strategy in STRATEGIES:
            algo = POPVOI(
                env,
                particles_w,
                particles_b,
                candidates,
                acquisition=_ACQ[strategy],
                seed=seed * 1_000_003 + u * 97 + {"random": 0, "bald": 1, "voi": 2}[strategy],
            )

            def answer_fn(phi, _wl=w_label, _b=b):
                return answer(phi, _wl, _b)

            log = algo.run(answer_fn, n_queries, w_true=w_task, b_true=b)
            regrets = [float(e.true_regret_after) for e in log]
            row = {
                "seed": seed,
                "user": u,
                "d": d,
                "strategy": strategy,
                "final_regret": regrets[-1],
                "regret_curve": regrets,
                "control": "orthogonal_label",
            }
            print(
                json.dumps(
                    {k: row[k] for k in ("seed", "user", "strategy", "final_regret")}
                ),
                flush=True,
            )
            out_file.write(json.dumps(row) + "\n")
            out_file.flush()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4, 5])
    p.add_argument("--d", type=int, default=8)
    p.add_argument("--n-users", type=int, default=10)
    p.add_argument("--n-queries", type=int, default=10)
    p.add_argument("--n-particles", type=int, default=48)
    p.add_argument("--n-candidates", type=int, default=24)
    p.add_argument("--out", type=Path, default=Path("experiments/runs/exp_neg.jsonl"))
    args = p.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        for seed in args.seeds:
            run_seed(
                seed,
                d=args.d,
                n_users=args.n_users,
                n_queries=args.n_queries,
                n_particles=args.n_particles,
                n_candidates=args.n_candidates,
                out_file=f,
            )


if __name__ == "__main__":
    main()
