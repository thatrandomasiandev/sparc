#!/usr/bin/env python3
"""E-prior-shift — misspecified population prior (Claim C-mech-3).

Fit / sample particles from mode centers A; evaluate on users from shifted centers B.
Same VOI acquisition. Prediction: population-A prior loses vs matched prior; may
approach uniform. Kill for overclaim: if shifted prior still matches matched prior,
prior geometry does not matter (unlikely).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import Tensor

from plr.algorithm import Acquisition, POPVOI, default_mode_centers
from plr.likelihood import unit
from plr.mdp import make_terrain_grid
from plr.users import answer, sample_gaussian_mixture_population, sample_population


def _candidate_queries(env, n: int, K: int, g: torch.Generator) -> Tensor:
    S, _d = env.phi.shape
    idx = torch.randint(0, S, (n, K), generator=g)
    return env.phi[idx]


def _shift_centers(centers: Tensor, seed: int, angle: float = 0.9) -> Tensor:
    """Rotate centers toward a fresh random frame (large shift)."""
    g = torch.Generator().manual_seed(seed + 12345)
    noise = torch.randn_like(centers, generator=g)
    return unit((1 - angle) * centers + angle * noise)


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
    centers_a = default_mode_centers(d, seed=seed)
    centers_b = _shift_centers(centers_a, seed=seed)
    users = sample_gaussian_mixture_population(
        n_users, d, mode_centers=centers_b, consistency_spread=0.3, seed=seed
    )
    matched = sample_gaussian_mixture_population(
        n_particles, d, mode_centers=centers_b, consistency_spread=0.3, seed=seed + 10_000
    )
    shifted = sample_gaussian_mixture_population(
        n_particles, d, mode_centers=centers_a, consistency_spread=0.3, seed=seed + 20_000
    )
    uniform = sample_population(n_particles, d, consistency_spread=0.3, seed=seed + 30_000)
    candidates = _candidate_queries(env, n_candidates, K=2, g=g)

    priors = {
        "matched": (matched.w, matched.b),
        "shifted": (shifted.w, shifted.b),
        "uniform": (uniform.w, uniform.b),
    }
    for u in range(n_users):
        for name, (pw, pb) in priors.items():
            algo = POPVOI(
                env,
                pw,
                pb,
                candidates,
                acquisition=Acquisition.VOI,
                seed=seed * 1_000_003 + u * 97 + {"matched": 0, "shifted": 1, "uniform": 2}[name],
            )
            log = algo.run(
                lambda phi, wu=users.w[u], bu=users.b[u]: answer(phi, wu, bu),
                n_queries,
                w_true=users.w[u],
                b_true=users.b[u],
            )
            regrets = [float(e.true_regret_after) for e in log]
            row = {
                "seed": seed,
                "user": u,
                "d": d,
                "prior": name,
                "strategy": "voi",
                "final_regret": regrets[-1],
                "regret_curve": regrets,
            }
            print(
                json.dumps(
                    {k: row[k] for k in ("seed", "user", "prior", "final_regret")}
                ),
                flush=True,
            )
            out_file.write(json.dumps(row) + "\n")
            out_file.flush()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4, 5])
    p.add_argument("--d", type=int, default=8)
    p.add_argument("--n-users", type=int, default=12)
    p.add_argument("--n-queries", type=int, default=10)
    p.add_argument("--n-particles", type=int, default=48)
    p.add_argument("--n-candidates", type=int, default=24)
    p.add_argument("--out", type=Path, default=Path("experiments/runs/exp_prior_shift.jsonl"))
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
