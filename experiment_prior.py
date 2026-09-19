#!/usr/bin/env python3
"""E-prior — is the population prior necessary for POP-VOI? (Claim C-mech-2).

Same VOI acquisition + same eval users; only the particle prior changes:
  (a) population mixture  (b) uniform-on-sphere / isotropic Gaussian
  (c) cheat: particles = true user only (upper prior)

Primary (pre-specified): final-query regret.
Prediction: (a) < (b) on paired final regret; (c) is a soft ceiling.
Kill: (a) not better than (b) → abstract must not claim prior makes VOI usable.
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


def _uniform_sphere_prior(n: int, d: int, seed: int) -> tuple[Tensor, Tensor]:
    pop = sample_population(n, d, consistency_spread=0.3, seed=seed)
    return pop.w, pop.b


def _population_prior(n: int, d: int, centers: Tensor, seed: int) -> tuple[Tensor, Tensor]:
    pop = sample_gaussian_mixture_population(
        n, d, mode_centers=centers, consistency_spread=0.3, seed=seed
    )
    return pop.w, pop.b


def _cheat_prior(w_true: Tensor, b_true: Tensor, n: int) -> tuple[Tensor, Tensor]:
    """All particles equal the true user — oracle-level prior, not usable in practice."""
    w = unit(w_true).unsqueeze(0).expand(n, -1).clone()
    b = torch.full((n,), float(b_true))
    return w, b


def run_user(
    env,
    particles_w: Tensor,
    particles_b: Tensor,
    candidates: Tensor,
    w_true: Tensor,
    b_true: Tensor,
    n_queries: int,
    seed: int,
) -> dict:
    algo = POPVOI(
        env,
        particles_w,
        particles_b,
        candidates,
        acquisition=Acquisition.VOI,
        seed=seed,
    )

    def answer_fn(phi: Tensor) -> Tensor:
        return answer(phi, w_true, b_true)

    log = algo.run(answer_fn, n_queries, w_true=w_true, b_true=b_true)
    regrets = [float(e.true_regret_after) for e in log]
    return {
        "final_regret": regrets[-1] if regrets else float("nan"),
        "regret_curve": regrets,
    }


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
    candidates = _candidate_queries(env, n_candidates, K=2, g=g)
    pop_w, pop_b = _population_prior(n_particles, d, centers, seed=seed + 10_000)
    uni_w, uni_b = _uniform_sphere_prior(n_particles, d, seed=seed + 20_000)

    prior_offset = {"population": 0, "uniform": 1, "cheat": 2}
    for u in range(n_users):
        cheat_w, cheat_b = _cheat_prior(users.w[u], users.b[u], n_particles)
        for prior_name, pw, pb in (
            ("population", pop_w, pop_b),
            ("uniform", uni_w, uni_b),
            ("cheat", cheat_w, cheat_b),
        ):
            result = run_user(
                env,
                pw,
                pb,
                candidates,
                users.w[u],
                users.b[u],
                n_queries,
                seed=seed * 1_000_003 + u * 97 + prior_offset[prior_name],
            )
            row = {
                "seed": seed,
                "user": u,
                "d": d,
                "prior": prior_name,
                "strategy": "voi",
                **result,
            }
            print(
                json.dumps(
                    {
                        k: row[k]
                        for k in ("seed", "user", "prior", "final_regret")
                    }
                ),
                flush=True,
            )
            out_file.write(json.dumps(row) + "\n")
            out_file.flush()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--d", type=int, default=8)
    p.add_argument("--n-users", type=int, default=12)
    p.add_argument("--n-queries", type=int, default=10)
    p.add_argument("--n-particles", type=int, default=48)
    p.add_argument("--n-candidates", type=int, default=24)
    p.add_argument("--out", type=Path, default=Path("experiments/runs/exp_prior.jsonl"))
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
