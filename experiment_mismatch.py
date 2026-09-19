#!/usr/bin/env python3
"""E-mismatch — graceful degradation under non-PL users (Claim C-mech-10).

Estimator still assumes Plackett–Luce; answers come from:
  pl (matched) | lex | satisficing | fatigue

Primary (pre-specified): paired (BALD − VOI) final-query regret per mismatch type.
Prediction: VOI advantage shrinks under mismatch but does not reverse vs random.
Kill: VOI collapses below random under any mismatch → method is model-brittle.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import Tensor

from plr.algorithm import Acquisition, POPVOI, build_population_prior, default_mode_centers
from plr.mdp import make_terrain_grid
from plr.users import (
    answer,
    answer_fatigue,
    answer_lexicographic,
    answer_satisficing,
    sample_gaussian_mixture_population,
)

STRATEGIES = ("random", "bald", "voi")
MISMATCH = ("pl", "lex", "satisficing", "fatigue")
_ACQ = {
    "random": Acquisition.RANDOM,
    "bald": Acquisition.BALD,
    "voi": Acquisition.VOI,
}


def _candidate_queries(env, n: int, K: int, g: torch.Generator) -> Tensor:
    S, _d = env.phi.shape
    idx = torch.randint(0, S, (n, K), generator=g)
    return env.phi[idx]


def _answer_fn(kind: str, w_true: Tensor, b_true: Tensor, g: torch.Generator):
    step = {"t": 0}

    def fn(phi: Tensor) -> Tensor:
        t = step["t"]
        step["t"] = t + 1
        if kind == "pl":
            return answer(phi, w_true, b_true, generator=g)
        if kind == "lex":
            return answer_lexicographic(phi, w_true, generator=g)
        if kind == "satisficing":
            return answer_satisficing(phi, w_true, b_true, generator=g)
        if kind == "fatigue":
            return answer_fatigue(phi, w_true, b_true, step=t, generator=g)
        raise ValueError(kind)

    return fn


def run_user(
    env,
    particles_w: Tensor,
    particles_b: Tensor,
    candidates: Tensor,
    w_true: Tensor,
    b_true: Tensor,
    strategy: str,
    mismatch: str,
    n_queries: int,
    seed: int,
) -> dict:
    algo = POPVOI(
        env,
        particles_w,
        particles_b,
        candidates,
        acquisition=_ACQ[strategy],
        seed=seed,
    )
    g = torch.Generator().manual_seed(seed + 17)
    log = algo.run(
        _answer_fn(mismatch, w_true, b_true, g),
        n_queries,
        w_true=w_true,
        b_true=b_true,
    )
    regrets = [float(e.true_regret_after) for e in log]
    return {
        "strategy": strategy,
        "mismatch": mismatch,
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
    particles_w, particles_b = build_population_prior(
        n_particles, d, centers, consistency_spread=0.3, seed=seed + 10_000
    )
    candidates = _candidate_queries(env, n_candidates, K=2, g=g)

    mismatch_offset = {"pl": 0, "lex": 10, "satisficing": 20, "fatigue": 30}
    strategy_offset = {"random": 0, "bald": 1, "voi": 2}
    for u in range(n_users):
        for mismatch in MISMATCH:
            for strategy in STRATEGIES:
                result = run_user(
                    env,
                    particles_w,
                    particles_b,
                    candidates,
                    users.w[u],
                    users.b[u],
                    strategy,
                    mismatch,
                    n_queries,
                    seed=seed * 1_000_003
                    + u * 97
                    + mismatch_offset[mismatch]
                    + strategy_offset[strategy],
                )
                row = {"seed": seed, "user": u, "d": d, **result}
                print(
                    json.dumps(
                        {
                            k: row[k]
                            for k in (
                                "seed",
                                "user",
                                "mismatch",
                                "strategy",
                                "final_regret",
                            )
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
    p.add_argument("--n-users", type=int, default=10)
    p.add_argument("--n-queries", type=int, default=10)
    p.add_argument("--n-particles", type=int, default=48)
    p.add_argument("--n-candidates", type=int, default=24)
    p.add_argument("--out", type=Path, default=Path("experiments/runs/exp_mismatch.jsonl"))
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
