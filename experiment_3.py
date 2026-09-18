#!/usr/bin/env python3
"""Experiment 3 — evaluate POP-VOI (decision-relevant elicitation).

**Primary metric (pre-specified):** final-query regret.
Secondary: queries-to-threshold (tie rate + censoring).

Claims: C-main-1, C-main-2, C-main-3 (see docs/CLAIMS.md).
Algorithm under test: ``plr.algorithm.POPVOI``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import Tensor

from plr.algorithm import (
    Acquisition,
    POPVOI,
    build_population_prior,
    default_mode_centers,
)
from plr.mdp import make_terrain_grid
from plr.users import answer, sample_gaussian_mixture_population

STRATEGIES = ("random", "bald", "voi", "oracle")

_ACQ = {
    "random": Acquisition.RANDOM,
    "bald": Acquisition.BALD,
    "voi": Acquisition.VOI,
    "voi_mean": Acquisition.VOI_MEAN,
    "oracle": Acquisition.ORACLE,
}


def _candidate_queries(env, n: int, K: int, g: torch.Generator) -> Tensor:
    S, _d = env.phi.shape
    idx = torch.randint(0, S, (n, K), generator=g)
    return env.phi[idx]


def run_user(
    env,
    particles_w: Tensor,
    particles_b: Tensor,
    candidates: Tensor,
    w_true: Tensor,
    b_true: Tensor,
    strategy: str,
    n_queries: int,
    regret_threshold: float,
    seed: int,
) -> dict:
    if strategy not in _ACQ:
        raise ValueError(f"unknown strategy {strategy!r}")
    algo = POPVOI(
        env,
        particles_w,
        particles_b,
        candidates,
        acquisition=_ACQ[strategy],
        seed=seed,
    )

    def answer_fn(phi: Tensor) -> Tensor:
        return answer(phi, w_true, b_true)

    log = algo.run(
        answer_fn,
        n_queries,
        w_true=w_true,
        b_true=b_true,
    )
    # Always pass b_true for oracle scoring inside select; harmless for others
    regrets = [float(e.true_regret_after) for e in log]
    queries_to_thresh = n_queries + 1
    reached = False
    for t, reg in enumerate(regrets):
        if reg <= regret_threshold:
            queries_to_thresh = t + 1
            reached = True
            break
    return {
        "strategy": strategy,
        "final_regret": regrets[-1] if regrets else float("nan"),
        "mean_regret": sum(regrets) / len(regrets) if regrets else float("nan"),
        "queries_to_threshold": queries_to_thresh,
        "reached_threshold": reached,
        "regret_curve": regrets,
    }


_SEED_OFFSET = {"random": 0, "bald": 1, "voi": 2, "voi_mean": 3, "oracle": 4}


def run_seed(
    seed: int,
    d: int = 8,
    n_users: int = 15,
    n_queries: int = 10,
    n_particles: int = 48,
    n_candidates: int = 24,
    regret_threshold: float = 1.0,
    strategies: tuple[str, ...] = STRATEGIES,
    out_file=None,
) -> list[dict]:
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)
    env = make_terrain_grid(n_rows=6, n_cols=6, d=d, seed=seed)
    centers = default_mode_centers(d, seed=seed)
    pop = sample_gaussian_mixture_population(
        n_users, d, mode_centers=centers, consistency_spread=0.3, seed=seed
    )
    particles_w, particles_b = build_population_prior(
        n_particles, d, centers, consistency_spread=0.3, seed=seed + 10_000
    )
    candidates = _candidate_queries(env, n_candidates, K=2, g=g)

    rows = []
    for u in range(n_users):
        for strategy in strategies:
            result = run_user(
                env,
                particles_w,
                particles_b,
                candidates,
                pop.w[u],
                pop.b[u],
                strategy,
                n_queries,
                regret_threshold,
                seed=seed * 1_000_003 + u * 97 + _SEED_OFFSET.get(strategy, 9),
            )
            result.update({"seed": seed, "user": u, "d": d})
            rows.append(result)
            print(
                json.dumps(
                    {
                        k: result[k]
                        for k in (
                            "seed",
                            "user",
                            "d",
                            "strategy",
                            "final_regret",
                            "queries_to_threshold",
                            "reached_threshold",
                        )
                    }
                ),
                flush=True,
            )
            if out_file is not None:
                out_file.write(json.dumps(result) + "\n")
                out_file.flush()
    return rows


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--seeds", type=int, nargs="*", default=None)
    p.add_argument("--d", type=int, default=8)
    p.add_argument("--n-users", type=int, default=15)
    p.add_argument("--n-queries", type=int, default=10)
    p.add_argument("--n-particles", type=int, default=48)
    p.add_argument("--n-candidates", type=int, default=24)
    p.add_argument("--regret-threshold", type=float, default=1.0)
    p.add_argument("--strategies", nargs="+", default=list(STRATEGIES))
    p.add_argument("--out", type=Path, default=Path("experiments/runs/exp3_power_d8.jsonl"))
    args = p.parse_args()

    seeds = args.seeds if args.seeds else [args.seed]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        for seed in seeds:
            run_seed(
                seed,
                d=args.d,
                n_users=args.n_users,
                n_queries=args.n_queries,
                n_particles=args.n_particles,
                n_candidates=args.n_candidates,
                regret_threshold=args.regret_threshold,
                strategies=tuple(args.strategies),
                out_file=f,
            )


if __name__ == "__main__":
    main()
