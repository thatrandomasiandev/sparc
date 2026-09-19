#!/usr/bin/env python3
"""E-act — does a better decision rule close the oracle gap? (Claim C-mech-4).

Acquisition fixed (VOI or BALD); only the acting rule changes after each update:
  mean | map | sample | softminimax

Primary (pre-specified): final-query regret under VOI acquisition.
Prediction: softminimax (or sample) < mean on final regret; shrinks VOI−oracle gap.
Kill: no rule beats mean → oracle gap is mostly acquisition, not acting.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import Tensor

from plr.algorithm import Acquisition, POPVOI, build_population_prior, default_mode_centers
from plr.mdp import make_terrain_grid
from plr.users import answer, sample_gaussian_mixture_population

DECISION_RULES = ("mean", "map", "sample", "softminimax")
ACQ = {"voi": Acquisition.VOI, "bald": Acquisition.BALD, "oracle": Acquisition.ORACLE}


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
    acquisition: str,
    decision_rule: str,
    n_queries: int,
    seed: int,
) -> dict:
    algo = POPVOI(
        env,
        particles_w,
        particles_b,
        candidates,
        acquisition=ACQ[acquisition],
        decision_rule=decision_rule,
        seed=seed,
    )

    def answer_fn(phi: Tensor) -> Tensor:
        return answer(phi, w_true, b_true)

    log = algo.run(answer_fn, n_queries, w_true=w_true, b_true=b_true)
    regrets = [float(e.true_regret_after) for e in log]
    return {
        "acquisition": acquisition,
        "decision_rule": decision_rule,
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
    acquisitions: tuple[str, ...],
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

    acq_offset = {"voi": 0, "bald": 10, "oracle": 20}
    rule_offset = {"mean": 0, "map": 1, "sample": 2, "softminimax": 3}
    for u in range(n_users):
        for acq in acquisitions:
            rules = ("mean",) if acq == "oracle" else DECISION_RULES
            for rule in rules:
                result = run_user(
                    env,
                    particles_w,
                    particles_b,
                    candidates,
                    users.w[u],
                    users.b[u],
                    acq,
                    rule,
                    n_queries,
                    seed=seed * 1_000_003
                    + u * 97
                    + acq_offset.get(acq, 30)
                    + rule_offset.get(rule, 0),
                )
                row = {"seed": seed, "user": u, "d": d, **result}
                print(
                    json.dumps(
                        {
                            k: row[k]
                            for k in (
                                "seed",
                                "user",
                                "acquisition",
                                "decision_rule",
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
    p.add_argument("--n-users", type=int, default=12)
    p.add_argument("--n-queries", type=int, default=10)
    p.add_argument("--n-particles", type=int, default=48)
    p.add_argument("--n-candidates", type=int, default=24)
    p.add_argument(
        "--acquisitions",
        nargs="+",
        default=["voi", "bald", "oracle"],
    )
    p.add_argument("--out", type=Path, default=Path("experiments/runs/exp_act.jsonl"))
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
                acquisitions=tuple(args.acquisitions),
                out_file=f,
            )


if __name__ == "__main__":
    main()
