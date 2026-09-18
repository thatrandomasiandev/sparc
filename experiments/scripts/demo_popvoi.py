#!/usr/bin/env python3
"""Smoke demo: POP-VOI vs random vs BALD on a tiny grid (not a paper claim — I8).

Usage:
  python experiments/scripts/demo_popvoi.py
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from plr.algorithm import POPVOI, Acquisition, build_population_prior, default_mode_centers
from plr.mdp import make_terrain_grid
from plr.users import answer, sample_gaussian_mixture_population


def main() -> None:
    seed = 0
    d = 6
    torch.manual_seed(seed)
    env = make_terrain_grid(n_rows=5, n_cols=5, d=d, seed=seed)
    centers = default_mode_centers(d, seed=seed)
    pw, pb = build_population_prior(24, d, centers, seed=seed)
    g = torch.Generator().manual_seed(seed)
    candidates = env.phi[torch.randint(0, env.S, (20, 2), generator=g)]
    user = sample_gaussian_mixture_population(1, d, mode_centers=centers, seed=seed + 7)
    w_true, b_true = user.w[0], user.b[0]

    def answer_fn(phi):
        return answer(phi, w_true, b_true)

    out = {}
    for acq in (Acquisition.RANDOM, Acquisition.BALD, Acquisition.VOI, Acquisition.ORACLE):
        algo = POPVOI(env, pw, pb, candidates, acquisition=acq, seed=seed)
        log = algo.run(answer_fn, n_queries=8, w_true=w_true, b_true=b_true)
        out[acq.value] = {
            "final_regret": log[-1].true_regret_after,
            "curve": [e.true_regret_after for e in log],
        }
        print(f"{acq.value:8s}  final_regret={log[-1].true_regret_after:.4f}")

    path = Path("experiments/runs/demo_popvoi.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2) + "\n")
    print(f"Wrote {path} (demo only — not a Claim ID result)")


if __name__ == "__main__":
    main()
