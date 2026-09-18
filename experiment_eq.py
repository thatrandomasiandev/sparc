#!/usr/bin/env python3
"""E-eq / E-eq-d — policy equivalence audit (Claims C-mot-1, C-mot-2).

Two complementary metrics (both pre-specified as reportable; primary for C-mot-1
is *local_stable_rate*, because exact pairwise matches on dense random features
are a straw man):

1. pairwise_match_rate — P(π*(w)=π*(w')) for random unit pairs
2. local_stable_rate — P(π* unchanged under ε-perturbation of w)
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import torch

from plr.domains import policy_fingerprint, sample_unit_rewards
from plr.likelihood import unit
from plr.mdp import make_terrain_grid


def audit_equivalence(
    *,
    d: int,
    n_samples: int,
    seed: int,
    n_rows: int = 6,
    n_cols: int = 6,
) -> dict:
    env = make_terrain_grid(n_rows=n_rows, n_cols=n_cols, d=d, seed=seed)
    ws = sample_unit_rewards(n_samples, d, seed=seed + 1)
    fps = []
    for i in range(n_samples):
        _, pi = env.value_iteration(ws[i])
        fps.append(policy_fingerprint(pi))

    counts = Counter(fps)
    n_policies = len(counts)
    match = 0
    total = 0
    for i in range(n_samples):
        for j in range(i + 1, n_samples):
            total += 1
            if fps[i] == fps[j]:
                match += 1
    sizes = list(counts.values())
    return {
        "kind": "pairwise",
        "d": d,
        "seed": seed,
        "n_samples": n_samples,
        "n_unique_policies": n_policies,
        "mean_cluster_size": float(sum(sizes) / n_policies),
        "max_cluster_size": int(max(sizes)),
        "pairwise_match_rate": match / total if total else 0.0,
        "compression": 1.0 - n_policies / n_samples,
    }


def audit_local_stability(
    *,
    d: int,
    n_centers: int,
    n_perturb: int,
    eps: float,
    seed: int,
    n_rows: int = 6,
    n_cols: int = 6,
) -> dict:
    env = make_terrain_grid(n_rows=n_rows, n_cols=n_cols, d=d, seed=seed)
    g = torch.Generator().manual_seed(seed + 2)
    centers = sample_unit_rewards(n_centers, d, seed=seed + 1)
    unchanged = 0
    total = 0
    for i in range(n_centers):
        _, pi0 = env.value_iteration(centers[i])
        fp0 = policy_fingerprint(pi0)
        for _ in range(n_perturb):
            noise = torch.randn(d, generator=g)
            w = unit(centers[i] + eps * noise)
            _, pi = env.value_iteration(w)
            total += 1
            if policy_fingerprint(pi) == fp0:
                unchanged += 1
    return {
        "kind": "local_stability",
        "d": d,
        "seed": seed,
        "n_centers": n_centers,
        "n_perturb": n_perturb,
        "eps": eps,
        "local_stable_rate": unchanged / total if total else 0.0,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--n-samples", type=int, default=200)
    p.add_argument("--dims", type=int, nargs="+", default=[2, 4, 8, 12, 16])
    p.add_argument("--out", type=Path, default=Path("experiments/runs/exp_eq.jsonl"))
    p.add_argument("--eps", type=float, default=0.1)
    p.add_argument("--n-centers", type=int, default=40)
    p.add_argument("--n-perturb", type=int, default=25)
    args = p.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        for d in args.dims:
            for seed in args.seeds:
                row = audit_equivalence(d=d, n_samples=args.n_samples, seed=seed)
                f.write(json.dumps(row) + "\n")
                print(json.dumps(row), flush=True)
                loc = audit_local_stability(
                    d=d,
                    n_centers=args.n_centers,
                    n_perturb=args.n_perturb,
                    eps=args.eps,
                    seed=seed,
                )
                f.write(json.dumps(loc) + "\n")
                f.flush()
                print(json.dumps(loc), flush=True)


if __name__ == "__main__":
    main()
