#!/usr/bin/env python3
"""Aggregate E-decouple: paired BALD−VOI final regret vs ρ."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def load(path: Path) -> list[dict]:
    rows = []
    with path.open() as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def bootstrap_mean(x: np.ndarray, n_boot: int = 10_000, seed: int = 0):
    rng = np.random.default_rng(seed)
    means = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(n_boot)]
    means = np.sort(means)
    return float(x.mean()), float(means[int(0.025 * n_boot)]), float(means[int(0.975 * n_boot)])


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("results", type=Path)
    p.add_argument("--n-boot", type=int, default=10_000)
    args = p.parse_args()
    rows = load(args.results)

    # key: (rho, seed, user) -> strategy -> regret
    by = defaultdict(dict)
    for r in rows:
        by[(r["rho"], r["seed"], r["user"])][r["strategy"]] = r["final_regret"]

    rhos = sorted({k[0] for k in by})
    print("=== E-decouple primary: paired BALD − VOI final regret vs ρ ===")
    slope_points = []
    for rho in rhos:
        diffs = []
        for (rr, _seed, _user), block in by.items():
            if rr != rho:
                continue
            if "bald" in block and "voi" in block:
                diffs.append(block["bald"] - block["voi"])
        diffs = np.asarray(diffs, dtype=float)
        mean, lo, hi = bootstrap_mean(diffs, n_boot=args.n_boot)
        slope_points.append((rho, mean))
        print(
            f"  ρ={rho:.2f}  n={len(diffs)}  BALD−VOI={mean:+.4f}  "
            f"95% CI [{lo:+.4f}, {hi:+.4f}]"
        )
        # Also means per strategy
        for strat in ("random", "bald", "voi", "oracle"):
            vals = [
                block[strat]
                for (rr, _, _), block in by.items()
                if rr == rho and strat in block
            ]
            if vals:
                print(f"           {strat:8s} mean_regret={np.mean(vals):.4f}")

    # Crude slope: cov(ρ, gap) / var(ρ) on cell means
    if len(slope_points) >= 2:
        xs = np.array([p[0] for p in slope_points])
        ys = np.array([p[1] for p in slope_points])
        slope = float(np.cov(xs, ys, ddof=1)[0, 1] / np.var(xs, ddof=1))
        print(f"\n  mean gap vs ρ slope ≈ {slope:+.4f}  (prediction: > 0)")
        if slope > 0:
            print("  Verdict lean: supports C-mech-1 (check CI per ρ + powered n)")
        else:
            print("  Verdict lean: KILL risk for C-mech-1 — gap not increasing in ρ")


if __name__ == "__main__":
    main()
