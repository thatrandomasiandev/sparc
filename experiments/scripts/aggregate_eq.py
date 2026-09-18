#!/usr/bin/env python3
"""Aggregate E-eq / E-eq-d: pairwise match + local π* stability vs d."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def load(path: Path) -> list[dict]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("results", type=Path)
    args = p.parse_args()
    rows = load(args.results)

    pair = [r for r in rows if r.get("kind", "pairwise") == "pairwise"]
    loc = [r for r in rows if r.get("kind") == "local_stability"]

    print("=== E-eq primary: local_stable_rate vs d (C-mot-1) ===")
    by_d: dict[int, list[float]] = defaultdict(list)
    for r in loc:
        by_d[r["d"]].append(r["local_stable_rate"])
    for d in sorted(by_d):
        x = np.asarray(by_d[d], dtype=float)
        se = x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else 0.0
        print(f"  d={d:2d}  n={len(x)}  local_stable={x.mean():.3f} ± {se:.3f}")

    print("\n=== E-eq-d secondary: pairwise_match_rate vs d (C-mot-2) ===")
    by_d = defaultdict(list)
    for r in pair:
        by_d[r["d"]].append(r["pairwise_match_rate"])
    for d in sorted(by_d):
        x = np.asarray(by_d[d], dtype=float)
        se = x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else 0.0
        print(
            f"  d={d:2d}  n={len(x)}  match={x.mean():.4f} ± {se:.4f}  "
            f"compression≈{np.mean([r['compression'] for r in pair if r['d']==d]):.3f}"
        )

    if by_d:
        ds = np.asarray(sorted(by_d), dtype=float)
        ys = np.asarray([np.mean(by_d[int(d)]) for d in ds])
        if len(ds) >= 2 and float(np.var(ds, ddof=1)) > 0:
            slope = float(np.cov(ds, ys, ddof=1)[0, 1] / np.var(ds, ddof=1))
            print(f"\n  pairwise match vs d slope ≈ {slope:+.6f}")


if __name__ == "__main__":
    main()
