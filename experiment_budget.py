#!/usr/bin/env python3
"""E-budget — regret vs query index curves from an exp3-family jsonl (C-mech-8).

Does not re-run elicitation; aggregates existing ``regret_curve`` fields.
Primary display: mean regret at each t with paired bootstrap CI for bald−voi.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def load_rows(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.open() if l.strip()]


def curves_by_strategy(rows: list[dict]) -> dict[str, np.ndarray]:
    """strategy -> (N, T) regret curves aligned on (seed, user)."""
    strategies = sorted({r["strategy"] for r in rows if "regret_curve" in r})
    keys = sorted({(r["seed"], r["user"]) for r in rows})
    by = {(s, u): {} for s, u in keys}
    for r in rows:
        if "regret_curve" not in r:
            continue
        by[(r["seed"], r["user"])][r["strategy"]] = r["regret_curve"]
    out = {s: [] for s in strategies}
    for k in keys:
        block = by[k]
        if not all(s in block for s in strategies):
            continue
        T = min(len(block[s]) for s in strategies)
        for s in strategies:
            out[s].append(block[s][:T])
    return {s: np.asarray(v, dtype=float) for s, v in out.items()}


def boot_ci(diff: np.ndarray, n_boot: int = 10_000, seed: int = 0):
    rng = np.random.default_rng(seed)
    n = len(diff)
    samples = [float(diff[rng.integers(0, n, n)].mean()) for _ in range(n_boot)]
    lo, hi = np.percentile(samples, [2.5, 97.5])
    return float(diff.mean()), float(lo), float(hi)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("path", type=Path)
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()
    curves = curves_by_strategy(load_rows(args.path))
    T = next(iter(curves.values())).shape[1]
    lines = [f"# E-budget from {args.path}", f"# N={next(iter(curves.values())).shape[0]} T={T}"]
    header = "t\t" + "\t".join(f"{s}_mean" for s in sorted(curves))
    if "bald" in curves and "voi" in curves:
        header += "\tbald-voi\tlo\thi"
    lines.append(header)
    for t in range(T):
        row = [str(t + 1)]
        for s in sorted(curves):
            row.append(f"{curves[s][:, t].mean():.4f}")
        if "bald" in curves and "voi" in curves:
            m, lo, hi = boot_ci(curves["bald"][:, t] - curves["voi"][:, t], seed=t)
            row += [f"{m:+.4f}", f"{lo:+.4f}", f"{hi:+.4f}"]
        lines.append("\t".join(row))
    text = "\n".join(lines) + "\n"
    print(text)
    if args.out:
        args.out.write_text(text)


if __name__ == "__main__":
    main()
