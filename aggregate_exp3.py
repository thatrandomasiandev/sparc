#!/usr/bin/env python3
"""Aggregate experiment 3 runs with paired bootstrap CIs.

Strategies share users and candidates — comparisons are paired. Unpaired SEs are wrong.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def load_rows(path: Path) -> list[dict]:
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def paired_table(rows: list[dict], metric: str) -> dict[str, np.ndarray]:
    """Map strategy -> array aligned on (seed, user)."""
    strategies = sorted({r["strategy"] for r in rows})
    keys = sorted({(r["seed"], r["user"]) for r in rows})
    by = defaultdict(dict)
    for r in rows:
        by[(r["seed"], r["user"])][r["strategy"]] = r[metric]
    out = {s: [] for s in strategies}
    for k in keys:
        block = by[k]
        if not all(s in block for s in strategies):
            continue
        for s in strategies:
            out[s].append(block[s])
    return {s: np.asarray(v, dtype=float) for s, v in out.items()}


def bootstrap_ci(
    diff: np.ndarray,
    n_boot: int = 10_000,
    seed: int = 0,
) -> tuple[float, float, float, float]:
    """Return mean, lo, hi, tie_rate for paired differences."""
    rng = np.random.default_rng(seed)
    n = len(diff)
    means = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        means.append(diff[idx].mean())
    means = np.sort(np.asarray(means))
    lo = float(means[int(0.025 * n_boot)])
    hi = float(means[int(0.975 * n_boot)])
    tie_rate = float(np.mean(np.abs(diff) < 1e-12))
    return float(diff.mean()), lo, hi, tie_rate


def summarize(rows: list[dict], metric: str, n_boot: int) -> None:
    table = paired_table(rows, metric)
    strategies = list(table.keys())
    n = len(next(iter(table.values())))
    print(f"\n=== metric: {metric} (n={n}) ===")
    for s in strategies:
        x = table[s]
        print(f"  {s:10s}  mean={x.mean():.4f}  se={x.std(ddof=1) / np.sqrt(len(x)):.4f}")

    pairs = []
    if "voi" in table and "bald" in table:
        pairs.append(("bald", "voi"))
    if "voi_mean" in table and "voi" in table:
        pairs.append(("voi_mean", "voi"))
    if "voi" in table and "voi_look2" in table:
        pairs.append(("voi", "voi_look2"))
    if "random" in table and "voi" in table:
        pairs.append(("random", "voi"))
    if "voi" in table and "oracle" in table:
        pairs.append(("voi", "oracle"))
    if "voi_look2" in table and "oracle" in table:
        pairs.append(("voi_look2", "oracle"))

    print("  paired bootstrap diffs (A - B):")
    for a, b in pairs:
        diff = table[a] - table[b]
        mean, lo, hi, ties = bootstrap_ci(diff, n_boot=n_boot)
        print(
            f"    {a}-{b}: {mean:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]  "
            f"tie_rate={ties:.2f}"
        )

    if metric == "queries_to_threshold":
        for s in strategies:
            reached = [
                1.0 if r.get("reached_threshold") else 0.0
                for r in rows
                if r["strategy"] == s
            ]
            if reached:
                print(f"  {s:10s}  reach_rate={np.mean(reached):.2f}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("results", type=Path, help="exp3 JSONL from experiment_3.py")
    p.add_argument("--n-boot", type=int, default=10_000)
    p.add_argument(
        "--primary",
        choices=["final_regret", "queries_to_threshold"],
        default="final_regret",
        help="Primary metric (pre-specify before runs)",
    )
    args = p.parse_args()
    rows = load_rows(args.results)
    if not rows:
        raise SystemExit(f"No rows in {args.results}")

    summarize(rows, args.primary, args.n_boot)
    secondary = (
        "queries_to_threshold" if args.primary == "final_regret" else "final_regret"
    )
    summarize(rows, secondary, args.n_boot)


if __name__ == "__main__":
    main()
