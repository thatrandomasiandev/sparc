#!/usr/bin/env python3
"""Generic paired bootstrap aggregator for mechanism experiments.

Usage examples::

  python experiments/scripts/aggregate_paired.py experiments/runs/exp_prior.jsonl \\
      --group prior --metric final_regret --pairs uniform,population cheat,population

  python experiments/scripts/aggregate_paired.py experiments/runs/exp_act.jsonl \\
      --group decision_rule --filter acquisition=voi --metric final_regret \\
      --pairs mean,softminimax mean,map mean,sample

  python experiments/scripts/aggregate_paired.py experiments/runs/exp_mismatch.jsonl \\
      --group strategy --split mismatch --metric final_regret --pairs bald,voi random,voi
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


def apply_filters(rows: list[dict], filters: list[str]) -> list[dict]:
    out = rows
    for spec in filters:
        key, _, val = spec.partition("=")
        out = [r for r in out if str(r.get(key)) == val]
    return out


def paired_arrays(
    rows: list[dict],
    group_key: str,
    metric: str,
    id_keys: tuple[str, ...] = ("seed", "user"),
) -> dict[str, np.ndarray]:
    groups = sorted({str(r[group_key]) for r in rows})
    keys = sorted({tuple(r[k] for k in id_keys) for r in rows})
    by: dict[tuple, dict] = defaultdict(dict)
    for r in rows:
        by[tuple(r[k] for k in id_keys)][str(r[group_key])] = r[metric]
    out = {g: [] for g in groups}
    for k in keys:
        block = by[k]
        if not all(g in block for g in groups):
            continue
        for g in groups:
            out[g].append(block[g])
    return {g: np.asarray(v, dtype=float) for g, v in out.items()}


def bootstrap_ci(
    diff: np.ndarray, n_boot: int = 10_000, seed: int = 0
) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    n = len(diff)
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boots.append(float(diff[idx].mean()))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return float(diff.mean()), float(lo), float(hi)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("path", type=Path)
    p.add_argument("--group", required=True, help="Field that labels conditions")
    p.add_argument("--metric", default="final_regret")
    p.add_argument("--filter", action="append", default=[], dest="filters")
    p.add_argument(
        "--split",
        default=None,
        help="If set, report separately for each value of this field",
    )
    p.add_argument(
        "--pairs",
        nargs="+",
        default=[],
        help="Comparisons a,b meaning mean(a-b); positive ⇒ a worse than b on regret",
    )
    p.add_argument(
        "--id-keys",
        nargs="+",
        default=["seed", "user"],
        help="Fields that identify a paired unit (default: seed user)",
    )
    p.add_argument("--n-boot", type=int, default=10_000)
    args = p.parse_args()

    rows = apply_filters(load_rows(args.path), args.filters)
    splits = [None]
    if args.split:
        splits = sorted({str(r[args.split]) for r in rows})

    id_keys = tuple(args.id_keys)
    for split_val in splits:
        subset = rows
        label = "all"
        if split_val is not None:
            subset = [r for r in rows if str(r[args.split]) == split_val]
            label = f"{args.split}={split_val}"
        arr = paired_arrays(subset, args.group, args.metric, id_keys=id_keys)
        print(f"\n=== {label}  n≈{len(next(iter(arr.values()), []))} ===")
        for g, v in arr.items():
            print(f"  {g:16s}  mean={v.mean():.4f}  sd={v.std(ddof=1):.4f}")
        for pair in args.pairs:
            a, _, b = pair.partition(",")
            if a not in arr or b not in arr:
                print(f"  skip {pair}: missing group")
                continue
            mean, lo, hi = bootstrap_ci(arr[a] - arr[b], n_boot=args.n_boot)
            print(f"  {a}−{b}: {mean:+.4f}  [{lo:+.4f}, {hi:+.4f}]")


if __name__ == "__main__":
    main()
