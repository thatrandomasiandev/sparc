#!/usr/bin/env python3
"""Plot E-eq (match rate vs d) and E-decouple (BALD−VOI vs ρ)."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load(path: Path) -> list[dict]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def plot_eq(rows: list[dict], out: Path) -> None:
    pair = [r for r in rows if r.get("kind", "pairwise") == "pairwise"]
    loc = [r for r in rows if r.get("kind") == "local_stability"]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    by_d = defaultdict(list)
    for r in pair:
        by_d[r["d"]].append(r["pairwise_match_rate"])
    ds = sorted(by_d)
    means = [np.mean(by_d[d]) for d in ds]
    ses = [np.std(by_d[d], ddof=1) / np.sqrt(len(by_d[d])) for d in ds]
    axes[0].errorbar(ds, means, yerr=ses, fmt="-o", color="#8a8a8a", capsize=4, lw=2)
    axes[0].set_xlabel("Reward dimension d")
    axes[0].set_ylabel("Pairwise π* match rate")
    axes[0].set_title("Exact pair matches (straw man)")
    axes[0].set_ylim(0, max(0.05, max(means) * 1.3 if means else 0.05))

    by_d = defaultdict(list)
    for r in loc:
        by_d[r["d"]].append(r["local_stable_rate"])
    if by_d:
        ds = sorted(by_d)
        means = [np.mean(by_d[d]) for d in ds]
        ses = [np.std(by_d[d], ddof=1) / np.sqrt(len(by_d[d])) for d in ds]
        axes[1].errorbar(ds, means, yerr=ses, fmt="-o", color="#f58518", capsize=4, lw=2)
        axes[1].set_ylim(0, 1.05)
    axes[1].set_xlabel("Reward dimension d")
    axes[1].set_ylabel(r"Local π*-stability rate ($\varepsilon$-ball)")
    axes[1].set_title("Local decision-irrelevant perturbations")

    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.suptitle("E-eq — C-mot-1/2", y=1.02)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    print(f"Wrote {out}")


def plot_decouple(rows: list[dict], out: Path) -> None:
    by = defaultdict(dict)
    for r in rows:
        by[(r["rho"], r["seed"], r["user"])][r["strategy"]] = r["final_regret"]
    rhos = sorted({k[0] for k in by})
    gaps, ses = [], []
    for rho in rhos:
        diffs = [
            block["bald"] - block["voi"]
            for (rr, _, _), block in by.items()
            if rr == rho and "bald" in block and "voi" in block
        ]
        diffs = np.asarray(diffs)
        gaps.append(diffs.mean())
        ses.append(diffs.std(ddof=1) / np.sqrt(len(diffs)))

    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.errorbar(rhos, gaps, yerr=ses, fmt="-o", color="#4c78a8", capsize=4, lw=2)
    ax.axhline(0.0, color="#888", lw=1, ls="--")
    ax.set_xlabel("Decoy fraction ρ")
    ax.set_ylabel("Paired BALD − VOI final regret")
    ax.set_title("E-decouple — VOI advantage vs structural irrelevance (C-mech-1)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    fig.savefig(out.with_suffix(".pdf"))
    print(f"Wrote {out}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--eq", type=Path, default=Path("experiments/runs/exp_eq.jsonl"))
    p.add_argument(
        "--decouple", type=Path, default=Path("experiments/runs/exp_decouple.jsonl")
    )
    p.add_argument("--outdir", type=Path, default=Path("paper/figures"))
    args = p.parse_args()
    if args.eq.exists():
        plot_eq(load(args.eq), args.outdir / "exp_eq_match_vs_d.png")
    if args.decouple.exists():
        plot_decouple(load(args.decouple), args.outdir / "exp_decouple_gap_vs_rho.png")


if __name__ == "__main__":
    main()
