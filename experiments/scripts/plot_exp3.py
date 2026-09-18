#!/usr/bin/env python3
"""Plot experiment-3 final regret (and optional regret curves) from a JSONL run."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ORDER = ["random", "voi_mean", "bald", "voi", "oracle"]
LABELS = {
    "random": "Random",
    "voi_mean": "VOI (mean)",
    "bald": "BALD",
    "voi": "VOI (particle)",
    "oracle": "Oracle",
}
COLORS = {
    "random": "#8a8a8a",
    "voi_mean": "#c4a35a",
    "bald": "#4c78a8",
    "voi": "#f58518",
    "oracle": "#54a24b",
}


def load_rows(path: Path) -> list[dict]:
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def paired_metric(rows: list[dict], metric: str) -> dict[str, np.ndarray]:
    strategies = sorted({r["strategy"] for r in rows})
    keys = sorted({(r["seed"], r["user"]) for r in rows})
    by = defaultdict(dict)
    for r in rows:
        by[(r["seed"], r["user"])][r["strategy"]] = r[metric]
    out: dict[str, list[float]] = {s: [] for s in strategies}
    for k in keys:
        block = by[k]
        if not all(s in block for s in strategies):
            continue
        for s in strategies:
            out[s].append(float(block[s]))
    return {s: np.asarray(v) for s, v in out.items()}


def plot_final_regret(rows: list[dict], out: Path) -> None:
    table = paired_metric(rows, "final_regret")
    strategies = [s for s in ORDER if s in table]
    means = [table[s].mean() for s in strategies]
    ses = [table[s].std(ddof=1) / np.sqrt(len(table[s])) for s in strategies]
    n = len(next(iter(table.values())))

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    x = np.arange(len(strategies))
    bars = ax.bar(
        x,
        means,
        yerr=ses,
        color=[COLORS[s] for s in strategies],
        edgecolor="black",
        linewidth=0.6,
        capsize=4,
        error_kw={"elinewidth": 1.2, "capthick": 1.2},
    )
    ax.set_xticks(x)
    ax.set_xticklabels([LABELS[s] for s in strategies])
    ax.set_ylabel("Final-query regret (↓ better)")
    ax.set_title(f"VOI ablation — final regret (n={n} paired user-runs, mean ± SE)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_ylim(0, max(means[i] + ses[i] for i in range(len(means))) * 1.18)

    for bar, m, se in zip(bars, means, ses):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            m + se + 0.08,
            f"{m:.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160)
    fig.savefig(out.with_suffix(".pdf"))
    print(f"Wrote {out} and {out.with_suffix('.pdf')}")


def plot_regret_curves(rows: list[dict], out: Path) -> None:
    """Mean regret vs query index, with SE band."""
    by_strat: dict[str, list[list[float]]] = defaultdict(list)
    for r in rows:
        curve = r.get("regret_curve")
        if curve:
            by_strat[r["strategy"]].append(curve)

    strategies = [s for s in ORDER if s in by_strat]
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for s in strategies:
        arr = np.asarray(by_strat[s], dtype=float)  # (N, T)
        mean = arr.mean(axis=0)
        se = arr.std(axis=0, ddof=1) / np.sqrt(arr.shape[0])
        t = np.arange(1, mean.shape[0] + 1)
        ax.plot(t, mean, label=LABELS[s], color=COLORS[s], lw=2)
        ax.fill_between(t, mean - se, mean + se, color=COLORS[s], alpha=0.18)

    ax.set_xlabel("Query index")
    ax.set_ylabel("Regret after query (↓ better)")
    ax.set_title("Regret curves — mean ± SE")
    ax.legend(frameon=False, loc="upper right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160)
    fig.savefig(out.with_suffix(".pdf"))
    print(f"Wrote {out} and {out.with_suffix('.pdf')}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "results",
        type=Path,
        nargs="?",
        default=Path("experiments/runs/exp3_voi_ablation2.jsonl"),
    )
    p.add_argument("--outdir", type=Path, default=Path("paper/figures"))
    args = p.parse_args()
    rows = load_rows(args.results)
    if not rows:
        raise SystemExit(f"No rows in {args.results}")
    stem = args.results.stem
    plot_final_regret(rows, args.outdir / f"{stem}_final_regret.png")
    plot_regret_curves(rows, args.outdir / f"{stem}_regret_curves.png")


if __name__ == "__main__":
    main()
