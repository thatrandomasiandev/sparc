#!/usr/bin/env python3
"""Regenerate Phase 4 figures from suite artifacts (RESULTS.md Week 26)."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

# Optional matplotlib — skip plotting if unavailable
try:
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover
    plt = None


def plot_learning_curves(curves_dir: Path, output_dir: Path) -> list[Path]:
    if plt is None:
        print("matplotlib not installed; skipping plots", file=sys.stderr)
        return []

    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for csv_path in sorted(curves_dir.glob("*_learning_curve.csv")):
        method = csv_path.stem.replace("_learning_curve", "")
        by_seed: dict[int, list[tuple[int, float]]] = {}
        with csv_path.open(encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                seed = int(row["seed"])
                by_seed.setdefault(seed, []).append(
                    (int(row["env_steps"]), float(row["mean_return"]))
                )
        fig, ax = plt.subplots(figsize=(8, 5))
        for seed, points in sorted(by_seed.items()):
            points.sort(key=lambda p: p[0])
            xs, ys = zip(*points, strict=True)
            ax.plot(xs, ys, alpha=0.7, label=f"seed {seed}")
        ax.set_xlabel("Environment steps")
        ax.set_ylabel("Mean eval return (learned)")
        ax.set_title(f"{method} — return vs steps")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        out = output_dir / f"{method}_learning_curve.png"
        fig.tight_layout()
        fig.savefig(out, dpi=120)
        plt.close(fig)
        written.append(out)
    return written


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot suite learning curves")
    parser.add_argument(
        "suite_dir",
        type=Path,
        help="Suite output directory containing curves/",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Plot output directory (default: suite_dir/figures)",
    )
    args = parser.parse_args()
    curves = args.suite_dir / "curves"
    if not curves.is_dir():
        print(f"no curves/ in {args.suite_dir}", file=sys.stderr)
        return 1
    out = args.output or args.suite_dir / "figures"
    paths = plot_learning_curves(curves, out)
    for p in paths:
        print(p)
    return 0 if paths or plt is None else 1


if __name__ == "__main__":
    raise SystemExit(main())
