#!/usr/bin/env python3
"""Plot placeholder curves from run metrics (extend once real logs exist)."""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, default=Path("experiments/runs"))
    parser.add_argument("--out", type=Path, default=Path("paper/figures/curves.png"))
    args = parser.parse_args()

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Install matplotlib (pip install -e '.[plot]') to use this script.")
        return 1

    # Scaffold: empty figure so the pipeline is wired.
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.set_title("Placeholder — replace with real learning curves")
    ax.set_xlabel("step")
    ax.set_ylabel("metric")
    ax.grid(True, alpha=0.3)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
