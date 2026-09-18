#!/usr/bin/env python3
"""Aggregate metrics.json files under experiments/runs into a summary table.

Usage:
  python experiments/scripts/summarize_runs.py experiments/runs
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def collect(root: Path) -> list[dict]:
    rows: list[dict] = []
    for metrics_path in sorted(root.rglob("metrics.json")):
        meta_path = metrics_path.parent / "meta.json"
        metrics = json.loads(metrics_path.read_text())
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        rows.append(
            {
                "run_dir": str(metrics_path.parent),
                "name": meta.get("name", metrics_path.parent.parent.name),
                "seed": meta.get("seed", metrics.get("seed")),
                "git_commit": meta.get("git_commit"),
                **{k: v for k, v in metrics.items() if k != "seed"},
            }
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs_root", type=Path, nargs="?", default=Path("experiments/runs"))
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of TSV")
    args = parser.parse_args()

    if not args.runs_root.exists():
        print(f"No runs directory at {args.runs_root}", file=sys.stderr)
        return 1

    rows = collect(args.runs_root)
    if not rows:
        print("No metrics.json files found.", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(rows, indent=2))
        return 0

    keys = sorted({k for row in rows for k in row})
    print("\t".join(keys))
    for row in rows:
        print("\t".join(str(row.get(k, "")) for k in keys))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
