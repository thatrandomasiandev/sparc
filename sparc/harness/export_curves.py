"""Export learning curves from suite run artifacts to CSV."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from sparc.harness.report import learning_curve_points


def export_learning_curves(suite_summary_path: Path | str, output_dir: Path | str) -> list[Path]:
    """
    Write one CSV per method: seed, env_steps, mean_return, total_queries.

    Returns paths of written CSV files.
    """
    suite = json.loads(Path(suite_summary_path).read_text(encoding="utf-8"))
    out_root = Path(output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    for entry in suite.get("methods", []):
        method = str(entry.get("method", "unknown"))
        summary = entry.get("summary", {})
        rows: list[dict[str, int | float]] = []
        for run in summary.get("runs", []):
            seed = int(run["seed"])
            for step, ret, queries in learning_curve_points(run):
                rows.append(
                    {
                        "seed": seed,
                        "env_steps": step,
                        "mean_return": ret,
                        "total_queries": queries,
                    }
                )
        if not rows:
            continue
        path = out_root / f"{method}_learning_curve.csv"
        with path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=["seed", "env_steps", "mean_return", "total_queries"]
            )
            writer.writeheader()
            writer.writerows(rows)
        written.append(path)
    return written
