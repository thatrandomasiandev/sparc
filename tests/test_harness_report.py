"""Tests for harness suite report generation."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from sparc.harness.report import suite_table_markdown, write_suite_report


def _fake_suite_summary() -> dict:
    return {
        "suite_name": "test_suite",
        "env_id": "Pendulum-v1",
        "seeds": [0, 1],
        "methods": [
            {
                "method": "pebble",
                "summary": {
                    "method": "pebble",
                    "summary": {
                        "n_seeds": 2,
                        "mean_asymptotic_return": 100.0,
                        "ci95_low": 90.0,
                        "ci95_high": 110.0,
                        "mean_asymptotic_true_return": 95.0,
                        "mean_total_queries": 50.0,
                        "mean_wall_clock_sec": 30.0,
                    },
                },
            },
            {
                "method": "sparc",
                "summary": {
                    "method": "sparc",
                    "summary": {
                        "n_seeds": 2,
                        "mean_asymptotic_return": 98.0,
                        "ci95_low": 88.0,
                        "ci95_high": 108.0,
                        "mean_asymptotic_true_return": 94.0,
                        "mean_total_queries": 10.0,
                        "mean_wall_clock_sec": 25.0,
                    },
                },
            },
        ],
        "comparisons": {
            "sparc_vs_pebble": {
                "mean_r_inf_a": 100.0,
                "mean_r_inf_b": 98.0,
                "mean_queries_a": 50.0,
                "mean_queries_b": 10.0,
                "query_reduction_pct": 80.0,
                "welch_p": 0.42,
            }
        },
    }


def test_suite_table_markdown() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "suite_summary.json"
        path.write_text(json.dumps(_fake_suite_summary()), encoding="utf-8")
        md = suite_table_markdown(path)
        assert "pebble" in md
        assert "sparc" in md
        assert "sparc_vs_pebble" in md or "Primary comparison" in md


def test_write_suite_report() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "suite_summary.json"
        path.write_text(json.dumps(_fake_suite_summary()), encoding="utf-8")
        out = write_suite_report(path)
        assert out.exists()
        assert out.name == "SUITE_REPORT.md"
