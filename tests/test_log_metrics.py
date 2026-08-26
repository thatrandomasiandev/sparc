"""Tests for JSONL log metric extraction."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from sparc.harness.log_metrics import extract_log_metrics


def test_wall_clock_per_query_from_jsonl() -> None:
    records = [
        {
            "ts": "2026-08-25T12:00:00+00:00",
            "event": "window_open",
            "window_id": 0,
            "num_queries": 2,
            "select_sec": 0.05,
        },
        {
            "ts": "2026-08-25T12:00:01+00:00",
            "event": "label_received",
            "window_id": 0,
        },
        {
            "ts": "2026-08-25T12:00:02+00:00",
            "event": "label_received",
            "window_id": 0,
        },
        {
            "ts": "2026-08-25T12:00:03+00:00",
            "event": "window_close",
            "window_id": 0,
        },
    ]
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "run.jsonl"
        path.write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")
        metrics = extract_log_metrics(path)
        assert metrics.n_windows == 1
        assert metrics.total_queries == 2
        assert metrics.mean_wall_clock_per_query_sec == 1.0  # 2s / 2 labels
        assert metrics.mean_select_sec == 0.05
