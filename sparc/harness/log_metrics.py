"""Parse JSONL training logs for EXPERIMENTS.md metrics (Week 18)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class WindowTiming:
    window_id: int
    open_ts: str
    close_ts: str | None
    num_queries: int
    select_sec: float
    label_events: int


@dataclass(frozen=True)
class LogMetrics:
    """Aggregates from one SPARC JSONL log."""

    n_windows: int
    total_queries: int
    mean_wall_clock_per_query_sec: float
    mean_select_sec: float
    window_timings: tuple[WindowTiming, ...]


def _parse_records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        records.append(json.loads(line))
    return records


def extract_log_metrics(jsonl_path: Path | str) -> LogMetrics:
    """
    Compute wall-clock per query from ``window_open`` / ``label_received`` events.

    EXPERIMENTS.md § T_query: mean over windows of (t_labels_complete - t_open) / |Q_w|.
    Uses ISO timestamps in JSONL records.
    """
    from datetime import datetime

    records = _parse_records(Path(jsonl_path))
    windows: dict[int, dict[str, Any]] = {}

    for rec in records:
        event = rec.get("event")
        if event == "window_open":
            wid = int(rec["window_id"])
            windows[wid] = {
                "open_ts": rec["ts"],
                "select_sec": float(rec.get("select_sec", 0.0)),
                "num_queries": int(rec.get("num_queries", 0)),
                "labels": [],
            }
        elif event == "label_received":
            wid = int(rec["window_id"])
            if wid in windows:
                windows[wid]["labels"].append(rec["ts"])
        elif event == "window_close":
            wid = int(rec["window_id"])
            if wid in windows:
                windows[wid]["close_ts"] = rec["ts"]

    timings: list[WindowTiming] = []
    per_query_secs: list[float] = []

    for wid, data in sorted(windows.items()):
        open_dt = datetime.fromisoformat(data["open_ts"])
        label_times = data.get("labels", [])
        if label_times:
            last_dt = datetime.fromisoformat(label_times[-1])
            elapsed = (last_dt - open_dt).total_seconds()
            n_q = max(len(label_times), 1)
            per_query_secs.append(elapsed / n_q)
        timings.append(
            WindowTiming(
                window_id=wid,
                open_ts=data["open_ts"],
                close_ts=data.get("close_ts"),
                num_queries=int(data.get("num_queries", 0)),
                select_sec=float(data.get("select_sec", 0.0)),
                label_events=len(label_times),
            )
        )

    total_q = sum(t.num_queries for t in timings)
    mean_per_q = float(sum(per_query_secs) / len(per_query_secs)) if per_query_secs else 0.0
    mean_select = float(sum(t.select_sec for t in timings) / len(timings)) if timings else 0.0

    return LogMetrics(
        n_windows=len(timings),
        total_queries=total_q,
        mean_wall_clock_per_query_sec=mean_per_q,
        mean_select_sec=mean_select,
        window_timings=tuple(timings),
    )
