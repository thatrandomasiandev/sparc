"""Tests for stress-regime JSONL verification."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from sparc.harness.stress_verify import verify_stress_from_jsonl


def test_verify_stress_from_jsonl_sprt_and_scripted() -> None:
    records = [
        {
            "event": "train_start",
            "config": {"regime": "stress", "train_steps_total": 6000},
        },
        {"event": "window_open", "window_id": 0, "operator_id": 1},
        {"event": "label_received", "operator_id": 1, "env_step": 2000},
        {"event": "window_open", "window_id": 1, "operator_id": 2},
        {"event": "scripted_drift", "operator_id": 2, "env_step": 3000},
        {"event": "label_received", "operator_id": 2, "env_step": 4000},
        {"event": "window_open", "window_id": 2, "operator_id": 3},
        {"event": "label_received", "operator_id": 3, "env_step": 4500},
        {"event": "drift_trigger", "operator_id": 2},
    ]
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "run.jsonl"
        path.write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")
        v = verify_stress_from_jsonl(path)
        assert v.regime == "stress"
        assert v.comms_windows is True
        assert v.multi_operator is True
        assert v.scripted_drift_events == 1
        assert v.sprt_drift_triggers == 1
        assert v.drift_detected is True
        assert v.stress_active is True
