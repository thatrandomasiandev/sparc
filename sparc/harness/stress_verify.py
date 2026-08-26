"""Verify stress-regime conditions from SPARC JSONL logs (PROBLEM.md gate 3)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class StressVerification:
    """Evidence that stress-regime configuration was active during a SPARC run."""

    regime: str | None
    n_windows: int
    operator_ids: set[int]
    sprt_drift_triggers: int
    scripted_drift_events: int
    labels_past_drift: bool
    drift_step: int | None
    multi_operator: bool
    comms_windows: bool

    @property
    def drift_detected(self) -> bool:
        return (
            self.scripted_drift_events > 0
            or self.sprt_drift_triggers > 0
            or self.labels_past_drift
        )

    @property
    def stress_active(self) -> bool:
        if self.regime != "stress":
            return False
        return self.comms_windows and self.multi_operator and self.drift_detected

    def to_dict(self) -> dict[str, Any]:
        return {
            "regime": self.regime,
            "n_windows": self.n_windows,
            "operator_ids": sorted(self.operator_ids),
            "sprt_drift_triggers": self.sprt_drift_triggers,
            "scripted_drift_events": self.scripted_drift_events,
            "labels_past_drift": self.labels_past_drift,
            "drift_step": self.drift_step,
            "multi_operator": self.multi_operator,
            "comms_windows": self.comms_windows,
            "drift_detected": self.drift_detected,
            "stress_active": self.stress_active,
        }


def verify_stress_from_jsonl(jsonl_path: Path | str, min_operators: int = 3) -> StressVerification:
    """
    Parse SPARC JSONL for comms windows, K operators, and Drift-1 evidence.

    Drift-1 may appear as ``scripted_drift`` log events, labels at/after the
    scripted drift step, or online ``drift_trigger`` (Pillar 4 SPRT).
    """
    path = Path(jsonl_path)
    windows: set[int] = set()
    operators: set[int] = set()
    sprt_triggers = 0
    scripted_drift = 0
    regime: str | None = None
    drift_step: int | None = None
    labels_past_drift = False

    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        event = rec.get("event")
        if event == "train_start":
            config = rec.get("config", {})
            regime = config.get("regime")
            train_steps = config.get("train_steps_total")
            if regime == "stress" and train_steps:
                drift_step = int(0.5 * int(train_steps))
        elif event == "window_open":
            windows.add(int(rec["window_id"]))
            operators.add(int(rec.get("operator_id", 0)))
        elif event == "label_received":
            operators.add(int(rec.get("operator_id", 0)))
            if drift_step is not None and int(rec.get("env_step", 0)) >= drift_step:
                labels_past_drift = True
        elif event == "scripted_drift":
            scripted_drift += 1
            labels_past_drift = True
        elif event == "drift_trigger":
            sprt_triggers += 1

    return StressVerification(
        regime=regime,
        n_windows=len(windows),
        operator_ids=operators - {0},
        sprt_drift_triggers=sprt_triggers,
        scripted_drift_events=scripted_drift,
        labels_past_drift=labels_past_drift,
        drift_step=drift_step,
        multi_operator=len(operators - {0}) >= min_operators,
        comms_windows=len(windows) >= 2,
    )
