"""Confidence-gated policy bounding configuration (DESIGN.md Pillar 3)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class BoundingMode(str, Enum):
    PASS_THROUGH = "pass-through"
    SHRINK = "shrink"
    FREEZE = "freeze"


@dataclass(frozen=True)
class BoundingConfig:
    tau_0: float = 0.01
    kappa: float = 1.0
    f_max: float = 0.5
    mode: BoundingMode = BoundingMode.SHRINK
    percentile: float = 90.0
