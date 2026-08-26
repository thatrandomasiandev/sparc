"""SPRT drift detector configuration (DESIGN.md / EXPERIMENTS.md)."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class DriftDetectorConfig:
    alpha: float = 0.05  # false alarm
    beta: float = 0.10  # missed drift
    eta: float = 1.0  # H1 shift on preference logit delta
    history_windows: int = 2
    top_fraction: float = 0.2  # top 20% disagreement queries for re-query pool

    @property
    def boundary_upper(self) -> float:
        """A = log((1 - beta) / alpha)."""
        return math.log((1.0 - self.beta) / self.alpha)

    @property
    def boundary_lower(self) -> float:
        """B = log(beta / (1 - alpha))."""
        return math.log(self.beta / (1.0 - self.alpha))
