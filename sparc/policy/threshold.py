"""Rolling disagreement percentile threshold (DESIGN.md § Pillar 3)."""

from __future__ import annotations

import numpy as np

from sparc.policy.config import BoundingConfig


def compute_disagreement_percentile(
    disagreements: list[float] | np.ndarray,
    percentile: float = 90.0,
) -> float:
    """q_{90,t} = P90({D(sigma) : sigma in buffer})."""
    if len(disagreements) == 0:
        return 0.0
    arr = np.asarray(disagreements, dtype=np.float64)
    return float(np.percentile(arr, percentile))


def compute_threshold(q90: float, config: BoundingConfig) -> float:
    """tau_t = tau_0 + kappa * q90."""
    return config.tau_0 + config.kappa * q90
