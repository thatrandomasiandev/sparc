"""Default synthetic annotator parameters from EXPERIMENTS.md."""

from __future__ import annotations

import numpy as np

# Per-operator bias vectors alpha_i (speed, clearance, energy)
OPERATOR_ALPHAS: dict[int, np.ndarray] = {
    1: np.array([1.0, 0.2, 0.1], dtype=np.float64),
    2: np.array([0.3, 1.0, 0.2], dtype=np.float64),
    3: np.array([0.4, 0.4, 1.0], dtype=np.float64),
}

DRIFT1_OPERATOR = 2
DRIFT1_ALPHA = np.array([0.8, 0.5, 0.1], dtype=np.float64)
DRIFT_FRACTION = 0.5

P_MISTAKE = 0.05
P_SKIP = 0.02
NUM_OPERATORS = 3
