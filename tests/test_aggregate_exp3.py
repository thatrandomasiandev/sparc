"""Tests for early-budget metrics used by E3-power-120 pre-registration."""

from __future__ import annotations

import numpy as np

from aggregate_exp3 import early_auc_regret, early_mean_regret


def test_early_mean_and_auc_on_hand_built_curve() -> None:
    """Hand-built regret_curve: mean q1–3 and trapezoid AUC q1–5 are exact."""
    curve = [1.0, 3.0, 5.0, 7.0, 9.0, 11.0]
    assert early_mean_regret(curve, 3) == 3.0  # (1+3+5)/3
    # Trapezoid over [1,3,5,7,9] with dx=1: (1+9)/2 + 3+5+7 = 5 + 15 = 20
    assert early_auc_regret(curve, 5) == 20.0
    assert np.isclose(early_mean_regret(np.asarray(curve), 3), 3.0)
