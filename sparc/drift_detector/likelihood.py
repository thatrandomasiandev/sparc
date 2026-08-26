"""Bradley–Terry log-likelihood ratios for SPRT (DESIGN.md Pillar 4)."""

from __future__ import annotations

import math

from sparc.env.dataset import PreferenceRecord
from sparc.reward_model.ensemble import RewardEnsemble


def log_sigmoid(x: float) -> float:
    """Numerically stable log(sigmoid(x))."""
    if x >= 0:
        return -math.log1p(math.exp(-x))
    return x - math.log1p(math.exp(x))


def label_log_prob(y: int, delta: float) -> float:
    """log P(y | delta) under Bradley–Terry (y=0 prefer seg0, y=1 prefer seg1)."""
    sign = 1.0 if y == 0 else -1.0
    return log_sigmoid(sign * delta)


def mean_preference_delta(
    ensemble: RewardEnsemble,
    record: PreferenceRecord,
) -> float:
    """Mean ensemble utility difference g(seg0) - g(seg1) for the query operator."""
    query = record.query
    operator_id = query.operator_id
    phi0 = ensemble.encode(query.segment_0).unsqueeze(0)
    phi1 = ensemble.encode(query.segment_1).unsqueeze(0)
    z_i = ensemble.z[operator_id]
    deltas: list[float] = []
    for head in ensemble.heads:
        u0 = head.utility(phi0, operator_id, z_i)
        u1 = head.utility(phi1, operator_id, z_i)
        deltas.append(float((u0 - u1).item()))
    return sum(deltas) / len(deltas)


def log_likelihood_ratio(
    y: int,
    delta_h0: float,
    eta: float,
) -> float:
    """ell = log P(y|H1) - log P(y|H0) with H1: delta -> delta + eta."""
    delta_h1 = delta_h0 + eta
    return label_log_prob(y, delta_h1) - label_log_prob(y, delta_h0)
