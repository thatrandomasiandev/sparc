"""Targeted re-query batch selection (DESIGN.md Pillar 4 → Pillar 2)."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from sparc.query_optimizer.config import QueryOptimizerConfig
from sparc.query_optimizer.eig import QueryCandidate

if TYPE_CHECKING:
    from sparc.drift_detector.sprt import DriftDetector
    from sparc.reward_model.ensemble import RewardEnsemble


def select_batch_with_requery(
    candidates: list[QueryCandidate],
    batch_size: int,
    operator_id: int,
    ensemble: RewardEnsemble,
    drift_detector: DriftDetector,
    requery_operator_id: int,
    config: QueryOptimizerConfig | None = None,
) -> list[QueryCandidate]:
    """
    Force B_requery = ceil(B/2) from top-disagreement history; fill rest with EIG (DESIGN.md).
    """
    from sparc.query_optimizer.batch import select_batch

    cfg = config or QueryOptimizerConfig()
    b_requery = math.ceil(batch_size / 2)
    requery_pool = drift_detector.top_disagreement_candidates(requery_operator_id, b_requery)

    selected: list[QueryCandidate] = []
    used_ids: set[int] = set()

    for cand in requery_pool:
        cid = id(cand)
        if cid in used_ids:
            continue
        selected.append(cand)
        used_ids.add(cid)
        if len(selected) >= b_requery:
            break

    remaining = batch_size - len(selected)
    if remaining > 0:
        rest_pool = [c for c in candidates if id(c) not in used_ids]
        if len(rest_pool) >= remaining:
            rest = select_batch(rest_pool, remaining, operator_id, ensemble, cfg)
            selected.extend(rest)

    if len(selected) < batch_size:
        rest_pool = [c for c in candidates if id(c) not in {id(s) for s in selected}]
        if rest_pool:
            extra = select_batch(
                rest_pool,
                min(batch_size - len(selected), len(rest_pool)),
                operator_id,
                ensemble,
                cfg,
            )
            selected.extend(extra)

    return selected[:batch_size]
