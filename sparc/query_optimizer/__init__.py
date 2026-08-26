"""Pillar 2 — latency-aware batch query optimizer. See DESIGN.md § Pillar 2."""

from sparc.query_optimizer.batch import (
    BatchSelectionResult,
    CommsWindowOptimizer,
    benchmark_batch_selection,
    candidates_to_queries,
    operator_for_window,
    select_batch,
)
from sparc.query_optimizer.config import QueryOptimizerConfig
from sparc.query_optimizer.diversity import diversity_bonus, similarity
from sparc.query_optimizer.eig import QueryCandidate, bernoulli_entropy, mean_pref_prob, score_eig

__all__ = [
    "QueryOptimizerConfig",
    "QueryCandidate",
    "score_eig",
    "mean_pref_prob",
    "bernoulli_entropy",
    "select_batch",
    "candidates_to_queries",
    "operator_for_window",
    "CommsWindowOptimizer",
    "BatchSelectionResult",
    "benchmark_batch_selection",
    "diversity_bonus",
    "similarity",
]
