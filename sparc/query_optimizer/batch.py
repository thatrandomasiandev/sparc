"""Greedy batch selection under per-window budget B."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from sparc.env.dataset import PreferenceQuery
from sparc.query_optimizer.config import QueryOptimizerConfig
from sparc.query_optimizer.diversity import diversity_bonus, similarity
from sparc.query_optimizer.eig import QueryCandidate, score_eig
from sparc.reward_model.ensemble import RewardEnsemble

if TYPE_CHECKING:
    from sparc.drift_detector.sprt import DriftDetector


def operator_for_window(window_id: int, num_operators: int = 3) -> int:
    """Deterministic rotation (EXPERIMENTS.md): window w mod K -> Op-(w mod K + 1)."""
    if num_operators == 1:
        return 1
    return (window_id % num_operators) + 1


def select_batch_random(
    candidates: list[QueryCandidate],
    batch_size: int,
    rng: np.random.Generator | None = None,
) -> list[QueryCandidate]:
    """Uniform random batch (Week 25: SPARC − Pillar 2)."""
    if batch_size > len(candidates):
        raise ValueError(f"batch_size {batch_size} exceeds |candidates|={len(candidates)}")
    gen = rng or np.random.default_rng()
    indices = gen.choice(len(candidates), size=batch_size, replace=False)
    return [candidates[int(i)] for i in indices]


def select_batch(
    candidates: list[QueryCandidate],
    batch_size: int,
    operator_id: int,
    ensemble: RewardEnsemble,
    config: QueryOptimizerConfig | None = None,
    rng: np.random.Generator | None = None,
) -> list[QueryCandidate]:
    """
    Greedy EIG batch selection with submodular similarity penalty (DESIGN.md SelectBatch).

    When ``config.selection_mode == "random"``, samples uniformly (Pillar 2 ablation).
    """
    cfg = config or QueryOptimizerConfig()
    if cfg.selection_mode == "random":
        return select_batch_random(candidates, batch_size, rng=rng)
    if batch_size > len(candidates):
        raise ValueError(f"batch_size {batch_size} exceeds |candidates|={len(candidates)}")

    for c in candidates:
        c.ensure_embeddings(ensemble)

    scores: dict[int, float] = {}
    for idx, q in enumerate(candidates):
        eig = score_eig(q, operator_id, ensemble, cfg)
        scores[idx] = eig + cfg.lambda_div * diversity_bonus(q, [])

    selected: list[QueryCandidate] = []
    selected_indices: list[int] = []

    while len(selected) < batch_size:
        best_idx = max(
            (i for i in range(len(candidates)) if i not in selected_indices),
            key=lambda i: scores[i],
        )
        selected.append(candidates[best_idx])
        selected_indices.append(best_idx)

        for idx, q in enumerate(candidates):
            if idx in selected_indices:
                continue
            max_sim = max(similarity(q, candidates[j]) for j in selected_indices)
            scores[idx] -= cfg.eta_similarity * max_sim

    return selected


def candidates_to_queries(
    selected: list[QueryCandidate],
    operator_id: int,
    window_id: int,
    env_step: int,
) -> list[PreferenceQuery]:
    return [
        PreferenceQuery(
            segment_0=c.segment_0,
            segment_1=c.segment_1,
            operator_id=operator_id,
            window_id=window_id,
            env_step=env_step,
        )
        for c in selected
    ]


@dataclass
class BatchSelectionResult:
    queries: list[PreferenceQuery]
    wall_clock_sec: float
    operator_id: int
    window_id: int


@dataclass
class CommsWindowOptimizer:
    """
    Hard window coupling: one batch burst per open, no mid-window re-issue (DESIGN.md).
    """

    ensemble: RewardEnsemble
    config: QueryOptimizerConfig
    num_operators: int = 3
    drift_detector: DriftDetector | None = None
    _outstanding: bool = False

    def on_window_open(
        self,
        window_id: int,
        candidates: list[QueryCandidate],
        env_step: int,
        operator_id: int | None = None,
        batch_size: int | None = None,
    ) -> BatchSelectionResult:
        if self._outstanding:
            raise RuntimeError("cannot issue queries — previous window batch still outstanding")

        op_id = operator_id if operator_id is not None else operator_for_window(
            window_id, self.num_operators
        )
        t0 = time.perf_counter()
        effective_batch = batch_size if batch_size is not None else self.config.batch_size

        requery_op: int | None = None
        if self.drift_detector is not None:
            requery_op = self.drift_detector.requery_operator()

        if requery_op is not None:
            from sparc.query_optimizer.requery import select_batch_with_requery

            assert self.drift_detector is not None
            selected = select_batch_with_requery(
                candidates,
                effective_batch,
                op_id,
                self.ensemble,
                self.drift_detector,
                requery_op,
                self.config,
            )
            self.drift_detector.clear_requery(requery_op)
        else:
            selected = select_batch(
                candidates,
                effective_batch,
                op_id,
                self.ensemble,
                self.config,
            )

        elapsed = time.perf_counter() - t0
        queries = candidates_to_queries(selected, op_id, window_id, env_step)

        if self.drift_detector is not None:
            self.drift_detector.record_window_queries(selected, window_id, op_id)

        self._outstanding = True
        return BatchSelectionResult(
            queries=queries,
            wall_clock_sec=elapsed,
            operator_id=op_id,
            window_id=window_id,
        )

    def on_labels_received(self) -> None:
        """Clear outstanding flag when window labels merged (next window may open)."""
        self._outstanding = False


def benchmark_batch_selection(
    ensemble: RewardEnsemble,
    candidates: list[QueryCandidate],
    config: QueryOptimizerConfig | None = None,
    operator_id: int = 1,
) -> float:
    """Return wall-clock seconds for SelectBatch (ROADMAP Week 11 benchmark)."""
    cfg = config or QueryOptimizerConfig()
    t0 = time.perf_counter()
    select_batch(candidates, cfg.batch_size, operator_id, ensemble, cfg)
    return time.perf_counter() - t0
