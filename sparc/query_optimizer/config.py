"""Query optimizer defaults (EXPERIMENTS.md / DESIGN.md)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class QueryOptimizerConfig:
    batch_size: int = 8  # B
    lambda_div: float = 0.1
    eta_similarity: float = 0.5
    newton_prior_sigma: float = 1.0
    wall_clock_budget_sec: float = 2.0  # ROADMAP Week 11 gate (CI headroom)
    max_candidates: int = 500  # |C_w| for benchmark
    selection_mode: str = "eig"  # "eig" | "random" (Week 25 Pillar 2 ablation)
