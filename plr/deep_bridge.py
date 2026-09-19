"""L6 bridge notes — B-Pref / PEBBLE query-selection swap.

Status (2026-09-18): scaffolding only. Full B-Pref training loops are out of
scope for the tabular exact-regret paper spine; this module documents the
interface a deep PbRL codebase must expose so POP-VOI can replace disagreement
/ entropy sampling.

Required from a PEBBLE-style loop
---------------------------------
1. A pool of segment pairs ``phi: (Q, K, d)`` (or clip embeddings).
2. An ensemble / particle set over reward parameters ``(P, d)`` with weights.
3. A callable ``answer(phi) -> ranking`` (human or B-Pref teacher).
4. After each answer: update particles; *act* by optimizing policy under
   ``decide_w`` (mean by default — E-act).

What we can reuse today
-----------------------
- ``plr.algorithm.POPVOI`` already implements select → update → decide for
  exact tabular MDPs.
- For deep control, replace ``Belief.loss`` / ``voi_score`` with a *sampled*
  policy-value estimate (MC rollouts under each particle reward). Exact VI
  will not transfer.

Minimum credible L6 experiment
------------------------------
- Task: one DMControl locomotion env from B-Pref (e.g. walker-walk).
- Teachers: B-Pref perfect + one irrational teacher.
- Swap only query acquisition: random | disagreement | entropy | approx-VOI.
- Primary: return @ fixed feedback budget (B-Pref metric).
- Kill: approx-VOI ≱ disagreement on return (CI).

Blockers before coding the loop
-------------------------------
- Need torch+mujoco+B-Pref install in this environment.
- Need an approximate VOI estimator (rollout-based) — new research surface.
- Do not claim L6 until that estimator is implemented and pre-registered.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol

from torch import Tensor


class SegmentPool(Protocol):
    def candidates(self) -> Tensor:
        """Return ``(Q, K, d)`` segment-feature (or embedding) pairs."""


class RewardParticles(Protocol):
    def weights(self) -> Tensor: ...
    def parameters(self) -> Tensor: ...
    def update(self, phi: Tensor, order: Tensor) -> None: ...


@dataclass
class DeepQueryStep:
    """One acquisition step record for a PEBBLE-style session logger."""

    t: int
    query_index: int
    acquisition: str
    score: float
    return_so_far: float | None = None


def select_index(
    scores: list[float],
    *,
    rng_randint: Callable[[int], int] | None = None,
) -> int:
    """Argmax with optional random fallback when all scores are equal."""
    if not scores:
        raise ValueError("empty score list")
    if rng_randint is not None and max(scores) - min(scores) < 1e-12:
        return int(rng_randint(len(scores)))
    return int(max(range(len(scores)), key=lambda i: scores[i]))
