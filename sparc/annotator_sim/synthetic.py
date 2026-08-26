"""Synthetic multi-operator preference annotator (EXPERIMENTS.md stress regime)."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from sparc.annotator_sim.defaults import (
    DRIFT1_ALPHA,
    DRIFT1_OPERATOR,
    DRIFT_FRACTION,
    NUM_OPERATORS,
    OPERATOR_ALPHAS,
    P_MISTAKE,
    P_SKIP,
)
from sparc.env.dataset import PreferenceQuery
from sparc.env.segments import TrajectorySegment


@dataclass
class DriftEvent:
    """Scripted preference drift for one operator at a fixed env step."""

    operator_id: int
    env_step: int
    new_alpha: np.ndarray


@dataclass
class SyntheticAnnotator:
    """
    Labels segment pairs via U_i(sigma) = alpha_i @ f_bar(sigma).

    Supports label noise (p_mistake), skips (p_skip), operator rotation by window,
    and scripted drift events.
    """

    operator_alphas: dict[int, np.ndarray] = field(default_factory=lambda: dict(OPERATOR_ALPHAS))
    p_mistake: float = P_MISTAKE
    p_skip: float = P_SKIP
    num_operators: int = NUM_OPERATORS
    drift_events: list[DriftEvent] = field(default_factory=list)
    feature_layout: str = "rover"
    seed: int | None = None

    def __post_init__(self) -> None:
        if not 0.0 <= self.p_mistake <= 1.0:
            raise ValueError("p_mistake must be in [0, 1]")
        if not 0.0 <= self.p_skip <= 1.0:
            raise ValueError("p_skip must be in [0, 1]")
        self._rng = np.random.default_rng(self.seed)
        self._applied_drift: set[tuple[int, int]] = set()
        self._alphas = {k: v.copy() for k, v in self.operator_alphas.items()}

    @classmethod
    def stress_regime(cls, train_steps: int, seed: int | None = None) -> SyntheticAnnotator:
        """Factory matching EXPERIMENTS.md stress regime with Drift-1 on Op-2."""
        drift_step = int(DRIFT_FRACTION * train_steps)
        return cls(
            drift_events=[
                DriftEvent(
                    operator_id=DRIFT1_OPERATOR,
                    env_step=drift_step,
                    new_alpha=DRIFT1_ALPHA.copy(),
                )
            ],
            seed=seed,
        )

    @classmethod
    def easy_regime(cls, seed: int | None = None) -> SyntheticAnnotator:
        """Single Op-2, no noise, no drift (EXPERIMENTS.md easy regime)."""
        return cls(
            operator_alphas={2: OPERATOR_ALPHAS[2].copy()},
            p_mistake=0.0,
            p_skip=0.0,
            num_operators=1,
            drift_events=[],
            seed=seed,
        )

    def operator_for_window(self, window_id: int) -> int:
        """Deterministic rotation: window w mod K -> Op-(w mod K + 1)."""
        if self.num_operators == 1:
            return int(next(iter(self._alphas)))
        return (window_id % self.num_operators) + 1

    def _apply_drift(self, env_step: int) -> list[DriftEvent]:
        """Apply scripted drift events whose threshold has been reached."""
        newly_applied: list[DriftEvent] = []
        for event in self.drift_events:
            key = (event.operator_id, event.env_step)
            if env_step >= event.env_step and key not in self._applied_drift:
                self._alphas[event.operator_id] = event.new_alpha.copy()
                self._applied_drift.add(key)
                newly_applied.append(event)
        return newly_applied

    def apply_drift_at(self, env_step: int) -> list[DriftEvent]:
        """Public entry: apply and return any newly fired scripted drift events."""
        return self._apply_drift(env_step)

    def alpha_for(self, operator_id: int, env_step: int) -> np.ndarray:
        self._apply_drift(env_step)
        if operator_id not in self._alphas:
            raise KeyError(f"unknown operator_id {operator_id}")
        return self._alphas[operator_id]

    def utility(
        self,
        segment: TrajectorySegment,
        operator_id: int,
        env_step: int,
    ) -> float:
        alpha = self.alpha_for(operator_id, env_step)
        features = segment.features(layout=self.feature_layout)
        return float(alpha @ features)

    def label_pair(
        self,
        segment_0: TrajectorySegment,
        segment_1: TrajectorySegment,
        operator_id: int,
        env_step: int,
    ) -> int | None:
        """
        Return preference label: 0 (prefer seg0), 1 (prefer seg1), or None (skip).

        Applies Bradley-Terry truth with p_mistake flip and p_skip.
        """
        if self._rng.random() < self.p_skip:
            return None

        u0 = self.utility(segment_0, operator_id, env_step)
        u1 = self.utility(segment_1, operator_id, env_step)
        label = 0 if u0 >= u1 else 1

        if self._rng.random() < self.p_mistake:
            label = 1 - label
        return label

    def label_query(self, query: PreferenceQuery) -> int | None:
        return self.label_pair(
            query.segment_0,
            query.segment_1,
            query.operator_id,
            query.env_step,
        )

    def label_batch(
        self,
        queries: list[PreferenceQuery],
    ) -> list[int | None]:
        return [self.label_query(q) for q in queries]
