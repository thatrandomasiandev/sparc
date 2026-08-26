"""Confidence gate integrating RewardEnsemble disagreement (Pillar 3)."""

from __future__ import annotations

from dataclasses import dataclass, field

from sparc.env.segments import TrajectorySegment
from sparc.policy.bounding import BoundingResult, apply_bounding, apply_minibatch_bounding
from sparc.policy.config import BoundingConfig
from sparc.policy.threshold import compute_disagreement_percentile, compute_threshold
from sparc.reward_model.ensemble import RewardEnsemble


@dataclass
class ConfidenceGate:
    """
    Maintains rolling tau_t from replay-buffer disagreement; bounds policy rewards.

    Call `update_threshold` once per comms window when labels merge (DESIGN.md cadence).
    """

    ensemble: RewardEnsemble
    config: BoundingConfig
    operator_id: int = 1
    q90: float = field(default=0.0)
    tau: float = field(default=0.0)

    def __post_init__(self) -> None:
        self.tau = compute_threshold(self.q90, self.config)

    def update_threshold(self, buffer_segments: list[TrajectorySegment]) -> float:
        """Recompute q90 and tau_t from segments in replay buffer."""
        disagreements = [
            self.ensemble.uncertainty(seg, self.operator_id) for seg in buffer_segments
        ]
        self.q90 = compute_disagreement_percentile(disagreements, self.config.percentile)
        self.tau = compute_threshold(self.q90, self.config)
        return self.tau

    def disagreement(self, segment: TrajectorySegment) -> float:
        return self.ensemble.uncertainty(segment, self.operator_id)

    def bound_reward(self, segment: TrajectorySegment, r_hat: float) -> BoundingResult:
        d = self.disagreement(segment)
        return apply_bounding(r_hat, d, self.tau, self.config.mode)

    def bound_minibatch(
        self,
        segments: list[TrajectorySegment],
        rewards: list[float],
    ) -> tuple[list[BoundingResult], bool]:
        disagreements = [self.disagreement(seg) for seg in segments]
        return apply_minibatch_bounding(rewards, disagreements, self.tau, self.config)
