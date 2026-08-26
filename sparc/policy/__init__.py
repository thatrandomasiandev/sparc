"""Policy training with confidence-gated bounding. See DESIGN.md § Pillar 3."""

from sparc.policy.bounding import (
    BoundingResult,
    UpdateAction,
    apply_bounding,
    apply_minibatch_bounding,
    effective_rewards,
)
from sparc.policy.config import BoundingConfig, BoundingMode
from sparc.policy.gate import ConfidenceGate
from sparc.policy.threshold import compute_disagreement_percentile, compute_threshold

__all__ = [
    "BoundingConfig",
    "BoundingMode",
    "BoundingResult",
    "UpdateAction",
    "ConfidenceGate",
    "apply_bounding",
    "apply_minibatch_bounding",
    "effective_rewards",
    "compute_disagreement_percentile",
    "compute_threshold",
]
