"""Pillar 4 — online preference-drift detector (SPRT). See DESIGN.md § Pillar 4."""

from sparc.drift_detector.config import DriftDetectorConfig
from sparc.drift_detector.likelihood import log_likelihood_ratio, mean_preference_delta
from sparc.drift_detector.sprt import (
    DriftDetector,
    DriftTriggerEvent,
    HistoricalQuery,
    OperatorDriftState,
)

__all__ = [
    "DriftDetectorConfig",
    "DriftDetector",
    "DriftTriggerEvent",
    "OperatorDriftState",
    "HistoricalQuery",
    "log_likelihood_ratio",
    "mean_preference_delta",
]
