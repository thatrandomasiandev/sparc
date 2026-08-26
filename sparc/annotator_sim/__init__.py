"""Synthetic multi-operator annotator. See EXPERIMENTS.md."""

from sparc.annotator_sim.defaults import (
    DRIFT1_ALPHA,
    DRIFT1_OPERATOR,
    DRIFT_FRACTION,
    NUM_OPERATORS,
    OPERATOR_ALPHAS,
    P_MISTAKE,
    P_SKIP,
)
from sparc.annotator_sim.synthetic import DriftEvent, SyntheticAnnotator

__all__ = [
    "DriftEvent",
    "SyntheticAnnotator",
    "OPERATOR_ALPHAS",
    "P_MISTAKE",
    "P_SKIP",
    "NUM_OPERATORS",
    "DRIFT1_OPERATOR",
    "DRIFT1_ALPHA",
    "DRIFT_FRACTION",
]
