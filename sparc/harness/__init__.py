"""Shared benchmark harness (Phase 3, Week 16)."""

from sparc.harness.config import HarnessConfig
from sparc.harness.metrics import BaselineRunResult, EvalMetrics

__all__ = ["HarnessConfig", "EvalMetrics", "BaselineRunResult", "run_baseline"]


def run_baseline(*args, **kwargs):
    """Lazy import to avoid circular dependency with baselines."""
    from sparc.harness.runner import run_baseline as _run

    return _run(*args, **kwargs)
