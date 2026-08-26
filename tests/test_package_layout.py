"""Phase 2 Week 8 — verify subpackage skeleton imports."""

from __future__ import annotations

import importlib

SUBPACKAGES = (
    "sparc.env",
    "sparc.reward_model",
    "sparc.query_optimizer",
    "sparc.policy",
    "sparc.annotator_sim",
    "sparc.drift_detector",
    "sparc.cli",
)


def test_subpackages_importable() -> None:
    for name in SUBPACKAGES:
        importlib.import_module(name)


def test_subpackage_count() -> None:
    assert len(SUBPACKAGES) == 7
