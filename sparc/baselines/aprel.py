"""APReL batch active learner baseline (disagreement + medoid batch selection)."""

from __future__ import annotations

from dataclasses import dataclass, replace

from sparc.baselines.config import BaselineConfig
from sparc.baselines.loop import run_preference_baseline
from sparc.harness.metrics import BaselineRunResult


@dataclass
class AprelTrainer:
    """Runs APReL-style batch active learning via feed_type=2 (medoid + disagreement)."""

    config: BaselineConfig

    def run(self) -> BaselineRunResult:
        cfg = replace(self.config, feed_type=2)
        return run_preference_baseline(cfg, method="aprel")
