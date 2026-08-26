"""PEBBLE baseline trainer (B-Pref / Lee et al. 2021) using SB3 SAC."""

from __future__ import annotations

from dataclasses import dataclass

from sparc.baselines.config import BaselineConfig
from sparc.baselines.loop import run_preference_baseline
from sparc.harness.metrics import BaselineRunResult


@dataclass
class PebbleTrainer:
    """Runs PEBBLE: unsupervised pretrain + disagreement queries + SAC."""

    config: BaselineConfig

    def run(self) -> BaselineRunResult:
        return run_preference_baseline(self.config, method="pebble")
