"""PrefPPO baseline trainer — PEBBLE reward model + SB3 PPO."""

from __future__ import annotations

from dataclasses import dataclass

from sparc.baselines.config import BaselineConfig
from sparc.baselines.loop import run_preference_baseline
from sparc.harness.metrics import BaselineRunResult


@dataclass
class PrefPpoTrainer:
    """Same query schedule as PEBBLE but policy is PPO (B-Pref PrefPPO)."""

    config: BaselineConfig

    def run(self) -> BaselineRunResult:
        cfg = self.config
        if cfg.method != "prefppo":
            object.__setattr__(cfg, "method", "prefppo")
        return run_preference_baseline(cfg, method="prefppo")
