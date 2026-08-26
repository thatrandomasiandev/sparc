"""Harness run configuration."""

from __future__ import annotations

from dataclasses import dataclass, field

from sparc.baselines.config import BaselineConfig


@dataclass
class HarnessConfig:
    """One harness run: method + baseline config + optional multi-seed."""

    method: str = "pebble"
    baseline: BaselineConfig = field(default_factory=BaselineConfig)
    seeds: list[int] = field(default_factory=lambda: [0])

    @classmethod
    def from_baseline_json(cls, path: str, method: str = "pebble") -> HarnessConfig:
        baseline = BaselineConfig.from_json_file(path)
        return cls(method=method, baseline=baseline, seeds=[baseline.seed])
