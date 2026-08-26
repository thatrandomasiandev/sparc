"""Multi-seed benchmark orchestration."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

from sparc.baselines.config import BaselineConfig
from sparc.engine.config import TrainConfig
from sparc.harness.config import HarnessConfig
from sparc.harness.drift_regret import regret_from_baseline_result
from sparc.harness.metrics import BaselineRunResult
from sparc.harness.runner import run_baseline
from sparc.harness.stats import SeedSummary, summarize_seed_runs

HarnessConfigType = HarnessConfig | BaselineConfig | TrainConfig


@dataclass
class MultiSeedResult:
    """Outcome of a full multi-seed benchmark suite."""

    method: str
    env_id: str
    seeds: list[int]
    runs: list[BaselineRunResult] = field(default_factory=list)
    summary: SeedSummary | None = None
    total_wall_clock_sec: float = 0.0

    def to_dict(self) -> dict:
        return {
            "method": self.method,
            "env_id": self.env_id,
            "seeds": self.seeds,
            "total_wall_clock_sec": self.total_wall_clock_sec,
            "summary": asdict(self.summary) if self.summary else None,
            "runs": [r.to_dict() for r in self.runs],
        }


def _resolve_env_id(
    config: HarnessConfig | BaselineConfig | TrainConfig,
    method: str,
) -> str:
    if isinstance(config, TrainConfig):
        return config.env_id or "toy"
    if isinstance(config, HarnessConfig):
        return config.baseline.env_id
    return config.env_id


def run_multi_seed_baseline(
    config: HarnessConfig | BaselineConfig | TrainConfig,
    method: str | None = None,
    seeds: list[int] | None = None,
    output_dir: Path | str | None = None,
) -> MultiSeedResult:
    """
    Run benchmark across multiple seeds with identical config except seed.

    Supports BaselineConfig (PEBBLE/PrefPPO) and TrainConfig (SPARC).
    """
    if isinstance(config, HarnessConfig):
        method = method or config.method
        inner: BaselineConfig | TrainConfig = config.baseline
        seed_list = seeds or config.seeds
    elif isinstance(config, TrainConfig):
        method = method or "sparc"
        inner = config
        seed_list = seeds or [config.seed]
    else:
        method = method or config.method
        inner = config
        seed_list = seeds or [config.seed]

    env_id = _resolve_env_id(config, method or "pebble")
    out_path = Path(output_dir) if output_dir else None
    if out_path is not None:
        out_path.mkdir(parents=True, exist_ok=True)

    suite_t0 = time.perf_counter()
    runs: list[BaselineRunResult] = []
    for seed in seed_list:
        run_cfg = replace(inner, seed=seed)
        if out_path is not None:
            run_cfg = replace(run_cfg, log_path=str(out_path / f"seed_{seed}.jsonl"))
        result = run_baseline(run_cfg, method=method)
        result = replace(result, regret_drift=regret_from_baseline_result(result))
        runs.append(result)
        if out_path is not None:
            seed_json = out_path / f"seed_{seed}.json"
            seed_json.write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")

    summary = summarize_seed_runs(
        [r.asymptotic_return for r in runs],
        [r.asymptotic_true_return for r in runs],
        [r.total_queries for r in runs],
        [r.wall_clock_sec for r in runs],
    )
    elapsed = time.perf_counter() - suite_t0
    multi = MultiSeedResult(
        method=method or "pebble",
        env_id=env_id,
        seeds=seed_list,
        runs=runs,
        summary=summary,
        total_wall_clock_sec=elapsed,
    )
    if out_path is not None:
        summary_path = out_path / "summary.json"
        summary_path.write_text(json.dumps(multi.to_dict(), indent=2), encoding="utf-8")
    return multi
