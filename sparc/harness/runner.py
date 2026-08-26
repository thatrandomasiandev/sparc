"""Run benchmark methods (baselines + SPARC) inside the shared harness."""

from __future__ import annotations

from dataclasses import replace

from sparc.baselines.aprel import AprelTrainer
from sparc.baselines.config import BaselineConfig
from sparc.baselines.pebble import PebbleTrainer
from sparc.baselines.prefppo import PrefPpoTrainer
from sparc.baselines.rune import RuneTrainer
from sparc.baselines.surf import SurfTrainer
from sparc.engine.config import TrainConfig
from sparc.harness.config import HarnessConfig
from sparc.harness.drift_regret import regret_from_baseline_result
from sparc.harness.env_defaults import apply_env_defaults
from sparc.harness.metrics import BaselineRunResult
from sparc.harness.sparc_runner import run_sparc


def _finalize_result(result: BaselineRunResult) -> BaselineRunResult:
    return replace(result, regret_drift=regret_from_baseline_result(result))


def run_baseline(
    config: HarnessConfig | BaselineConfig | TrainConfig,
    method: str | None = None,
) -> BaselineRunResult:
    """
    Execute one harness run (PEBBLE, PrefPPO, or SPARC).

    Accepts BaselineConfig for baselines, TrainConfig for SPARC, or HarnessConfig.
    """
    if isinstance(config, HarnessConfig):
        method = method or config.method
        if method == "sparc":
            raise TypeError("HarnessConfig for SPARC must wrap TrainConfig in baseline field")
        baseline_cfg = config.baseline
    elif isinstance(config, TrainConfig):
        method = method or "sparc"
        if not (method == "sparc" or method.startswith("sparc_")):
            raise ValueError("TrainConfig requires method='sparc' or sparc_* ablation")
        result = run_sparc(config)
        return replace(result, method=method)
    else:
        baseline_cfg = config
        method = method or baseline_cfg.method

    if method == "pebble":
        return _finalize_result(PebbleTrainer(apply_env_defaults(baseline_cfg)).run())
    if method == "prefppo":
        return _finalize_result(PrefPpoTrainer(apply_env_defaults(baseline_cfg)).run())
    if method == "rune":
        return _finalize_result(RuneTrainer(apply_env_defaults(baseline_cfg)).run())
    if method == "surf":
        return _finalize_result(SurfTrainer(apply_env_defaults(baseline_cfg)).run())
    if method == "aprel":
        return _finalize_result(AprelTrainer(apply_env_defaults(baseline_cfg)).run())
    if method == "sparc":
        raise ValueError("use TrainConfig for method='sparc'")
    raise ValueError(f"unsupported method: {method}")
