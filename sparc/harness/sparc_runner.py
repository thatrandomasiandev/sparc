"""Run SPARC through the shared benchmark harness."""

from __future__ import annotations

from dataclasses import replace

from sparc.engine.config import TrainConfig
from sparc.engine.trainer import SparcTrainer
from sparc.harness.drift_regret import regret_from_baseline_result
from sparc.harness.metrics import BaselineRunResult
from sparc.harness.sparc_defaults import apply_sparc_env_defaults
from sparc.harness.stress_defaults import apply_stress_defaults


def run_sparc(config: TrainConfig) -> BaselineRunResult:
    """Execute one SPARC training run and return harness-comparable metrics."""
    cfg = apply_stress_defaults(config)
    result = SparcTrainer(cfg).run()
    env_id = cfg.env_id or "toy"
    baseline = BaselineRunResult(
        method="sparc",
        seed=cfg.seed,
        env_id=env_id,
        total_queries=result.total_queries,
        total_env_steps=result.total_env_steps,
        wall_clock_sec=result.wall_clock_sec,
        eval_history=result.eval_history,
        final_train_acc=1.0 - min(result.final_reward_loss, 1.0),
    )
    return replace(baseline, regret_drift=regret_from_baseline_result(baseline))
