"""Regret under preference drift (EXPERIMENTS.md metric 3)."""

from __future__ import annotations

from sparc.harness.metrics import BaselineRunResult, EvalMetrics


def drift_step_index(train_steps: int, drift_fraction: float = 0.5) -> int:
    return int(drift_fraction * train_steps)


def post_drift_eval_window(
    train_steps: int,
    drift_fraction: float = 0.5,
    window_fraction: float = 0.1,
) -> tuple[int, int]:
    """Inclusive env-step range [start, end] after Drift-1."""
    start = drift_step_index(train_steps, drift_fraction)
    end = start + int(window_fraction * train_steps)
    return start, end


def regret_under_drift(
    eval_history: list[EvalMetrics],
    train_steps: int,
    drift_fraction: float = 0.5,
    window_fraction: float = 0.1,
) -> float | None:
    """
    Mean shortfall vs. best post-drift true return in the evaluation window.

    EXPERIMENTS.md defines regret against an oracle policy under post-drift Op-2
    preferences. We approximate R* as the max ``mean_true_return`` observed in the
    post-drift eval window on the same run (best checkpoint proxy).
    """
    start, end = post_drift_eval_window(train_steps, drift_fraction, window_fraction)
    post = [m for m in eval_history if start <= m.env_steps <= end]
    if not post:
        return None
    best = max(m.mean_true_return for m in post)
    return sum(best - m.mean_true_return for m in post) / len(post)


def regret_from_run_dict(run: dict, train_steps: int | None = None) -> float | None:
    """Compute regret from a serialized ``BaselineRunResult`` dict."""
    steps = train_steps if train_steps is not None else int(run.get("total_env_steps", 0))
    if steps <= 0:
        return None
    history = [
        EvalMetrics(
            env_steps=int(h["env_steps"]),
            mean_return=float(h["mean_return"]),
            mean_true_return=float(h["mean_true_return"]),
            total_queries=int(h["total_queries"]),
            success_rate=h.get("success_rate"),
        )
        for h in run.get("eval_history", [])
    ]
    return regret_under_drift(history, steps)


def regret_from_baseline_result(result: BaselineRunResult) -> float | None:
    return regret_under_drift(result.eval_history, result.total_env_steps)
