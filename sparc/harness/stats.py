"""Multi-seed summary statistics (EXPERIMENTS.md Week 18)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:
    from scipy import stats as scipy_stats
except ImportError:  # pragma: no cover - optional for CI without scipy
    scipy_stats = None


@dataclass(frozen=True)
class SeedSummary:
    """Aggregated metrics across seeds for one (method, env) run."""

    n_seeds: int
    mean_asymptotic_return: float
    std_asymptotic_return: float
    ci95_low: float
    ci95_high: float
    mean_asymptotic_true_return: float
    mean_total_queries: float
    std_total_queries: float
    mean_wall_clock_sec: float


def _t_critical_975(df: int) -> float:
    if scipy_stats is not None:
        return float(scipy_stats.t.ppf(0.975, df))
    # Normal approx fallback (conservative for small n would need table)
    table = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 9: 2.262, 19: 2.093, 29: 2.045}
    if df in table:
        return table[df]
    return 1.96


def confidence_interval_95(values: list[float] | np.ndarray) -> tuple[float, float, float]:
    """Return (mean, ci_low, ci_high) using Student-t 95% CI."""
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return 0.0, 0.0, 0.0
    mean = float(np.mean(arr))
    if arr.size == 1:
        return mean, mean, mean
    sem = float(np.std(arr, ddof=1) / np.sqrt(arr.size))
    tcrit = _t_critical_975(int(arr.size) - 1)
    return mean, mean - tcrit * sem, mean + tcrit * sem


def welch_ttest(a: list[float], b: list[float]) -> tuple[float, float]:
    """
    Welch's t-test (two-sided).

    Returns (t_statistic, p_value). Requires scipy; otherwise (nan, nan).
    """
    if scipy_stats is None or len(a) < 2 or len(b) < 2:
        return float("nan"), float("nan")
    result = scipy_stats.ttest_ind(a, b, equal_var=False)
    return float(result.statistic), float(result.pvalue)


def summarize_seed_runs(
    asymptotic_returns: list[float],
    asymptotic_true_returns: list[float],
    total_queries: list[int],
    wall_clock_secs: list[float],
) -> SeedSummary:
    mean_r, ci_lo, ci_hi = confidence_interval_95(asymptotic_returns)
    std_r = float(np.std(asymptotic_returns, ddof=1)) if len(asymptotic_returns) > 1 else 0.0
    mean_true = float(np.mean(asymptotic_true_returns)) if asymptotic_true_returns else 0.0
    mean_q = float(np.mean(total_queries)) if total_queries else 0.0
    std_q = float(np.std(total_queries, ddof=1)) if len(total_queries) > 1 else 0.0
    mean_wall = float(np.mean(wall_clock_secs)) if wall_clock_secs else 0.0
    return SeedSummary(
        n_seeds=len(asymptotic_returns),
        mean_asymptotic_return=mean_r,
        std_asymptotic_return=std_r,
        ci95_low=ci_lo,
        ci95_high=ci_hi,
        mean_asymptotic_true_return=mean_true,
        mean_total_queries=mean_q,
        std_total_queries=std_q,
        mean_wall_clock_sec=mean_wall,
    )
