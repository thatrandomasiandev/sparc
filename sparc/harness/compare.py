"""Compare multi-seed benchmark summaries (EXPERIMENTS.md statistical tests)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sparc.harness.stats import welch_ttest


@dataclass(frozen=True)
class ComparisonReport:
    method_a: str
    method_b: str
    env_id_a: str
    env_id_b: str
    n_seeds_a: int
    n_seeds_b: int
    mean_r_inf_a: float
    mean_r_inf_b: float
    mean_queries_a: float
    mean_queries_b: float
    welch_t: float
    welch_p: float
    query_reduction_pct: float

    def to_dict(self) -> dict:
        return {
            "method_a": self.method_a,
            "method_b": self.method_b,
            "env_id_a": self.env_id_a,
            "env_id_b": self.env_id_b,
            "n_seeds_a": self.n_seeds_a,
            "n_seeds_b": self.n_seeds_b,
            "mean_r_inf_a": self.mean_r_inf_a,
            "mean_r_inf_b": self.mean_r_inf_b,
            "mean_queries_a": self.mean_queries_a,
            "mean_queries_b": self.mean_queries_b,
            "welch_t": self.welch_t,
            "welch_p": self.welch_p,
            "query_reduction_pct": self.query_reduction_pct,
        }


def _load_summary(path: Path) -> dict[str, Any]:
    data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return data


def _asymptotic_returns(summary: dict) -> list[float]:
    runs = summary.get("runs", [])
    out: list[float] = []
    for run in runs:
        history = run.get("eval_history", [])
        if not history:
            out.append(0.0)
            continue
        tail = history[-5:] if len(history) >= 5 else history
        out.append(sum(h["mean_return"] for h in tail) / len(tail))
    if not out and summary.get("summary"):
        out.append(float(summary["summary"]["mean_asymptotic_return"]))
    return out


def _mean_queries(summary: dict) -> float:
    if summary.get("summary"):
        return float(summary["summary"]["mean_total_queries"])
    runs = summary.get("runs", [])
    if not runs:
        return 0.0
    return sum(int(r["total_queries"]) for r in runs) / len(runs)


def compare_summaries(path_a: Path | str, path_b: Path | str) -> ComparisonReport:
    """Welch t-test on asymptotic return; query reduction B vs A."""
    a = _load_summary(Path(path_a))
    b = _load_summary(Path(path_b))
    returns_a = _asymptotic_returns(a)
    returns_b = _asymptotic_returns(b)
    t_stat, p_val = welch_ttest(returns_a, returns_b)
    q_a = _mean_queries(a)
    q_b = _mean_queries(b)
    reduction = 100.0 * (1.0 - q_b / q_a) if q_a > 0 else 0.0
    return ComparisonReport(
        method_a=str(a.get("method", "a")),
        method_b=str(b.get("method", "b")),
        env_id_a=str(a.get("env_id", "")),
        env_id_b=str(b.get("env_id", "")),
        n_seeds_a=len(returns_a),
        n_seeds_b=len(returns_b),
        mean_r_inf_a=float(sum(returns_a) / len(returns_a)) if returns_a else 0.0,
        mean_r_inf_b=float(sum(returns_b) / len(returns_b)) if returns_b else 0.0,
        mean_queries_a=q_a,
        mean_queries_b=q_b,
        welch_t=t_stat,
        welch_p=p_val,
        query_reduction_pct=reduction,
    )
