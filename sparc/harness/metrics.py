"""Evaluation metrics per EXPERIMENTS.md."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class EvalMetrics:
    """Snapshot from one evaluation checkpoint."""

    env_steps: int
    mean_return: float
    mean_true_return: float
    total_queries: int
    success_rate: float | None = None


@dataclass
class BaselineRunResult:
    """Outcome of one baseline training run."""

    method: str
    seed: int
    env_id: str
    total_queries: int
    total_env_steps: int
    wall_clock_sec: float
    eval_history: list[EvalMetrics] = field(default_factory=list)
    final_train_acc: float = 0.0
    regret_drift: float | None = None

    @property
    def asymptotic_return(self) -> float:
        if not self.eval_history:
            return 0.0
        tail = self.eval_history[-5:]
        return sum(m.mean_return for m in tail) / len(tail)

    @property
    def asymptotic_true_return(self) -> float:
        if not self.eval_history:
            return 0.0
        tail = self.eval_history[-5:]
        return sum(m.mean_true_return for m in tail) / len(tail)

    def to_dict(self) -> dict:
        return {
            "method": self.method,
            "seed": self.seed,
            "env_id": self.env_id,
            "total_queries": self.total_queries,
            "total_env_steps": self.total_env_steps,
            "wall_clock_sec": self.wall_clock_sec,
            "asymptotic_return": self.asymptotic_return,
            "asymptotic_true_return": self.asymptotic_true_return,
            "final_train_acc": self.final_train_acc,
            "regret_drift": self.regret_drift,
            "eval_history": [m.__dict__ for m in self.eval_history],
        }
