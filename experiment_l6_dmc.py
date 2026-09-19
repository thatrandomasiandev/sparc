#!/usr/bin/env python3
"""E-L6 — B-Pref-style acquisition transfer on dm_control (Claim C-l6-1).

Legacy B-Pref (Python 3.6 / PyTorch 1.4) does not install on this stack. This
experiment is the **minimum credible L6 proxy**: same *question* as B-Pref's
query-selection bake-off (random vs disagreement vs decision-relevant score)
on a continuous-control domain with an ensemble reward model.

Domain: dm_control ``cartpole`` / ``balance`` (fast; MuJoCo present).
Primary (pre-specified): mean true episode return of a CEM planner optimizing
the *learned mean reward*, after a fixed preference budget.
Secondary: Spearman corr of predicted vs true segment returns on a holdout pool.

Strategies: random | disagreement (B-Pref) | approx_voi (return-variance VOI).
Kill for C-l6-1: approx_voi not better than disagreement on primary (CI∋0).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

from plr.approx_voi import (
    approx_voi_score,
    disagreement_score,
    fit_step,
    segment_returns,
)

try:
    from dm_control import suite
except ImportError as e:  # pragma: no cover
    raise SystemExit("dm_control required for experiment_l6_dmc.py") from e


def _obs_vec(time_step) -> np.ndarray:
    parts = []
    for v in time_step.observation.values():
        parts.append(np.asarray(v, dtype=np.float32).ravel())
    return np.concatenate(parts, axis=0)


def collect_segments(
    domain: str,
    task: str,
    *,
    n_segments: int,
    seg_len: int,
    seed: int,
) -> tuple[Tensor, Tensor]:
    """Random-policy segments + true discounted returns (sum of env rewards)."""
    env = suite.load(domain, task, task_kwargs={"random": seed})
    spec = env.action_spec()
    g = np.random.default_rng(seed)
    segs = []
    true_r = []
    while len(segs) < n_segments:
        ts = env.reset()
        traj_x, traj_r = [], []
        for _ in range(seg_len * 4):
            if ts.last():
                break
            a = g.uniform(spec.minimum, spec.maximum, size=spec.shape).astype(np.float32)
            ts = env.step(a)
            traj_x.append(_obs_vec(ts))
            traj_r.append(float(ts.reward or 0.0))
        # cut non-overlapping segments
        for i in range(0, len(traj_x) - seg_len + 1, seg_len):
            segs.append(np.stack(traj_x[i : i + seg_len], axis=0))
            true_r.append(float(np.sum(traj_r[i : i + seg_len])))
            if len(segs) >= n_segments:
                break
    x = torch.tensor(np.stack(segs, axis=0), dtype=torch.float32)  # (N, T, d)
    r = torch.tensor(true_r[: n_segments], dtype=torch.float32)
    return x, r


class RewardNet(nn.Module):
    def __init__(self, d: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x).squeeze(-1)


def make_ensemble(d: int, e: int, seed: int) -> list[RewardNet]:
    g = torch.Generator().manual_seed(seed)
    models = []
    for i in range(e):
        m = RewardNet(d)
        # distinct init
        with torch.no_grad():
            for p in m.parameters():
                p.add_(0.01 * torch.randn(p.shape, generator=g))
        models.append(m)
    return models


def teacher_prefer_a(true_a: float, true_b: float, beta: float, g: np.random.Generator) -> bool:
    """Bradley–Terry teacher on true segment returns."""
    # P(a ≻ b) = sigmoid(beta * (Ra - Rb))
    p = 1.0 / (1.0 + np.exp(-beta * (true_a - true_b)))
    return bool(g.random() < p)


def cem_return(
    domain: str,
    task: str,
    reward_fn,
    *,
    seed: int,
    horizon: int = 40,
    pop: int = 32,
    elites: int = 8,
    iters: int = 4,
    n_eval: int = 3,
) -> float:
    """Open-loop CEM on predicted reward; evaluate true return of best plan."""
    env = suite.load(domain, task, task_kwargs={"random": seed + 7})
    spec = env.action_spec()
    act_dim = int(np.prod(spec.shape))
    lo = np.asarray(spec.minimum, dtype=np.float32).ravel()
    hi = np.asarray(spec.maximum, dtype=np.float32).ravel()
    g = np.random.default_rng(seed + 99)
    mean = np.zeros((horizon, act_dim), dtype=np.float32)
    std = np.ones((horizon, act_dim), dtype=np.float32) * 0.4

    def pred_return(plan: np.ndarray) -> float:
        ts = env.reset()
        total = 0.0
        for t in range(horizon):
            if ts.last():
                break
            a = plan[t].reshape(spec.shape)
            a = np.clip(a, lo.reshape(spec.shape), hi.reshape(spec.shape))
            ts = env.step(a)
            x = torch.tensor(_obs_vec(ts), dtype=torch.float32)
            with torch.no_grad():
                total += float(reward_fn(x.unsqueeze(0)).item())
        return total

    for _ in range(iters):
        plans = mean + std * g.normal(size=(pop, horizon, act_dim)).astype(np.float32)
        plans = np.clip(plans, lo, hi)
        scores = np.array([pred_return(p) for p in plans])
        keep = plans[np.argsort(scores)[-elites:]]
        mean = keep.mean(axis=0)
        std = keep.std(axis=0) + 1e-3

    # evaluate best mean plan under true reward
    true_scores = []
    for k in range(n_eval):
        env_e = suite.load(domain, task, task_kwargs={"random": seed + 1000 + k})
        ts = env_e.reset()
        total = 0.0
        for t in range(horizon):
            if ts.last():
                break
            a = mean[t].reshape(spec.shape)
            a = np.clip(a, lo.reshape(spec.shape), hi.reshape(spec.shape))
            ts = env_e.step(a)
            total += float(ts.reward or 0.0)
        true_scores.append(total)
    return float(np.mean(true_scores))


def mean_reward_fn(models: list[RewardNet]):
    def fn(x: Tensor) -> Tensor:
        # x: (B, d)
        preds = torch.stack([m(x) for m in models], dim=0).mean(dim=0)
        return preds

    return fn


def run_seed(
    seed: int,
    *,
    strategy: str,
    domain: str,
    task: str,
    n_pool: int,
    n_eval: int,
    n_queries: int,
    seg_len: int,
    ensemble: int,
    beta: float,
) -> dict:
    torch.manual_seed(seed)
    g = np.random.default_rng(seed)
    segments, true_returns = collect_segments(
        domain, task, n_segments=n_pool + n_eval, seg_len=seg_len, seed=seed
    )
    pool, holdout = segments[:n_pool], segments[n_pool:]
    true_pool, true_hold = true_returns[:n_pool], true_returns[n_pool:]
    d = pool.shape[-1]
    models = make_ensemble(d, ensemble, seed=seed + 1)
    params = [p for m in models for p in m.parameters()]
    opt = torch.optim.Adam(params, lr=3e-4)
    weights = torch.full((ensemble,), 1.0 / ensemble)

    # candidate pairs: random indices into pool
    n_cand = min(64, n_pool * (n_pool - 1) // 2)
    pairs = []
    while len(pairs) < n_cand:
        i, j = int(g.integers(0, n_pool)), int(g.integers(0, n_pool))
        if i != j and (i, j) not in pairs and (j, i) not in pairs:
            pairs.append((i, j))

    labeled = []
    for t in range(n_queries):
        if strategy == "random":
            q = int(g.integers(0, len(pairs)))
        else:
            scores = []
            for i, j in pairs:
                a, b = pool[i], pool[j]
                if strategy == "disagreement":
                    scores.append(disagreement_score(models, a, b))
                elif strategy == "approx_voi":
                    scores.append(
                        approx_voi_score(models, weights, a, b, holdout)
                    )
                else:
                    raise ValueError(strategy)
            q = int(np.argmax(scores))
        i, j = pairs[q]
        prefer_a = teacher_prefer_a(
            float(true_pool[i]), float(true_pool[j]), beta, g
        )
        # several SGD steps on the selected pair
        for _ in range(8):
            fit_step(models, opt, pool[i], pool[j], prefer_a)
        labeled.append((i, j, prefer_a))
        # drop selected pair from candidates
        pairs.pop(q)
        if not pairs:
            break

    # secondary: spearman of mean predicted vs true on holdout
    with torch.no_grad():
        pred = segment_returns(models, holdout).mean(dim=0).numpy()
    true = true_hold.numpy()
    if pred.std() < 1e-8 or true.std() < 1e-8:
        spearman = 0.0
    else:
        spearman = float(
            np.corrcoef(pred.argsort().argsort(), true.argsort().argsort())[0, 1]
        )

    ret = cem_return(domain, task, mean_reward_fn(models), seed=seed)
    return {
        "seed": seed,
        "strategy": strategy,
        "true_return": ret,
        "spearman_holdout": spearman,
        "n_queries": len(labeled),
        "domain": domain,
        "task": task,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4, 5])
    p.add_argument("--strategies", nargs="+", default=["random", "disagreement", "approx_voi"])
    p.add_argument("--domain", default="cartpole")
    p.add_argument("--task", default="balance")
    p.add_argument("--n-pool", type=int, default=80)
    p.add_argument("--n-eval", type=int, default=40)
    p.add_argument("--n-queries", type=int, default=20)
    p.add_argument("--seg-len", type=int, default=15)
    p.add_argument("--ensemble", type=int, default=5)
    p.add_argument("--beta", type=float, default=5.0)
    p.add_argument("--out", type=Path, default=Path("experiments/runs/exp_l6_dmc.jsonl"))
    args = p.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        for seed in args.seeds:
            for strategy in args.strategies:
                row = run_seed(
                    seed,
                    strategy=strategy,
                    domain=args.domain,
                    task=args.task,
                    n_pool=args.n_pool,
                    n_eval=args.n_eval,
                    n_queries=args.n_queries,
                    seg_len=args.seg_len,
                    ensemble=args.ensemble,
                    beta=args.beta,
                )
                print(json.dumps(row), flush=True)
                f.write(json.dumps(row) + "\n")
                f.flush()


if __name__ == "__main__":
    main()
