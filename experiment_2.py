#!/usr/bin/env python3
"""Experiment 2 — is consistency recoverable under controlled query difficulty?

Supplies the *true* w so the question is isolated. Easy queries are catastrophic
(median abs error ~23); hard/mixed recover b. Caveat: random scored corr=0.753 —
never report the hard-query number without that caveat (PROJECT.md §5.2).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import Tensor

from plr.likelihood import plackett_luce_logp, reward_gap, unit
from plr.users import answer, sample_population


def _make_query(d: int, regime: str, w: Tensor, g: torch.Generator) -> Tensor:
    """Build a 2-option query with controlled difficulty under true w."""
    if regime == "random":
        return torch.randn(2, d, generator=g)

    base = torch.randn(2, d, generator=g)
    w_u = unit(w)
    if regime == "easy":
        base[0] = base[0] + 3.0 * w_u
        base[1] = base[1] - 3.0 * w_u
    elif regime == "hard":
        shared = torch.randn(d, generator=g)
        base[0] = shared + 0.05 * w_u
        base[1] = shared - 0.05 * w_u
    elif regime == "mixed":
        base[0] = base[0] + 0.8 * w_u
        base[1] = base[1] - 0.8 * w_u
    else:
        raise ValueError(regime)
    return base


def mle_b(
    queries: Tensor,
    orders: Tensor,
    w_true: Tensor,
    grid: Tensor,
) -> float:
    """Grid MLE for b given true w."""
    best_lp = -float("inf")
    best_b = float(grid[0])
    for b in grid:
        b_t = torch.tensor(b)
        lp = 0.0
        for i in range(queries.shape[0]):
            lp = lp + float(
                plackett_luce_logp(
                    queries[i].unsqueeze(0),
                    w_true.unsqueeze(0),
                    b_t.unsqueeze(0),
                    orders[i].unsqueeze(0),
                )
            )
        if lp > best_lp:
            best_lp = lp
            best_b = float(b)
    return best_b


def run_regime(regime: str, seed: int, n_users: int = 80, n_queries: int = 20, d: int = 6) -> dict:
    g = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)
    pop = sample_population(n_users, d, consistency_spread=0.5, b_median=3.0, seed=seed)
    grid = torch.linspace(0.2, 40.0, 80)
    b_hats = []
    gaps = []
    for i in range(n_users):
        qs = []
        ords = []
        for _ in range(n_queries):
            phi = _make_query(d, regime, pop.w[i], g)
            order = answer(phi, pop.w[i], pop.b[i], generator=g)
            qs.append(phi)
            ords.append(order)
            gaps.append(float(reward_gap(phi, pop.w[i])))
        b_hat = mle_b(torch.stack(qs), torch.stack(ords), pop.w[i], grid)
        b_hats.append(b_hat)
    b_hat_t = torch.tensor(b_hats)
    corr = torch.corrcoef(torch.stack([b_hat_t, pop.b]))[0, 1].item()
    med_abs = float((b_hat_t - pop.b).abs().median())
    out = {
        "regime": regime,
        "corr_b": corr,
        "median_abs_error": med_abs,
        "mean_gap": sum(gaps) / len(gaps),
        "seed": seed,
        "n_users": n_users,
    }
    print(json.dumps(out))
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--regimes", nargs="+", default=["easy", "hard", "mixed", "random"])
    p.add_argument("--out", type=Path, default=Path("experiments/runs/exp2_results.jsonl"))
    args = p.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        for regime in args.regimes:
            row = run_regime(regime, args.seed)
            f.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
