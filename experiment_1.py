#!/usr/bin/env python3
"""Experiment 1 — factored latent vs VPL-style global-b baseline.

Result so far: NEGATIVE (see PROJECT.md §5.1). Do not tune to erase the negative;
queue item 3 asks whether difficulty-stratified context sets change the story.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor

from plr.encoder import FactoredEncoder, GlobalConsistencyEncoder, annotation_features
from plr.likelihood import plackett_luce_logp, unit
from plr.users import Population, answer, sample_population


def _random_queries(n: int, K: int, d: int, g: torch.Generator) -> Tensor:
    return torch.randn(n, K, d, generator=g)


def _pack_user_feats(
    pop: Population,
    queries: Tensor,
    n_context: int,
    g: torch.Generator,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Build context features + held-out targets for each user (I5: disjoint)."""
    N, d = queries.shape[0], queries.shape[2]
    assert n_context < N
    # Same query pool; per user shuffle which are context vs target
    B = pop.w.shape[0]
    feat_dim = 2 * d + 2
    ctx_feats = torch.zeros(B, n_context, feat_dim)
    masks = torch.ones(B, n_context, dtype=torch.bool)
    target_phi = []
    target_order = []
    for i in range(B):
        perm = torch.randperm(N, generator=g)
        ctx_idx = perm[:n_context]
        tgt_idx = perm[n_context : n_context + 1]
        for j, q in enumerate(ctx_idx.tolist()):
            phi = queries[q]
            order = answer(phi, pop.w[i], pop.b[i], generator=g)
            ctx_feats[i, j] = annotation_features(phi, order)
        phi_t = queries[tgt_idx[0].item()]
        order_t = answer(phi_t, pop.w[i], pop.b[i], generator=g)
        target_phi.append(phi_t)
        target_order.append(order_t)
    return ctx_feats, masks, torch.stack(target_phi), torch.stack(target_order)


def _train(
    model: torch.nn.Module,
    kind: str,
    ctx_feats: Tensor,
    masks: Tensor,
    target_phi: Tensor,
    target_order: Tensor,
    steps: int,
    lr: float,
) -> None:
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        if kind == "factored":
            w, b = model.rsample(ctx_feats, masks)
            kl = model.kl_to_standard_normal(ctx_feats, masks).mean()
        else:
            w, b = model.rsample(ctx_feats, masks)
            kl = torch.tensor(0.0)
        logp = plackett_luce_logp(target_phi, w, b, target_order).mean()
        loss = -logp + 0.01 * kl
        loss.backward()
        opt.step()


@torch.no_grad()
def _eval(
    model: torch.nn.Module,
    kind: str,
    ctx_feats: Tensor,
    masks: Tensor,
    target_phi: Tensor,
    target_order: Tensor,
    pop: Population,
) -> dict:
    if kind == "factored":
        w_mu, _, log_b_mu, _ = model.forward(ctx_feats, masks)
        w_hat = unit(w_mu)
        b_hat = log_b_mu.exp()
    else:
        w_mu, _, b = model.forward(ctx_feats, masks)
        w_hat = unit(w_mu)
        b_hat = b
    cos = F.cosine_similarity(w_hat, pop.w, dim=-1).mean().item()
    # corr(b) — undefined / n/a when spread=0 for baseline narrative; still compute
    b_true = pop.b
    if b_true.std() < 1e-8:
        corr_b = float("nan")
    else:
        corr_b = torch.corrcoef(torch.stack([b_hat, b_true]))[0, 1].item()
    logp = plackett_luce_logp(target_phi, w_hat, b_hat, target_order).mean().item()
    return {"cos_w": cos, "corr_b": corr_b, "heldout_logp": logp}


def run(spread: float, seed: int, n_users: int = 128, d: int = 8, steps: int = 400) -> list[dict]:
    g = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)
    pop = sample_population(n_users, d, consistency_spread=spread, seed=seed)
    queries = _random_queries(24, K=2, d=d, g=g)
    ctx, mask, tgt_phi, tgt_order = _pack_user_feats(pop, queries, n_context=8, g=g)

    results = []
    for kind, model in [
        ("baseline", GlobalConsistencyEncoder(d)),
        ("factored", FactoredEncoder(d)),
    ]:
        _train(model, kind, ctx, mask, tgt_phi, tgt_order, steps=steps, lr=1e-3)
        metrics = _eval(model, kind, ctx, mask, tgt_phi, tgt_order, pop)
        metrics.update({"spread": spread, "model": kind, "seed": seed})
        results.append(metrics)
        print(json.dumps(metrics))
    return results


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--spreads", type=float, nargs="+", default=[0.0, 0.4, 0.8])
    p.add_argument("--n-users", type=int, default=128)
    p.add_argument("--steps", type=int, default=400)
    p.add_argument("--out", type=Path, default=Path("experiments/runs/exp1_results.jsonl"))
    args = p.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        for spread in args.spreads:
            for row in run(spread, args.seed, n_users=args.n_users, steps=args.steps):
                f.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
