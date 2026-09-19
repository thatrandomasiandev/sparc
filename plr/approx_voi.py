"""Approximate decision-relevant query scoring for deep reward ensembles (L6).

Mirrors tabular POP-VOI: score a pairwise segment query by expected reduction in a
*policy-relevant loss* under a particle/ensemble belief — not by parameter
disagreement alone.

Loss used here (transferable without exact VI)::

    Loss(β) = mean_n Var_{e ~ β}[ R_e(seg_n) ]

where ``seg_n`` are held-out evaluation segments and ``R_e`` is ensemble member
``e``'s predicted discounted return. Queries that shrink return uncertainty on
those segments score high; queries that only move irrelevant reward dimensions
score low.

B-Pref ``disagreement_sampling`` is the parameter-space baseline (variance of
preference probs across ensemble members).
"""

from __future__ import annotations

import torch
from torch import Tensor
from torch.nn.functional import binary_cross_entropy_with_logits, softmax


def segment_returns(reward_fns: list, segments: Tensor) -> Tensor:
    """``segments: (N, T, d)`` → ``(E, N)`` predicted returns (sum of step rewards)."""
    # segments may be state-only features; each reward_fn maps (B, d) -> (B,)
    outs = []
    for fn in reward_fns:
        # (N, T, d) -> (N*T, d)
        N, T, d = segments.shape
        flat = segments.reshape(N * T, d)
        r = fn(flat).reshape(N, T).sum(dim=-1)  # (N,)
        outs.append(r)
    return torch.stack(outs, dim=0)  # (E, N)


def return_variance_loss(weights: Tensor, returns: Tensor) -> Tensor:
    """Weighted ensemble variance of predicted returns, averaged over segments.

    ``weights: (E,)``, ``returns: (E, N)``.
    """
    # E_w[R], E_w[R^2]
    mean = (weights.unsqueeze(-1) * returns).sum(dim=0)  # (N,)
    second = (weights.unsqueeze(-1) * returns.square()).sum(dim=0)
    var = (second - mean.square()).clamp_min(0.0)
    return var.mean()


def preference_logits(reward_fns: list, seg_a: Tensor, seg_b: Tensor) -> Tensor:
    """``(E,)`` logits = R_e(a) - R_e(b) for a single pair ``(T, d)`` each."""
    a = seg_a.unsqueeze(0)  # (1, T, d)
    b = seg_b.unsqueeze(0)
    ra = segment_returns(reward_fns, a).squeeze(-1)  # (E,)
    rb = segment_returns(reward_fns, b).squeeze(-1)
    return ra - rb


def disagreement_score(reward_fns: list, seg_a: Tensor, seg_b: Tensor) -> float:
    """B-Pref-style: variance of P(a ≻ b) across ensemble members."""
    logits = preference_logits(reward_fns, seg_a, seg_b)  # (E,)
    probs = torch.sigmoid(logits)
    return float(probs.var(unbiased=False).item())


def approx_voi_score(
    reward_fns: list,
    weights: Tensor,
    seg_a: Tensor,
    seg_b: Tensor,
    eval_segments: Tensor,
    *,
    temperature: float = 1.0,
) -> float:
    """Expected drop in return-variance loss from asking ``(seg_a, seg_b)``.

    Answer model: Bradley–Terry with ensemble logits; posterior = reweight members.
    """
    returns = segment_returns(reward_fns, eval_segments)  # (E, N)
    loss_now = return_variance_loss(weights, returns)

    logits = preference_logits(reward_fns, seg_a, seg_b) / temperature  # (E,)
    # P(y=1 | e) = sigmoid(logit_e) for "a preferred"
    p_a_given_e = torch.sigmoid(logits)
    # Marginal P(y=1)
    p_a = (weights * p_a_given_e).sum()
    p_b = 1.0 - p_a

    def posterior(y_a: bool) -> Tensor:
        like = p_a_given_e if y_a else (1.0 - p_a_given_e)
        lw = torch.log(weights.clamp_min(1e-12)) + torch.log(like.clamp_min(1e-12))
        return softmax(lw, dim=0)

    loss_after = p_a * return_variance_loss(posterior(True), returns) + p_b * return_variance_loss(
        posterior(False), returns
    )
    return float((loss_now - loss_after).item())


def fit_step(
    models: list[torch.nn.Module],
    opt: torch.optim.Optimizer,
    seg_a: Tensor,
    seg_b: Tensor,
    prefer_a: bool,
) -> float:
    """One SGD step of BT loss on all ensemble members (shared batch)."""
    opt.zero_grad()
    losses = []
    target = torch.tensor([1.0 if prefer_a else 0.0])
    for m in models:
        # m: (B,d)->(B,); segments (T,d)
        ra = m(seg_a).sum()
        rb = m(seg_b).sum()
        logit = (ra - rb).unsqueeze(0)
        losses.append(binary_cross_entropy_with_logits(logit, target))
    loss = torch.stack(losses).mean()
    loss.backward()
    opt.step()
    return float(loss.item())
