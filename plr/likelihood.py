"""Bradley-Terry and Plackett-Luce answer models.

Invariant I1 lives here: under BT/PL only the product ``b * ||w||`` is identifiable, so we
pin ``||w|| = 1`` and put all scale into consistency ``b``. Callers must never compute
``w @ phi`` directly — use ``rewards`` / ``unit``.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor
from torch.nn.functional import logsigmoid


def unit(w: Tensor, eps: float = 1e-8) -> Tensor:
    """Normalize ``w`` along the last axis with a clamped norm.

    Clamping avoids NaNs when a particle collapses to the zero vector during early
    training; a tiny vector becomes a unit vector in an arbitrary direction, which is
    preferable to poisoning the likelihood with NaN.
    """
    # w: (..., d)
    norm = torch.linalg.vector_norm(w, dim=-1, keepdim=True).clamp_min(eps)
    return w / norm


def rewards(phi: Tensor, w: Tensor) -> Tensor:
    """Linear rewards ``r = unit(w) · phi``.

    Normalizes internally so a caller that forgot I1 cannot silently double-count scale
    into ``b``.
    """
    # phi: (..., K, d) or (..., d); w: (..., d) or (d,)
    w_u = unit(w)
    if phi.dim() >= w_u.dim() + 1:
        # phi: (..., K, d); w: (..., d)
        return (phi * w_u.unsqueeze(-2)).sum(dim=-1)
    return (phi * w_u).sum(dim=-1)


def _as_positive_b(b: Tensor) -> Tensor:
    if (b <= 0).any():
        raise ValueError("consistency b must be strictly positive")
    return b


def binary_logp(delta: Tensor, w: Tensor, b: Tensor) -> Tensor:
    """Log Bradley-Terry probability that A is preferred given ``delta = f_A - f_B``.

    Uses ``logsigmoid`` because ``log(sigmoid(x))`` underflows once ``b * gap`` exceeds
    ~30, which happens routinely for confident users on easy queries.
    """
    # delta: (..., d); w: (..., d); b: (...,) or scalar tensor
    if not isinstance(b, Tensor):
        b = torch.as_tensor(b, dtype=delta.dtype, device=delta.device)
    b = _as_positive_b(b)
    w_u = unit(w)
    gap = (delta * w_u).sum(dim=-1)  # (...)
    return logsigmoid(b * gap)


def binary_logp_from_options(phi: Tensor, w: Tensor, b: Tensor, prefer: int = 0) -> Tensor:
    """Binary log-prob from a 2-option feature matrix ``phi`` with shape ``(..., 2, d)``."""
    # phi: (..., 2, d)
    if phi.shape[-2] != 2:
        raise ValueError(f"binary query expects K=2 options, got K={phi.shape[-2]}")
    delta = phi[..., 0, :] - phi[..., 1, :]
    log_p0 = binary_logp(delta, w, b)
    return log_p0 if prefer == 0 else logsigmoid(-_as_positive_b(b) * (unit(w) * delta).sum(dim=-1))


def plackett_luce_logp(phi: Tensor, w: Tensor, b: Tensor, order: Tensor) -> Tensor:
    """Log Plackett-Luce probability of a full ranking ``order`` over K options.

    ``order[..., k]`` is the option index chosen at stage k (best-first). Must agree with
    Bradley-Terry exactly at K=2 — asserted in tests.
    """
    # phi: (..., K, d); order: (..., K) long
    if not isinstance(b, Tensor):
        b = torch.as_tensor(b, dtype=phi.dtype, device=phi.device)
    b = _as_positive_b(b)
    r = rewards(phi, w)  # (..., K)
    scaled = b.unsqueeze(-1) * r  # (..., K)
    K = phi.shape[-2]
    logp = torch.zeros(phi.shape[:-2], dtype=phi.dtype, device=phi.device)
    remaining = torch.ones_like(scaled, dtype=torch.bool)
    for k in range(K - 1):
        idx = order[..., k]  # (...,)
        chosen = torch.gather(scaled, -1, idx.unsqueeze(-1)).squeeze(-1)  # (...)
        # Mask already-chosen options out of the logsumexp.
        neg_inf = torch.full_like(scaled, -math.inf)
        pool = torch.where(remaining, scaled, neg_inf)
        logp = logp + chosen - torch.logsumexp(pool, dim=-1)
        remaining = remaining.scatter(-1, idx.unsqueeze(-1), False)
    return logp


def reward_gap(phi: Tensor, w: Tensor) -> Tensor:
    """Difficulty of a query: smallest gap between consecutive sorted option rewards.

    Defined **without** ``b`` (invariant I2): difficulty is a property of the question,
    so we can schedule it before knowing the user. Small gap ⇒ hard.
    """
    # phi: (..., K, d); w: (..., d) or (d,)
    r = rewards(phi, w)  # (..., K)
    r_sorted, _ = torch.sort(r, dim=-1)
    gaps = r_sorted[..., 1:] - r_sorted[..., :-1]  # (..., K-1)
    return gaps.min(dim=-1).values
