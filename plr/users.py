"""Simulated user populations with ground-truth preference ``(w, b)``.

``consistency_spread`` is the experimental independent variable: 0 means everyone is
equally reliable (the world VPL implicitly assumes), higher values spread ``b``.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from plr.likelihood import plackett_luce_logp, unit


@dataclass
class Population:
    """Ground-truth users."""

    w: Tensor  # (N, d) unit-norm
    b: Tensor  # (N,) positive


def sample_population(
    n: int,
    d: int,
    *,
    consistency_spread: float = 0.0,
    b_median: float = 3.0,
    seed: int | None = None,
    device: torch.device | None = None,
) -> Population:
    """Draw ``n`` users with unit-norm ``w`` and log-normal-ish ``b``.

    At ``consistency_spread == 0`` every user shares ``b_median``. Otherwise
    ``log b ~ N(log b_median, consistency_spread)``.
    """
    if n <= 0 or d <= 0:
        raise ValueError("n and d must be positive")
    if b_median <= 0:
        raise ValueError("b_median must be positive")
    if consistency_spread < 0:
        raise ValueError("consistency_spread must be non-negative")

    g = torch.Generator(device="cpu")
    if seed is not None:
        g.manual_seed(seed)

    w = torch.randn(n, d, generator=g)
    w = unit(w)
    if consistency_spread == 0.0:
        b = torch.full((n,), float(b_median))
    else:
        log_b = math_log(b_median) + consistency_spread * torch.randn(n, generator=g)
        b = log_b.exp()

    if device is not None:
        w = w.to(device)
        b = b.to(device)
    return Population(w=w, b=b)


def math_log(x: float) -> float:
    import math

    return math.log(x)


def sample_gaussian_mixture_population(
    n: int,
    d: int,
    *,
    mode_centers: Tensor,
    mode_weights: Tensor | None = None,
    b_median: float = 3.0,
    consistency_spread: float = 0.3,
    seed: int | None = None,
) -> Population:
    """Three-mode (or M-mode) population prior used by experiment 3.

    Centers are unit-normalized. Users are drawn by picking a mode then adding isotropic
    Gaussian noise in ambient space and re-normalizing.
    """
    # mode_centers: (M, d)
    if mode_centers.dim() != 2 or mode_centers.shape[1] != d:
        raise ValueError("mode_centers must have shape (M, d)")
    M = mode_centers.shape[0]
    g = torch.Generator(device="cpu")
    if seed is not None:
        g.manual_seed(seed)

    if mode_weights is None:
        mode_weights = torch.ones(M) / M
    mode_weights = mode_weights / mode_weights.sum()
    modes = torch.multinomial(mode_weights, n, replacement=True, generator=g)  # (N,)
    centers = unit(mode_centers)[modes]  # (N, d)
    noise = 0.35 * torch.randn(n, d, generator=g)
    w = unit(centers + noise)

    if consistency_spread == 0.0:
        b = torch.full((n,), float(b_median))
    else:
        log_b = math_log(b_median) + consistency_spread * torch.randn(n, generator=g)
        b = log_b.exp()
    return Population(w=w, b=b)


def answer(
    phi: Tensor,
    w: Tensor,
    b: Tensor,
    *,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Sample a Plackett-Luce ranking over the K options in ``phi``.

    Uses the same model the likelihood assumes — deliberately generous; model mismatch is
    a separate experiment (queue item 5).
    """
    # phi: (K, d) or (B, K, d); w: (d,) or (B, d); b: scalar or (B,)
    if phi.dim() == 2:
        phi = phi.unsqueeze(0)
        squeeze = True
    else:
        squeeze = False
    # phi: (B, K, d)
    B, K, _ = phi.shape
    from plr.likelihood import rewards

    r = rewards(phi, w if w.dim() > 1 else w.unsqueeze(0).expand(B, -1))  # (B, K)
    b_t = b if isinstance(b, Tensor) else torch.as_tensor(b, dtype=phi.dtype)
    if b_t.dim() == 0:
        b_t = b_t.expand(B)
    scaled = b_t.unsqueeze(-1) * r  # (B, K)

    remaining = torch.ones(B, K, dtype=torch.bool, device=phi.device)
    order = torch.empty(B, K, dtype=torch.long, device=phi.device)
    for k in range(K):
        neg_inf = torch.full_like(scaled, -float("inf"))
        logits = torch.where(remaining, scaled, neg_inf)
        # Gumbel-max for sampling without requiring a batched Categorical mask API.
        u = torch.rand(logits.shape, generator=generator, device=logits.device, dtype=logits.dtype)
        gumbel = logits - torch.log(-torch.log(u.clamp_min(1e-8)))
        choice = gumbel.argmax(dim=-1)  # (B,)
        order[:, k] = choice
        remaining.scatter_(1, choice.unsqueeze(-1), False)

    return order.squeeze(0) if squeeze else order


def ranking_log_prob(phi: Tensor, w: Tensor, b: Tensor, order: Tensor) -> Tensor:
    """Likelihood of an observed ranking — thin wrapper for call sites."""
    return plackett_luce_logp(phi, w, b, order)
