"""Factored variational encoder ``q(w, b | answers)``.

DeepSets-style: embed each annotation, mean-pool (order-invariant — a person is the same
whether you asked question 3 or 7 first), then separate heads for ``w`` and ``log b``.

Invariant I4: per-annotation features include option **spread** (+ option count). Removing
spread is a valid ablation; forgetting it silently is a fatal bug.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch import Tensor

from plr.likelihood import unit


def annotation_features(phi: Tensor, order: Tensor) -> Tensor:
    """Build per-annotation features for one or a batch of rankings.

    Features: direction ``best - mean(rest)``, context ``mean(options)``,
    spread ``||options - mean||`` mean, and option count.

    Parameters
    ----------
    phi : (N, K, d) or (K, d)
    order : (N, K) or (K,) — best-first option indices
    """
    if phi.dim() == 2:
        phi = phi.unsqueeze(0)
        order = order.unsqueeze(0)
        squeeze = True
    else:
        squeeze = False
    # phi: (N, K, d); order: (N, K)
    N, K, d = phi.shape
    best_idx = order[:, 0]  # (N,)
    best = phi[torch.arange(N), best_idx]  # (N, d)
    mean_all = phi.mean(dim=1)  # (N, d)
    # mean of non-best options
    rest_sum = phi.sum(dim=1) - best
    mean_rest = rest_sum / max(K - 1, 1)
    direction = best - mean_rest  # (N, d)
    context = mean_all  # (N, d)
    spread = (phi - mean_all.unsqueeze(1)).norm(dim=-1).mean(dim=-1, keepdim=True)  # (N, 1)
    k_feat = torch.full((N, 1), float(K), dtype=phi.dtype, device=phi.device)
    feats = torch.cat([direction, context, spread, k_feat], dim=-1)  # (N, 2d+2)
    return feats.squeeze(0) if squeeze else feats


class FactoredEncoder(nn.Module):
    """``q(w, log b | answers)`` with independent diagonal Gaussians after mean-pool."""

    def __init__(self, d: int, hidden: int = 64):
        super().__init__()
        self.d = d
        in_dim = 2 * d + 2
        self.embed = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.w_mu = nn.Linear(hidden, d)
        self.w_log_std = nn.Linear(hidden, d)
        self.b_mu = nn.Linear(hidden, 1)
        self.b_log_std = nn.Linear(hidden, 1)

    def encode_set(self, feats: Tensor, mask: Tensor | None = None) -> Tensor:
        """Mean-pool embedded annotations.

        feats: (B, N, F); mask: (B, N) True = valid
        """
        h = self.embed(feats)  # (B, N, H)
        if mask is None:
            return h.mean(dim=1)
        mask_f = mask.unsqueeze(-1).to(h.dtype)  # (B, N, 1)
        summed = (h * mask_f).sum(dim=1)
        denom = mask_f.sum(dim=1).clamp_min(1.0)
        return summed / denom  # (B, H)

    def forward(
        self,
        feats: Tensor,
        mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Return ``(w_mu, w_log_std, log_b_mu, log_b_log_std)``.

        Log-std clamped to [-5, 2]: lower bound prevents posterior collapse (KL explosion);
        upper bound prevents noise explosion (likelihood becomes meaningless).
        """
        # feats: (B, N, F)
        pooled = self.encode_set(feats, mask)  # (B, H)
        w_mu = self.w_mu(pooled)  # (B, d)
        w_ls = self.w_log_std(pooled).clamp(-5.0, 2.0)  # (B, d)
        log_b_mu = self.b_mu(pooled).squeeze(-1)  # (B,)
        log_b_ls = self.b_log_std(pooled).squeeze(-1).clamp(-5.0, 2.0)  # (B,)
        return w_mu, w_ls, log_b_mu, log_b_ls

    def rsample(
        self,
        feats: Tensor,
        mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Reparameterized sample of unit ``w`` and positive ``b``."""
        w_mu, w_ls, log_b_mu, log_b_ls = self.forward(feats, mask)
        w_eps = torch.randn_like(w_mu)
        w = unit(w_mu + w_eps * w_ls.exp())
        log_b = log_b_mu + torch.randn_like(log_b_mu) * log_b_ls.exp()
        b = log_b.exp()
        return w, b

    def kl_to_standard_normal(
        self,
        feats: Tensor,
        mask: Tensor | None = None,
    ) -> Tensor:
        """KL of the factorized Gaussian (pre-normalization for w) to N(0,I).

        Applied to unconstrained ``w_mu`` / ``log b`` Gaussians — standard VAE regularizer.
        """
        w_mu, w_ls, log_b_mu, log_b_ls = self.forward(feats, mask)
        # KL for diagonal Gaussian vs N(0,1): 0.5 * sum(mu^2 + sig^2 - 1 - 2 log sig)
        kl_w = 0.5 * ((w_mu**2 + w_ls.exp().pow(2) - 1.0 - 2.0 * w_ls).sum(dim=-1))
        kl_b = 0.5 * (log_b_mu**2 + log_b_ls.exp().pow(2) - 1.0 - 2.0 * log_b_ls)
        return kl_w + kl_b


class GlobalConsistencyEncoder(nn.Module):
    """VPL-style baseline: encodes ``w`` only; ``b`` is one global learned constant.

    Used in experiment 1 so the comparison isolates the factored consistency head.
    """

    def __init__(self, d: int, hidden: int = 64, init_b: float = 3.0):
        super().__init__()
        self.d = d
        in_dim = 2 * d + 2
        self.embed = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.w_mu = nn.Linear(hidden, d)
        self.w_log_std = nn.Linear(hidden, d)
        self.log_b = nn.Parameter(torch.tensor(math.log(init_b)))

    def forward(self, feats: Tensor, mask: Tensor | None = None) -> tuple[Tensor, Tensor, Tensor]:
        h = self.embed(feats)
        if mask is None:
            pooled = h.mean(dim=1)
        else:
            mask_f = mask.unsqueeze(-1).to(h.dtype)
            pooled = (h * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp_min(1.0)
        w_mu = self.w_mu(pooled)
        w_ls = self.w_log_std(pooled).clamp(-5.0, 2.0)
        b = self.log_b.exp().expand(feats.shape[0])
        return w_mu, w_ls, b

    def rsample(self, feats: Tensor, mask: Tensor | None = None) -> tuple[Tensor, Tensor]:
        w_mu, w_ls, b = self.forward(feats, mask)
        w = unit(w_mu + torch.randn_like(w_mu) * w_ls.exp())
        return w, b
