"""Acquisition functions: BALD, EPIG, VOI, and effort normalization.

BALD subtracts mean per-particle entropy so the trivial query (identical options) scores
exactly zero — the pathology that sank volume-removal selection (Sadigh et al. 2017).
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import torch
from torch import Tensor
from torch.nn.functional import softmax

from plr.likelihood import plackett_luce_logp, reward_gap


@dataclass(frozen=True)
class QueryCosts:
    """Measured seconds per modality. Missing keys raise — never default (invariant I3)."""

    seconds: dict[str, float]

    def cost(self, modality: str) -> float:
        try:
            return self.seconds[modality]
        except KeyError as exc:
            raise KeyError(
                f"No measured cost for modality {modality!r}. "
                "Run a pilot and add it; do not invent a default."
            ) from exc


def _all_rankings(K: int) -> Tensor:
    """Enumerate all K! permutations as a ``(K!, K)`` long tensor."""
    return torch.tensor(list(itertools.permutations(range(K))), dtype=torch.long)


def predictive(
    phi: Tensor,
    particles_w: Tensor,
    particles_b: Tensor,
    weights: Tensor | None = None,
) -> Tensor:
    """Posterior-predictive distribution over all K! rankings.

    Enumerates outcomes (K stays small) and renormalizes in log space via softmax for
    stability at large ``b``.

    Parameters
    ----------
    phi : (K, d)
    particles_w : (P, d)
    particles_b : (P,)
    weights : (P,) optional, sum to 1

    Returns
    -------
    probs : (K!,) predictive probabilities
    """
    K = phi.shape[-2]
    orders = _all_rankings(K).to(phi.device)  # (M, K)
    M = orders.shape[0]
    P = particles_w.shape[0]
    # Broadcast: phi (P, K, d), order (P, M, K) — compute logp per particle per ranking
    phi_b = phi.unsqueeze(0).expand(P, -1, -1)  # (P, K, d)
    logps = []
    for m in range(M):
        order_m = orders[m].unsqueeze(0).expand(P, -1)  # (P, K)
        logps.append(plackett_luce_logp(phi_b, particles_w, particles_b, order_m))
    logp = torch.stack(logps, dim=-1)  # (P, M)
    if weights is None:
        weights = torch.full((P,), 1.0 / P, dtype=phi.dtype, device=phi.device)
    # log E_w[p(y|w)] = logsumexp(log w + log p) 
    log_mix = torch.logsumexp(torch.log(weights.clamp_min(1e-12)).unsqueeze(-1) + logp, dim=0)
    return softmax(log_mix, dim=0)  # (M,)


def entropy(probs: Tensor, dim: int = -1) -> Tensor:
    p = probs.clamp_min(1e-12)
    return -(p * p.log()).sum(dim=dim)


def bald_score(
    phi: Tensor,
    particles_w: Tensor,
    particles_b: Tensor,
    weights: Tensor | None = None,
) -> Tensor:
    """BALD: ``H[E_w p(y|w)] - E_w H[p(y|w)]``.

    The second term subtracts irreducible answer noise so identical options score ~0.
    """
    K = phi.shape[-2]
    orders = _all_rankings(K).to(phi.device)
    M = orders.shape[0]
    P = particles_w.shape[0]
    if weights is None:
        weights = torch.full((P,), 1.0 / P, dtype=phi.dtype, device=phi.device)

    phi_b = phi.unsqueeze(0).expand(P, -1, -1)
    logps = []
    for m in range(M):
        order_m = orders[m].unsqueeze(0).expand(P, -1)
        logps.append(plackett_luce_logp(phi_b, particles_w, particles_b, order_m))
    logp = torch.stack(logps, dim=-1)  # (P, M)
    p_w = softmax(logp, dim=-1)  # (P, M)
    # Marginal predictive
    p_bar = (weights.unsqueeze(-1) * p_w).sum(dim=0)  # (M,)
    h_bar = entropy(p_bar)
    h_each = entropy(p_w, dim=-1)  # (P,)
    return h_bar - (weights * h_each).sum()


def epig_score(
    phi_q: Tensor,
    phi_targets: Tensor,
    particles_w: Tensor,
    particles_b: Tensor,
    weights: Tensor | None = None,
) -> Tensor:
    """EPIG: expected mutual information between answers on ``q`` and target queries.

    Uses conditional independence of answers given the latent person.
    ``phi_targets``: (T, K, d).
    """
    if weights is None:
        P = particles_w.shape[0]
        weights = torch.full((P,), 1.0 / P, dtype=phi_q.dtype, device=phi_q.device)

    # For binary queries EPIG reduces to a cheap 2x2 mutual information average.
    # General K: use predictive joints via particle mixtures.
    K = phi_q.shape[-2]
    orders = _all_rankings(K).to(phi_q.device)
    M = orders.shape[0]
    P = particles_w.shape[0]

    def particle_probs(phi: Tensor) -> Tensor:
        phi_b = phi.unsqueeze(0).expand(P, -1, -1)
        cols = []
        for m in range(M):
            order_m = orders[m].unsqueeze(0).expand(P, -1)
            cols.append(plackett_luce_logp(phi_b, particles_w, particles_b, order_m))
        return softmax(torch.stack(cols, dim=-1), dim=-1)  # (P, M)

    p_q = particle_probs(phi_q)  # (P, M)
    mi_sum = torch.zeros((), dtype=phi_q.dtype, device=phi_q.device)
    T = phi_targets.shape[0]
    for t in range(T):
        p_t = particle_probs(phi_targets[t])  # (P, M)
        # Joint under mixture: E_w[p_q(a) p_t(c)]
        # joint[a,c] = sum_w w_w * p_q[w,a] * p_t[w,c]
        joint = torch.einsum("p,pa,pc->ac", weights, p_q, p_t)  # (M, M)
        p_a = joint.sum(dim=1)
        p_c = joint.sum(dim=0)
        # I = sum joint log joint/(pa pc)
        ratio = joint / (p_a.unsqueeze(1) * p_c.unsqueeze(0)).clamp_min(1e-12)
        mi = (joint.clamp_min(0) * torch.log(ratio.clamp_min(1e-12))).sum()
        mi_sum = mi_sum + mi
    return mi_sum / max(T, 1)


def effort_normalized(score: Tensor, costs: QueryCosts, modality: str) -> Tensor:
    """``score / measured_seconds`` — costs never assumed."""
    return score / costs.cost(modality)


def difficulty_scores(phi: Tensor, w_ref: Tensor) -> Tensor:
    """Batch difficulty via ``reward_gap`` under a reference (unit) direction — no ``b``."""
    # phi: (Q, K, d); w_ref: (d,)
    return reward_gap(phi, w_ref)
