"""Expected information gain scoring (DESIGN.md Pillar 2)."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import cast

import torch

from sparc.env.segments import TrajectorySegment
from sparc.query_optimizer.config import QueryOptimizerConfig
from sparc.reward_model.ensemble import RewardEnsemble


@dataclass
class QueryCandidate:
    """Candidate preference pair from frozen policy snapshot."""

    segment_0: TrajectorySegment
    segment_1: TrajectorySegment
    phi0: torch.Tensor | None = None
    phi1: torch.Tensor | None = None

    def ensure_embeddings(self, ensemble: RewardEnsemble) -> None:
        if self.phi0 is None:
            self.phi0 = ensemble.encode(self.segment_0)
        if self.phi1 is None:
            self.phi1 = ensemble.encode(self.segment_1)

    @property
    def diff(self) -> torch.Tensor:
        if self.phi0 is None or self.phi1 is None:
            raise RuntimeError("call ensure_embeddings before diff")
        return self.phi0 - self.phi1


def bernoulli_entropy(p: float) -> float:
    p = min(max(p, 1e-8), 1.0 - 1e-8)
    return float(-(p * math.log(p) + (1.0 - p) * math.log(1.0 - p)))


def _newton_step_w(
    w0: torch.Tensor,
    d: torch.Tensor,
    y: int,
    prior_sigma: float,
) -> torch.Tensor:
    """One Newton step on MAP for w given hypothetical Bradley–Terry label."""
    wd = torch.dot(w0, d)
    p = torch.sigmoid(wd)
    if y == 0:
        grad = -d * (1.0 - p)
    else:
        grad = d * p
    grad = grad + w0 / (prior_sigma**2)
    h = p * (1.0 - p)
    dim = d.shape[0]
    H = h * torch.outer(d, d) + torch.eye(dim, device=d.device, dtype=d.dtype) / (prior_sigma**2)
    step = torch.linalg.solve(H, grad)
    return cast(torch.Tensor, w0 - step)


def mean_pref_prob(
    candidate: QueryCandidate,
    operator_id: int,
    ensemble: RewardEnsemble,
    *,
    hypothetical_y: int | None = None,
    config: QueryOptimizerConfig | None = None,
) -> float:
    """
    p(y=0 | q, D) ≈ mean_m σ_BT(Δ_i^{(m)}).

    If hypothetical_y is set, apply one Newton step on each head's w_i (DESIGN.md note).
    """
    cfg = config or QueryOptimizerConfig()
    candidate.ensure_embeddings(ensemble)
    d = candidate.diff
    z_i = ensemble.z[operator_id]
    probs: list[float] = []
    for head in ensemble.heads:
        w0 = head.weight(z_i).detach()
        w = (
            _newton_step_w(w0, d, hypothetical_y, cfg.newton_prior_sigma)
            if hypothetical_y is not None
            else w0
        )
        probs.append(float(torch.sigmoid(torch.dot(w, d)).item()))
    return sum(probs) / len(probs)


def score_eig(
    candidate: QueryCandidate,
    operator_id: int,
    ensemble: RewardEnsemble,
    config: QueryOptimizerConfig | None = None,
) -> float:
    """EIG(q, i) via two-outcome Monte Carlo (exact for binary preferences)."""
    cfg = config or QueryOptimizerConfig()
    p0 = mean_pref_prob(candidate, operator_id, ensemble, config=cfg)
    h_prior = bernoulli_entropy(p0)
    h_post = 0.0
    for y in (0, 1):
        py = p0 if y == 0 else (1.0 - p0)
        p0_post = mean_pref_prob(
            candidate,
            operator_id,
            ensemble,
            hypothetical_y=y,
            config=cfg,
        )
        h_post += py * bernoulli_entropy(p0_post)
    return h_prior - h_post
