"""POP-VOI: Population-prior Value-of-Information preference elicitation.

This is the paper algorithm. It chooses the next trajectory-segment comparison by
*expected reduction in policy regret* under a particle prior drawn from previous
users — not by information gain about reward parameters (BALD).

Formal objective (myopic)::

    Loss(β)  = E_{w ~ β}[ V*_w - V^{π(decision(β))}_w ]
    score(q) = Loss(β) - Σ_y P_β(y | q) Loss(β | q, y)

``decision(β)`` defaults to the optimal policy for the posterior-mean reward
(unit-normalized). Particles come from a population prior so we never search all
of reward space (the move that makes Regan & Boutilier–style VOI affordable here).

Novelty boundary: the *criterion* is classical; this module is the instantiation
for human pairwise/ranking comparisons + population particles + exact tabular regret.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import torch
from torch import Tensor
from torch.nn.functional import softmax

from plr.acquisition import bald_score
from plr.likelihood import plackett_luce_logp, unit
from plr.mdp import (
    Belief,
    Gridworld,
    oracle_true_regret_score,
    posterior_mean_decision,
    voi_score,
)


class Acquisition(str, Enum):
    """How to score candidate queries."""

    VOI = "voi"  # decision-relevant (ours)
    VOI_MEAN = "voi_mean"  # VOI with mean-decision loss (legacy)
    BALD = "bald"  # parameter information gain (baseline)
    RANDOM = "random"
    ORACLE = "oracle"  # true-regret ceiling — evaluation only


@dataclass
class StepLog:
    """One elicitation step — enough to regenerate paper curves from a run."""

    t: int
    query_index: int
    phi: Tensor
    order: Tensor
    acquisition: str
    score: float
    loss_before: float
    true_regret_after: float | None = None


@dataclass
class POPVOI:
    """Online preference elicitation under a fixed population particle prior.

    Parameters
    ----------
    env:
        Tabular MDP with linear rewards (exact VI / exact regret).
    particles_w, particles_b:
        Population prior particles ``(P, d)`` and ``(P,)``. Fixed for the session;
        only the categorical weights change (I6-friendly, cache-friendly).
    candidates:
        Query pool ``(Q, K, d)`` of segment-feature pairs (K=2 binary by default).
    acquisition:
        Default ``VOI`` — the decision-relevant score.
    loss_mode:
        ``particle`` (default) or ``mean`` — matches ``voi_score`` / Belief.loss.
    seed:
        RNG for RANDOM acquisition and any stochastic tie-breaking.
    """

    env: Gridworld
    particles_w: Tensor
    particles_b: Tensor
    candidates: Tensor
    acquisition: Acquisition = Acquisition.VOI
    loss_mode: str = "particle"
    seed: int = 0
    belief: Belief = field(init=False)
    log: list[StepLog] = field(default_factory=list)
    _generator: torch.Generator = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.particles_w.dim() != 2:
            raise ValueError("particles_w must be (P, d)")
        if self.particles_b.shape[0] != self.particles_w.shape[0]:
            raise ValueError("particles_b must align with particles_w")
        if self.candidates.dim() != 3:
            raise ValueError("candidates must be (Q, K, d)")
        if self.candidates.shape[-1] != self.particles_w.shape[-1]:
            raise ValueError("candidate feature dim must match particles")
        if self.loss_mode not in ("particle", "mean"):
            raise ValueError("loss_mode must be 'particle' or 'mean'")

        self.belief = Belief.create(self.env, self.particles_w, self.particles_b)
        self._generator = torch.Generator()
        self._generator.manual_seed(self.seed)
        self.log = []

    # ------------------------------------------------------------------ API

    def reset(self) -> None:
        """Restore uniform weights over the population prior; clear the log."""
        P = self.particles_w.shape[0]
        self.belief = self.belief.with_weights(
            torch.full((P,), 1.0 / P, dtype=self.particles_w.dtype)
        )
        self.log = []

    def score_query(
        self,
        q: int,
        *,
        w_true: Tensor | None = None,
        b_true: Tensor | None = None,
    ) -> float:
        """Score candidate index ``q`` under the current belief."""
        phi = self.candidates[q]
        acq = self.acquisition
        if acq is Acquisition.VOI:
            return float(voi_score(self.belief, phi, loss_mode="particle"))
        if acq is Acquisition.VOI_MEAN:
            return float(voi_score(self.belief, phi, loss_mode="mean"))
        if acq is Acquisition.BALD:
            return float(
                bald_score(
                    phi,
                    self.belief.particles_w,
                    self.belief.particles_b,
                    self.belief.weights,
                )
            )
        if acq is Acquisition.ORACLE:
            if w_true is None or b_true is None:
                raise ValueError("ORACLE acquisition requires w_true and b_true")
            return float(oracle_true_regret_score(self.belief, phi, w_true, b_true))
        if acq is Acquisition.RANDOM:
            return float(torch.rand((), generator=self._generator).item())
        raise ValueError(acq)

    def select_query(
        self,
        *,
        w_true: Tensor | None = None,
        b_true: Tensor | None = None,
    ) -> tuple[int, float]:
        """Return ``(query_index, score)`` maximizing the acquisition score.

        Algorithm step (paper)::

            q* ← argmax_{q ∈ Q} score(q ; β_t)
        """
        Q = self.candidates.shape[0]
        if self.acquisition is Acquisition.RANDOM:
            idx = int(torch.randint(0, Q, (1,), generator=self._generator).item())
            return idx, 0.0

        scores = [
            self.score_query(q, w_true=w_true, b_true=b_true) for q in range(Q)
        ]
        idx = int(torch.tensor(scores).argmax().item())
        return idx, scores[idx]

    def update(self, query_index: int, order: Tensor) -> Belief:
        """Bayes update of particle weights given observed ranking ``order``.

        Algorithm step (paper)::

            β_{t+1}(w) ∝ β_t(w) P(order | q_t, w, b)
        """
        if order.dim() != 1:
            raise ValueError("order must be a 1-D ranking of option indices")
        phi = self.candidates[query_index]
        P = self.belief.particles_w.shape[0]
        logp = plackett_luce_logp(
            phi.unsqueeze(0).expand(P, -1, -1),
            self.belief.particles_w,
            self.belief.particles_b,
            order.unsqueeze(0).expand(P, -1),
        )
        new_w = softmax(
            torch.log(self.belief.weights.clamp_min(1e-12)) + logp, dim=0
        )
        self.belief = self.belief.with_weights(new_w)
        return self.belief

    def decide_w(self) -> Tensor:
        """Point estimate used for acting: unit posterior-mean reward direction."""
        return posterior_mean_decision(self.belief.weights, self.belief.particles_w)

    def decide_policy(self) -> Tensor:
        """Optimal policy for ``decide_w()`` on ``env``."""
        _, pi = self.env.value_iteration(self.decide_w())
        return pi

    def true_regret(self, w_true: Tensor) -> Tensor:
        """Policy regret of current decision under ground-truth ``w_true``."""
        return self.env.regret(w_true, self.decide_w())

    def step(
        self,
        order: Tensor,
        *,
        w_true: Tensor | None = None,
        b_true: Tensor | None = None,
    ) -> StepLog:
        """One full elicitation step: select → update → (optional) measure regret.

        ``order`` is the human (or simulated) ranking for the *selected* query,
        best-first. Callers that need to show the query to a human should use
        ``select_query`` then ``update`` separately instead.
        """
        loss_before = float(self.belief.loss(self.loss_mode))
        q, score = self.select_query(w_true=w_true, b_true=b_true)
        phi = self.candidates[q].clone()
        self.update(q, order)
        reg = None
        if w_true is not None:
            reg = float(self.true_regret(w_true))
        entry = StepLog(
            t=len(self.log),
            query_index=q,
            phi=phi,
            order=order.clone(),
            acquisition=self.acquisition.value,
            score=score,
            loss_before=loss_before,
            true_regret_after=reg,
        )
        self.log.append(entry)
        return entry

    def run(
        self,
        answer_fn,
        n_queries: int,
        *,
        w_true: Tensor | None = None,
        b_true: Tensor | None = None,
    ) -> list[StepLog]:
        """Run ``n_queries`` steps.

        ``answer_fn(phi) -> order`` maps a ``(K, d)`` query to a ranking tensor.
        For simulated users, wrap ``plr.users.answer``.
        """
        self.reset()
        for _ in range(n_queries):
            loss_before = float(self.belief.loss(self.loss_mode))
            q, score = self.select_query(w_true=w_true, b_true=b_true)
            phi = self.candidates[q]
            order = answer_fn(phi)
            if not isinstance(order, Tensor):
                order = torch.as_tensor(order, dtype=torch.long)
            self.update(q, order)
            reg = float(self.true_regret(w_true)) if w_true is not None else None
            self.log.append(
                StepLog(
                    t=len(self.log),
                    query_index=q,
                    phi=phi.clone(),
                    order=order.clone(),
                    acquisition=self.acquisition.value,
                    score=float(score),
                    loss_before=loss_before,
                    true_regret_after=reg,
                )
            )
        return list(self.log)


def build_population_prior(
    n_particles: int,
    d: int,
    mode_centers: Tensor,
    *,
    consistency_spread: float = 0.3,
    b_median: float = 3.0,
    seed: int = 0,
) -> tuple[Tensor, Tensor]:
    """Sample ``(particles_w, particles_b)`` from a mixture population prior."""
    from plr.users import sample_gaussian_mixture_population

    pop = sample_gaussian_mixture_population(
        n_particles,
        d,
        mode_centers=mode_centers,
        consistency_spread=consistency_spread,
        b_median=b_median,
        seed=seed,
    )
    return pop.w, pop.b


def default_mode_centers(d: int, seed: int = 0) -> Tensor:
    """Three canonical preference modes in ``R^d`` (used across experiments)."""
    g = torch.Generator().manual_seed(seed)
    centers = unit(
        torch.tensor(
            [
                [1.0] + [0.0] * (d - 1),
                [0.0, 1.0] + [0.0] * (d - 2),
                [1.0, 1.0] + [0.0] * (d - 2),
            ],
            dtype=torch.float32,
        )
    )
    if d > 3:
        centers = unit(centers + 0.1 * torch.randn(3, d, generator=g))
    return centers
