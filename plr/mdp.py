"""Tabular gridworld with exact value iteration and exact policy regret.

Key trick: reward is linear in ``w``, so the value of a *fixed* policy is too:
``V^π(w) = (I - γ P^π)^{-1} Φ w``. Invert once per policy, then score thousands of
candidate rewards with one matmul. Only ``V*(w)`` needs value iteration; particle
optimal values can be precomputed when particles are fixed.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import Tensor
from torch.nn.functional import softmax

from plr.likelihood import unit


@dataclass
class Gridworld:
    """Deterministic 4-action grid with absorbing goal and linear terrain features."""

    n_rows: int
    n_cols: int
    phi: Tensor  # (S, d) features per state
    gamma: float = 0.95
    goal: int | None = None  # absorbing state index; default = last cell
    walls: set[int] | None = None

    def __post_init__(self) -> None:
        self.S = self.n_rows * self.n_cols
        if self.phi.shape[0] != self.S:
            raise ValueError("phi must have one row per state")
        self.d = self.phi.shape[1]
        self.goal = self.S - 1 if self.goal is None else self.goal
        self.walls = set() if self.walls is None else set(self.walls)
        self.n_actions = 4  # N,E,S,W
        self.P = self._build_transitions()  # (A, S, S)

    def _idx(self, r: int, c: int) -> int:
        return r * self.n_cols + c

    def _build_transitions(self) -> Tensor:
        # P[a, s, s']
        P = torch.zeros(self.n_actions, self.S, self.S)
        deltas = [(-1, 0), (0, 1), (1, 0), (0, -1)]
        for s in range(self.S):
            if s == self.goal or s in self.walls:
                P[:, s, s] = 1.0
                continue
            r, c = divmod(s, self.n_cols)
            for a, (dr, dc) in enumerate(deltas):
                nr, nc = r + dr, c + dc
                if not (0 <= nr < self.n_rows and 0 <= nc < self.n_cols):
                    ns = s
                else:
                    ns = self._idx(nr, nc)
                    if ns in self.walls:
                        ns = s
                P[a, s, ns] = 1.0
        return P

    def reward_vector(self, w: Tensor) -> Tensor:
        """``r(s) = unit(w) · phi(s)`` — (S,) or (P, S)."""
        w_u = unit(w)
        if w_u.dim() == 1:
            return self.phi @ w_u  # (S,)
        return w_u @ self.phi.T  # (P, S)

    def value_iteration(
        self, w: Tensor, tol: float = 1e-6, max_iter: int = 50_000
    ) -> tuple[Tensor, Tensor]:
        """Exact VI. Returns ``(V*, pi*)`` with ``pi*: (S,)`` greedy actions.

        Discounted finite MDPs always contract; we stop on successive-V gap or
        after ``max_iter``. Residual ~1e-7 is harmless for regret comparisons.
        """
        r = self.reward_vector(w)  # (S,)
        if not torch.isfinite(r).all():
            raise ValueError("non-finite rewards in value_iteration")
        V = torch.zeros(self.S, dtype=self.phi.dtype)
        pi = torch.zeros(self.S, dtype=torch.long)
        delta = torch.tensor(float("inf"))
        for _ in range(max_iter):
            Q = r.unsqueeze(0) + self.gamma * (self.P @ V)  # (A, S)
            V_new, pi = Q.max(dim=0)
            delta = torch.max(torch.abs(V_new - V))
            V = V_new
            if float(delta) < tol:
                break
        return V, pi

    def policy_transition(self, pi: Tensor) -> Tensor:
        """``P^π`` with shape ``(S, S)``."""
        P_pi = torch.zeros(self.S, self.S, dtype=self.phi.dtype)
        for s in range(self.S):
            P_pi[s] = self.P[int(pi[s]), s]
        return P_pi

    def policy_value_matrix(self, pi: Tensor) -> Tensor:
        """Return ``M`` such that ``V^π(w) = M @ unit(w)``.

        ``M = (I - γ P^π)^{-1} Φ``. Invert once; reuse across particles.
        """
        P_pi = self.policy_transition(pi)
        eye = torch.eye(self.S, dtype=self.phi.dtype)
        A = eye - self.gamma * P_pi
        return torch.linalg.solve(A, self.phi)  # (S, d)

    def policy_value(self, pi: Tensor, w: Tensor, M: Tensor | None = None) -> Tensor:
        """Mean start-state value under fixed ``pi`` for one or many ``w``."""
        if M is None:
            M = self.policy_value_matrix(pi)
        w_u = unit(w)
        starts = self._start_mask()
        if w_u.dim() == 1:
            V = M @ w_u  # (S,)
            return (V * starts).sum() / starts.sum()
        V = w_u @ M.T  # (P, S)
        return (V * starts.unsqueeze(0)).sum(dim=-1) / starts.sum()

    def _start_mask(self) -> Tensor:
        m = torch.ones(self.S, dtype=self.phi.dtype)
        m[self.goal] = 0.0
        for wall in self.walls:
            m[wall] = 0.0
        return m

    def optimal_values(self, particles_w: Tensor) -> Tensor:
        """``V*(w_p)`` mean start-state value for each particle — (P,)."""
        vals = []
        starts = self._start_mask()
        for p in range(particles_w.shape[0]):
            V, _ = self.value_iteration(particles_w[p])
            vals.append((V * starts).sum() / starts.sum())
        return torch.stack(vals)

    def regret(
        self,
        w_true: Tensor,
        w_decision: Tensor,
        V_star_true: float | Tensor | None = None,
    ) -> Tensor:
        """Policy regret of acting optimally for ``w_decision`` under true ``w_true``."""
        _, pi_hat = self.value_iteration(w_decision)
        if V_star_true is None:
            V_star, _ = self.value_iteration(w_true)
            starts = self._start_mask()
            v_star = (V_star * starts).sum() / starts.sum()
        else:
            v_star = torch.as_tensor(V_star_true, dtype=self.phi.dtype)
        v_pi = self.policy_value(pi_hat, w_true)
        return v_star - v_pi


def make_terrain_grid(
    n_rows: int = 6,
    n_cols: int = 6,
    d: int = 8,
    seed: int = 0,
    gamma: float = 0.95,
) -> Gridworld:
    """Build a grid whose cells have random non-negative terrain features."""
    g = torch.Generator().manual_seed(seed)
    S = n_rows * n_cols
    phi = torch.rand(S, d, generator=g)
    mask = torch.rand(S, d, generator=g) > 0.4
    phi = phi * mask.float()
    phi[-1] = 0.0
    return Gridworld(n_rows=n_rows, n_cols=n_cols, phi=phi, gamma=gamma)


def posterior_mean_decision(weights: Tensor, particles_w: Tensor) -> Tensor:
    """``decision(posterior) = unit(E[w])`` — mean reward direction."""
    mean = (weights.unsqueeze(-1) * particles_w).sum(dim=0)
    return unit(mean)


@dataclass
class Belief:
    """Particle belief with cached per-particle optimal policies.

    Caching is why population-prior VOI is affordable: particles are fixed, only
    weights change, so each particle's ``π*`` and ``M^π`` are computed once.
    """

    env: Gridworld
    particles_w: Tensor  # (P, d)
    particles_b: Tensor  # (P,)
    weights: Tensor  # (P,)
    V_star: Tensor  # (P,)
    policies: list[Tensor] = field(default_factory=list)  # P policies (S,)
    # policy fingerprint -> M matrix (S, d)
    _M_cache: dict[tuple[int, ...], Tensor] = field(default_factory=dict, repr=False)
    # (P, P) with V^{π_i}(w_j) mean start values — filled lazily
    _value_table: Tensor | None = field(default=None, repr=False)

    @classmethod
    def create(
        cls,
        env: Gridworld,
        particles_w: Tensor,
        particles_b: Tensor,
        weights: Tensor | None = None,
    ) -> Belief:
        P = particles_w.shape[0]
        if weights is None:
            weights = torch.full((P,), 1.0 / P, dtype=particles_w.dtype)
        policies = []
        V_star = []
        starts = env._start_mask()
        for p in range(P):
            V, pi = env.value_iteration(particles_w[p])
            policies.append(pi)
            V_star.append((V * starts).sum() / starts.sum())
        return cls(
            env=env,
            particles_w=particles_w,
            particles_b=particles_b,
            weights=weights,
            V_star=torch.stack(V_star),
            policies=policies,
        )

    def with_weights(self, weights: Tensor) -> Belief:
        """Cheap copy with new weights — keeps policy / M caches."""
        return Belief(
            env=self.env,
            particles_w=self.particles_w,
            particles_b=self.particles_b,
            weights=weights,
            V_star=self.V_star,
            policies=self.policies,
            _M_cache=self._M_cache,
            _value_table=self._value_table,
        )

    def _policy_key(self, pi: Tensor) -> tuple[int, ...]:
        return tuple(int(a) for a in pi.tolist())

    def M_for(self, pi: Tensor) -> Tensor:
        key = self._policy_key(pi)
        if key not in self._M_cache:
            self._M_cache[key] = self.env.policy_value_matrix(pi)
        return self._M_cache[key]

    def value_table(self) -> Tensor:
        """``V[i, j] =`` mean start value of ``π*(w_i)`` under reward ``w_j`` — (P, P)."""
        if self._value_table is not None:
            return self._value_table
        P = self.particles_w.shape[0]
        table = torch.empty(P, P, dtype=self.particles_w.dtype)
        for i in range(P):
            M = self.M_for(self.policies[i])
            # values under all particles for fixed π_i
            table[i] = self.env.policy_value(self.policies[i], self.particles_w, M=M)
        self._value_table = table
        return table

    def loss_mean(self) -> Tensor:
        """``E_w[V*_w - V^{π(mean)}_w]`` — original VOI loss (queue-1 baseline)."""
        decision = posterior_mean_decision(self.weights, self.particles_w)
        _, pi = self.env.value_iteration(decision)
        M = self.M_for(pi)
        v_pi = self.env.policy_value(pi, self.particles_w, M=M)  # (P,)
        return (self.weights * (self.V_star - v_pi)).sum()

    def loss_particle(self) -> Tensor:
        """``E_{i,j ~ post}[V*_j - V^{π_i}_j]`` — expected regret of a posterior policy draw.

        Captures policy disagreement the mean decision averages away. Queries that
        collapse distinct near-optimal policies score higher under this loss.
        """
        # table[i,j] = V^{π_i}(w_j); regret_ij = V*_j - table[i,j]
        table = self.value_table()  # (P, P)
        regret = self.V_star.unsqueeze(0) - table  # (P, P) broadcast V*_j on columns
        # E_i E_j = w^T regret w  (weights on rows and cols)
        return self.weights @ (regret @ self.weights)

    def loss(self, mode: str = "particle") -> Tensor:
        if mode == "mean":
            return self.loss_mean()
        if mode == "particle":
            return self.loss_particle()
        raise ValueError(f"unknown loss mode {mode!r}")


def expected_regret(
    env: Gridworld,
    weights: Tensor,
    particles_w: Tensor,
    V_star: Tensor,
    decision_w: Tensor | None = None,
) -> Tensor:
    """Backward-compatible mean-decision expected regret."""
    if decision_w is None:
        decision_w = posterior_mean_decision(weights, particles_w)
    _, pi = env.value_iteration(decision_w)
    M = env.policy_value_matrix(pi)
    v_pi = env.policy_value(pi, particles_w, M=M)
    return (weights * (V_star - v_pi)).sum()


def _answer_logprobs(
    phi_query: Tensor,
    particles_w: Tensor,
    particles_b: Tensor,
) -> tuple[Tensor, Tensor]:
    """Return ``(orders, logp)`` with ``logp: (P, M)`` over all K! rankings."""
    from plr.acquisition import _all_rankings
    from plr.likelihood import plackett_luce_logp

    K = phi_query.shape[-2]
    orders = _all_rankings(K).to(phi_query.device)
    M = orders.shape[0]
    P = particles_w.shape[0]
    phi_b = phi_query.unsqueeze(0).expand(P, -1, -1)
    cols = []
    for m in range(M):
        order_m = orders[m].unsqueeze(0).expand(P, -1)
        cols.append(plackett_luce_logp(phi_b, particles_w, particles_b, order_m))
    return orders, torch.stack(cols, dim=-1)  # (P, M)


def voi_score(
    belief: Belief,
    phi_query: Tensor,
    *,
    loss_mode: str = "particle",
    answer_probs: Tensor | None = None,
) -> Tensor:
    """Expected reduction in policy loss from asking ``phi_query``.

    Parameters
    ----------
    loss_mode:
        ``particle`` (default, queue-1 improvement) or ``mean`` (legacy).
    answer_probs:
        Optional ``(M,)`` distribution over rankings. Default: posterior predictive.
        Oracle passes the *true* user answer distribution here.
    """
    _, logp = _answer_logprobs(phi_query, belief.particles_w, belief.particles_b)
    # logp: (P, M)
    p_w = softmax(logp, dim=-1)
    if answer_probs is None:
        p_y = (belief.weights.unsqueeze(-1) * p_w).sum(dim=0)  # (M,)
    else:
        p_y = answer_probs

    loss_now = belief.loss(loss_mode)
    expected_after = torch.zeros((), dtype=belief.weights.dtype)
    for m in range(logp.shape[-1]):
        lw = torch.log(belief.weights.clamp_min(1e-12)) + logp[:, m]
        w_post = softmax(lw, dim=0)
        expected_after = expected_after + p_y[m] * belief.with_weights(w_post).loss(loss_mode)
    return loss_now - expected_after


def oracle_score(
    belief: Belief,
    phi_query: Tensor,
    w_true: Tensor,
    b_true: Tensor,
    *,
    loss_mode: str = "particle",
) -> Tensor:
    """Ceiling: same loss as VOI, but answer probs from the true user."""
    from plr.acquisition import _all_rankings
    from plr.likelihood import plackett_luce_logp

    orders = _all_rankings(phi_query.shape[-2]).to(phi_query.device)
    log_true = []
    b = b_true if b_true.dim() == 0 else b_true.reshape(())
    for m in range(orders.shape[0]):
        log_true.append(
            plackett_luce_logp(
                phi_query.unsqueeze(0),
                w_true.unsqueeze(0),
                b.unsqueeze(0),
                orders[m].unsqueeze(0),
            ).squeeze(0)
        )
    p_true = softmax(torch.stack(log_true), dim=0)
    return voi_score(belief, phi_query, loss_mode=loss_mode, answer_probs=p_true)


def oracle_true_regret_score(
    belief: Belief,
    phi_query: Tensor,
    w_true: Tensor,
    b_true: Tensor,
) -> Tensor:
    """Ceiling for final-query regret: expected drop in *true* mean-decision regret.

    Differs from ``oracle_score`` (posterior loss under true answer probs). This is the
    quantity our primary metric actually measures, so it is the right acquisition ceiling.
    """
    from plr.acquisition import _all_rankings
    from plr.likelihood import plackett_luce_logp

    def true_reg(weights: Tensor) -> Tensor:
        decision = posterior_mean_decision(weights, belief.particles_w)
        return belief.env.regret(w_true, decision)

    loss_now = true_reg(belief.weights)
    orders = _all_rankings(phi_query.shape[-2]).to(phi_query.device)
    b = b_true if b_true.dim() == 0 else b_true.reshape(())
    log_true = []
    for m in range(orders.shape[0]):
        log_true.append(
            plackett_luce_logp(
                phi_query.unsqueeze(0),
                w_true.unsqueeze(0),
                b.unsqueeze(0),
                orders[m].unsqueeze(0),
            ).squeeze(0)
        )
    p_true = softmax(torch.stack(log_true), dim=0)

    _, logp = _answer_logprobs(phi_query, belief.particles_w, belief.particles_b)
    expected_after = torch.zeros((), dtype=belief.weights.dtype)
    for m in range(logp.shape[-1]):
        lw = torch.log(belief.weights.clamp_min(1e-12)) + logp[:, m]
        w_post = softmax(lw, dim=0)
        expected_after = expected_after + p_true[m] * true_reg(w_post)
    return loss_now - expected_after


def voi_score_lookahead(
    belief: Belief,
    phi_query: Tensor,
    candidates: Tensor,
    *,
    loss_mode: str = "particle",
    shortlist: int = 8,
    exclude_index: int | None = None,
) -> Tensor:
    """One-step VOI plus expected best follow-up VOI (greedy depth-2).

    ``score(q) = VOI_1(q) + E_y[ max_{q'} VOI_1(q' | y) ]``.

    ``shortlist`` caps the inner max over follow-ups (top-1-step VOI under current belief)
    so depth-2 stays affordable.
    """
    _, logp = _answer_logprobs(phi_query, belief.particles_w, belief.particles_b)
    p_w = softmax(logp, dim=-1)
    p_y = (belief.weights.unsqueeze(-1) * p_w).sum(dim=0)

    # Shortlist follow-ups by current one-step VOI (exclude the query itself if indexed).
    follow_scores = []
    follow_idx = []
    for q in range(candidates.shape[0]):
        if exclude_index is not None and q == exclude_index:
            continue
        s = float(voi_score(belief, candidates[q], loss_mode=loss_mode))
        follow_scores.append(s)
        follow_idx.append(q)
    if not follow_idx:
        return voi_score(belief, phi_query, loss_mode=loss_mode)

    order = torch.tensor(follow_scores).argsort(descending=True)
    keep = [follow_idx[int(i)] for i in order[:shortlist].tolist()]

    voi1 = voi_score(belief, phi_query, loss_mode=loss_mode)
    bonus = torch.zeros((), dtype=belief.weights.dtype)
    for m in range(logp.shape[-1]):
        lw = torch.log(belief.weights.clamp_min(1e-12)) + logp[:, m]
        w_post = softmax(lw, dim=0)
        bel_y = belief.with_weights(w_post)
        best = max(
            float(voi_score(bel_y, candidates[q], loss_mode=loss_mode)) for q in keep
        )
        bonus = bonus + p_y[m] * best
    return voi1 + bonus
