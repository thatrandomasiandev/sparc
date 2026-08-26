"""ApplyBounding and minibatch policy gate (DESIGN.md pseudocode)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto

from sparc.policy.config import BoundingConfig, BoundingMode


class UpdateAction(Enum):
    APPLY = auto()
    SKIP_TRANSITION = auto()
    SKIP_MINIBATCH = auto()


@dataclass(frozen=True)
class BoundingResult:
    """Outcome of ApplyBounding for one transition."""

    action: UpdateAction
    reward: float

    @property
    def skip_transition(self) -> bool:
        return self.action == UpdateAction.SKIP_TRANSITION


def apply_bounding(
    r_hat: float,
    disagreement: float,
    tau: float,
    mode: BoundingMode,
) -> BoundingResult:
    """
    Gate reward signal by ensemble disagreement vs scalar threshold tau_t.

    Returns SKIP_TRANSITION for freeze mode when D > tau.
    """
    if disagreement <= tau:
        return BoundingResult(UpdateAction.APPLY, r_hat)

    if mode == BoundingMode.PASS_THROUGH:
        return BoundingResult(UpdateAction.APPLY, r_hat)
    if mode == BoundingMode.SHRINK:
        alpha = tau / disagreement
        return BoundingResult(UpdateAction.APPLY, alpha * r_hat)
    if mode == BoundingMode.FREEZE:
        return BoundingResult(UpdateAction.SKIP_TRANSITION, r_hat)

    raise ValueError(f"unknown mode: {mode}")


def apply_minibatch_bounding(
    rewards: list[float],
    disagreements: list[float],
    tau: float,
    config: BoundingConfig,
) -> tuple[list[BoundingResult], bool]:
    """
    Apply bounding per transition; skip entire minibatch if gated fraction > f_max.

    Returns (per-transition results, skip_whole_minibatch).
    """
    if len(rewards) != len(disagreements):
        raise ValueError("rewards and disagreements must have same length")
    if not rewards:
        return [], False

    gated = sum(1 for d in disagreements if d > tau)
    if gated / len(disagreements) > config.f_max:
        return [
            BoundingResult(UpdateAction.SKIP_MINIBATCH, r) for r in rewards
        ], True

    results = [
        apply_bounding(r, d, tau, config.mode) for r, d in zip(rewards, disagreements, strict=True)
    ]
    return results, False


def effective_rewards(results: list[BoundingResult]) -> list[float | None]:
    """Rewards for policy update; None marks skipped transitions."""
    out: list[float | None] = []
    for res in results:
        if res.action == UpdateAction.SKIP_TRANSITION:
            out.append(None)
        elif res.action == UpdateAction.SKIP_MINIBATCH:
            out.append(None)
        else:
            out.append(res.reward)
    return out
