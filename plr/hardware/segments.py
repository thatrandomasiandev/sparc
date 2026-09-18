"""Trajectory segments → feature vectors for preference queries.

On the gridworld, ``phi`` is terrain. On a robot, ``phi`` is a fixed featurization of
recorded states (end-effector pose stats, object distances, speeds, …). Keep it linear
and low-d so the same acquisition stack applies; do not sneak in a deep reward net.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import torch
from torch import Tensor


@dataclass
class Segment:
    """One short behavior clip shown to a human."""

    segment_id: str
    states: Tensor  # (T, s_dim) raw recorded state
    phi: Tensor  # (d,) discounted / pooled features used by BT/PL
    meta: dict = field(default_factory=dict)


def featurize_states(
    states: Tensor,
    *,
    gamma: float = 0.95,
    feature_fn: str = "mean_stack",
) -> Tensor:
    """Map a state sequence to a single feature vector ``f``.

    ``mean_stack``: concatenate mean and final state, then caller may project.
    Replace with a platform-specific linear map once the robot/task is locked.
    """
    # states: (T, s)
    if states.dim() != 2:
        raise ValueError(f"states must be (T, s), got {tuple(states.shape)}")
    T = states.shape[0]
    if feature_fn == "mean_stack":
        mean = states.mean(dim=0)
        last = states[-1]
        return torch.cat([mean, last], dim=0)
    if feature_fn == "discounted_sum":
        # f = sum_t gamma^t s_t
        weights = gamma ** torch.arange(T, dtype=states.dtype, device=states.device)
        return (weights.unsqueeze(-1) * states).sum(dim=0)
    raise ValueError(f"unknown feature_fn {feature_fn!r}")


@dataclass
class SegmentLibrary:
    """Pool of candidate segments for acquisition (analog of exp3 candidates)."""

    segments: list[Segment]

    def phi_stack(self) -> Tensor:
        """Return ``(N, d)`` feature matrix."""
        if not self.segments:
            raise ValueError("empty SegmentLibrary")
        return torch.stack([s.phi for s in self.segments], dim=0)

    def pair_features(self, i: int, j: int) -> Tensor:
        """Binary query features ``(2, d)`` for acquisition / likelihood."""
        return torch.stack([self.segments[i].phi, self.segments[j].phi], dim=0)

    @classmethod
    def from_state_files(
        cls,
        paths: list[Path],
        *,
        gamma: float = 0.95,
        feature_fn: str = "discounted_sum",
        project: Tensor | None = None,
    ) -> SegmentLibrary:
        """Load ``.pt`` tensors of shape ``(T, s)``; optional linear ``project`` (s→d)."""
        segs: list[Segment] = []
        for p in paths:
            states = torch.load(p, map_location="cpu", weights_only=True)
            if not isinstance(states, Tensor):
                raise TypeError(f"{p} must contain a Tensor, got {type(states)}")
            phi = featurize_states(states, gamma=gamma, feature_fn=feature_fn)
            if project is not None:
                # project: (d, s_feat)
                phi = project @ phi
            segs.append(Segment(segment_id=p.stem, states=states, phi=phi, meta={"path": str(p)}))
        return cls(segments=segs)
