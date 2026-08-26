"""Per-operator adapter head and tensor batching helpers."""

from __future__ import annotations

from typing import cast

import torch
from torch import nn

from sparc.env.segments import TrajectorySegment


class EnsembleHead(nn.Module):
    """Bootstrap head m: w_i = A @ z_i + mu_w, g_i = w_i · φ + b_i."""

    def __init__(
        self,
        embed_dim: int,
        latent_dim: int,
        operator_ids: list[int],
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.latent_dim = latent_dim
        self.operator_ids = operator_ids
        self.A = nn.Parameter(torch.randn(embed_dim, latent_dim) * 0.01)
        self.mu_w = nn.Parameter(torch.zeros(embed_dim))
        self.biases = nn.ParameterDict(
            {str(op_id): nn.Parameter(torch.zeros(1)) for op_id in operator_ids}
        )

    def weight(self, z_i: torch.Tensor) -> torch.Tensor:
        """z_i: (latent_dim,) detached during M-step."""
        return self.A @ z_i + self.mu_w

    def utility(self, phi: torch.Tensor, operator_id: int, z_i: torch.Tensor) -> torch.Tensor:
        """
        phi: (B, embed_dim)
        returns: (B,)
        """
        w = self.weight(z_i)
        b = self.biases[str(operator_id)]
        return cast(torch.Tensor, phi @ w + b.squeeze())


def segments_to_tensors(
    segments: list[TrajectorySegment],
) -> tuple[torch.Tensor, torch.Tensor]:
    states = torch.stack([torch.from_numpy(s.states).float() for s in segments])
    actions = torch.stack([torch.from_numpy(s.actions).float() for s in segments])
    return states, actions
