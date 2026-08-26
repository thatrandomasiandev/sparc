"""Shared trajectory encoder ψ."""

from __future__ import annotations

from typing import cast

import torch
from torch import nn


class TrajectoryEncoder(nn.Module):
    """Maps (states, actions) segment tensors to embedding φ of dim `embed_dim`."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        horizon: int,
        embed_dim: int = 128,
        hidden_dim: int = 256,
    ) -> None:
        super().__init__()
        input_dim = horizon * (state_dim + action_dim)
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim),
        )
        self.horizon = horizon
        self.state_dim = state_dim
        self.action_dim = action_dim

    def forward(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """
        states: (B, H, state_dim)
        actions: (B, H, action_dim)
        returns: (B, embed_dim)
        """
        if states.dim() != 3 or actions.dim() != 3:
            raise ValueError("states and actions must be 3-D batch tensors")
        batch = torch.cat([states, actions], dim=-1)
        flat = batch.reshape(batch.size(0), -1)
        return cast(torch.Tensor, self.net(flat))
