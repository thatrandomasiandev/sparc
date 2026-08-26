"""B-Pref-style ensemble reward MLP for PEBBLE / PrefPPO."""

from __future__ import annotations

from typing import cast

import numpy as np
import torch
from torch import nn


def _mlp(
    in_dim: int,
    hidden: int = 256,
    layers: int = 3,
    activation: str = "tanh",
) -> nn.Sequential:
    act: type[nn.Module]
    if activation == "tanh":
        act = nn.Tanh
    elif activation == "sig":
        act = nn.Sigmoid
    else:
        act = nn.ReLU

    modules: list[nn.Module] = []
    dim = in_dim
    for _ in range(layers):
        modules.append(nn.Linear(dim, hidden))
        modules.append(act())
        dim = hidden
    modules.append(nn.Linear(dim, 1))
    return nn.Sequential(*modules)


class RewardMLP(nn.Module):
    """Per-timestep reward; sums over segment dimension for preferences."""

    def __init__(self, sa_dim: int, hidden: int = 256, activation: str = "tanh") -> None:
        super().__init__()
        self.net = _mlp(sa_dim, hidden=hidden, activation=activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            return cast(torch.Tensor, self.net(x).squeeze(-1))
        if x.dim() == 2:
            return cast(torch.Tensor, self.net(x).squeeze(-1))
        if x.dim() == 3:
            batch, seg, dim = x.shape
            flat = x.reshape(batch * seg, dim)
            out = self.net(flat).reshape(batch, seg)
            return cast(torch.Tensor, out)
        raise ValueError(f"expected 1-3D input, got shape {tuple(x.shape)}")


def segment_reward(model: RewardMLP, sa_segment: np.ndarray) -> float:
    """Scalar segment return under one ensemble member."""
    with torch.no_grad():
        x = torch.from_numpy(sa_segment).float()
        r = model(x)
        return float(r.sum().item() if r.dim() > 0 else r.item())


def ensemble_disagreement(members: list[RewardMLP], sa1: np.ndarray, sa2: np.ndarray) -> float:
    """Std dev of P(seg1 > seg2) across ensemble (B-Pref disagreement sampling)."""
    probs: list[float] = []
    for member in members:
        with torch.no_grad():
            x1 = torch.from_numpy(sa1).float()
            x2 = torch.from_numpy(sa2).float()
            r1 = member(x1).sum()
            r2 = member(x2).sum()
            stacked = torch.stack([r1, r2])
            prob = torch.softmax(stacked, dim=0)[0].item()
            probs.append(prob)
    return float(np.std(probs))
