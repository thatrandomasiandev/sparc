"""Bradley–Terry preference losses."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def bradley_terry_loss(labels: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
    """
    Logistic loss for pairwise preferences.

    labels: (B,) with 0 = prefer seg0, 1 = prefer seg1
    delta: (B,) utility difference g(seg0) - g(seg1)
    """
    sign = torch.where(labels == 0, torch.ones_like(delta), -torch.ones_like(delta))
    return F.softplus(-sign * delta).mean()


def preference_delta(u0: torch.Tensor, u1: torch.Tensor) -> torch.Tensor:
    return u0 - u1
