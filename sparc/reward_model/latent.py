"""MAP inference for operator latent z_i (E-step)."""

from __future__ import annotations

import torch
from torch.optim import Adam

from sparc.env.dataset import PreferenceRecord
from sparc.reward_model.adapter import EnsembleHead, segments_to_tensors
from sparc.reward_model.encoder import TrajectoryEncoder
from sparc.reward_model.losses import bradley_terry_loss, preference_delta


def infer_operator_latent(
    records: list[PreferenceRecord],
    operator_id: int,
    encoder: TrajectoryEncoder,
    head: EnsembleHead,
    *,
    latent_dim: int,
    sigma_z: float = 1.0,
    steps: int = 100,
    lr: float = 0.05,
    device: torch.device | None = None,
) -> torch.Tensor:
    """
    MAP estimate of z_i from labeled preferences (EM E-step).

    Uses fixed encoder and head.A / head.mu_w; optimizes z only.
    """
    if device is None:
        device = torch.device("cpu")

    labeled = [
        r
        for r in records
        if r.is_labeled and r.query.operator_id == operator_id
    ]
    z = torch.zeros(latent_dim, device=device, requires_grad=True)
    if not labeled:
        return z.detach()

    encoder.eval()
    head.eval()
    for param in encoder.parameters():
        param.requires_grad_(False)
    for param in head.parameters():
        param.requires_grad_(False)

    optimizer = Adam([z], lr=lr)
    seg0 = [r.query.segment_0 for r in labeled]
    seg1 = [r.query.segment_1 for r in labeled]
    labels = torch.tensor([r.label for r in labeled], dtype=torch.long, device=device)
    s0, a0 = segments_to_tensors(seg0)
    s1, a1 = segments_to_tensors(seg1)
    s0, a0, s1, a1 = s0.to(device), a0.to(device), s1.to(device), a1.to(device)

    with torch.enable_grad():
        for _ in range(steps):
            optimizer.zero_grad()
            phi0 = encoder(s0, a0)
            phi1 = encoder(s1, a1)
            u0 = head.utility(phi0, operator_id, z)
            u1 = head.utility(phi1, operator_id, z)
            delta = preference_delta(u0, u1)
            bt = bradley_terry_loss(labels, delta)
            prior = (z.pow(2).sum()) / (2.0 * sigma_z**2)
            loss = bt + prior
            loss.backward()
            optimizer.step()

    for param in encoder.parameters():
        param.requires_grad_(True)
    for param in head.parameters():
        param.requires_grad_(True)
    encoder.train()
    head.train()
    return z.detach()
