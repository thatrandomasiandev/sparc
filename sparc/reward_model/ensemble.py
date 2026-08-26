"""Bootstrap ensemble reward model (Pillar 1)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import cast

import numpy as np
import torch
from torch import nn
from torch.optim import Adam

from sparc.env.dataset import PreferenceDataset, PreferenceRecord
from sparc.env.segments import TrajectorySegment
from sparc.reward_model.adapter import EnsembleHead, segments_to_tensors
from sparc.reward_model.config import RewardModelConfig
from sparc.reward_model.encoder import TrajectoryEncoder
from sparc.reward_model.latent import infer_operator_latent
from sparc.reward_model.losses import bradley_terry_loss, preference_delta


def bootstrap_indices(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, n, size=n)


@dataclass
class RewardEnsemble:
    """
    Shared encoder + M bootstrap adapter heads (DESIGN.md Pillar 1).

    EM-style alternation: z_i MAP-inferred (E-step), encoder/heads trained with z_i frozen.
    """

    encoder: TrajectoryEncoder
    heads: nn.ModuleList
    config: RewardModelConfig
    operator_ids: list[int]
    z: dict[int, torch.Tensor] = field(default_factory=dict)
    bootstrap_indices_per_head: list[np.ndarray] = field(default_factory=list)
    _optimizer: Adam | None = field(default=None, repr=False)

    @classmethod
    def create(
        cls,
        state_dim: int,
        action_dim: int,
        horizon: int,
        config: RewardModelConfig | None = None,
        operator_ids: list[int] | None = None,
        seed: int = 0,
    ) -> RewardEnsemble:
        cfg = config or RewardModelConfig()
        op_ids = operator_ids or list(range(1, cfg.num_operators + 1))
        torch.manual_seed(seed)
        encoder = TrajectoryEncoder(
            state_dim=state_dim,
            action_dim=action_dim,
            horizon=horizon,
            embed_dim=cfg.embed_dim,
            hidden_dim=cfg.hidden_dim,
        )
        heads = nn.ModuleList(
            [
                EnsembleHead(cfg.embed_dim, cfg.latent_dim, op_ids)
                for _ in range(cfg.ensemble_size)
            ]
        )
        model = cls(
            encoder=encoder,
            heads=heads,
            config=cfg,
            operator_ids=op_ids,
        )
        model._init_latents()
        model._init_optimizer()
        return model

    def _init_latents(self) -> None:
        for op_id in self.operator_ids:
            self.z[op_id] = torch.zeros(self.config.latent_dim)

    def _init_optimizer(self) -> None:
        params = list(self.encoder.parameters())
        for head in self.heads:
            params.extend(head.parameters())
        self._optimizer = Adam(params, lr=self.config.train_lr)

    def set_bootstrap_indices(self, dataset: PreferenceDataset) -> None:
        labeled = dataset.labeled()
        n = len(labeled)
        self.bootstrap_indices_per_head = [
            bootstrap_indices(n, seed=m) for m in range(self.config.ensemble_size)
        ]
        self._labeled_cache = labeled

    def e_step(self, dataset: PreferenceDataset) -> None:
        """Recompute MAP z_i for all operators using head 0 adapter."""
        for op_id in self.operator_ids:
            self.z[op_id] = infer_operator_latent(
                dataset.labeled(),
                op_id,
                self.encoder,
                cast(EnsembleHead, self.heads[0]),
                latent_dim=self.config.latent_dim,
                sigma_z=self.config.sigma_z,
                steps=self.config.map_steps,
                lr=self.config.map_lr,
            )

    def _batch_from_records(
        self,
        records: list[PreferenceRecord],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list[int]]:
        seg0 = [r.query.segment_0 for r in records]
        seg1 = [r.query.segment_1 for r in records]
        labels = torch.tensor([r.label for r in records], dtype=torch.long)
        op_ids = [r.query.operator_id for r in records]
        s0, a0 = segments_to_tensors(seg0)
        s1, a1 = segments_to_tensors(seg1)
        return s0, a0, s1, a1, labels, op_ids

    def train_m_step(self, head_index: int) -> float:
        """One Adam step on encoder + head `head_index` with z_i frozen."""
        if not self.bootstrap_indices_per_head:
            raise RuntimeError("call set_bootstrap_indices before training")
        indices = self.bootstrap_indices_per_head[head_index]
        records = [self._labeled_cache[i] for i in indices]
        if not records:
            return 0.0

        s0, a0, s1, a1, labels, op_ids = self._batch_from_records(records)
        head = self.heads[head_index]
        assert self._optimizer is not None

        self._optimizer.zero_grad()
        phi0 = self.encoder(s0, a0)
        phi1 = self.encoder(s1, a1)
        utilities0: list[torch.Tensor] = []
        utilities1: list[torch.Tensor] = []
        for idx, op_id in enumerate(op_ids):
            z_i = self.z[op_id]
            utilities0.append(head.utility(phi0[idx : idx + 1], op_id, z_i))
            utilities1.append(head.utility(phi1[idx : idx + 1], op_id, z_i))
        u0 = torch.cat(utilities0)
        u1 = torch.cat(utilities1)
        delta = preference_delta(u0, u1)
        loss = bradley_terry_loss(labels, delta)
        loss.backward()
        self._optimizer.step()
        return float(loss.item())

    def train_epoch(self, dataset: PreferenceDataset) -> float:
        """Full EM epoch: E-step then M-step for each bootstrap head."""
        self.set_bootstrap_indices(dataset)
        self.e_step(dataset)
        total = 0.0
        for m in range(self.config.ensemble_size):
            total += self.train_m_step(m)
        return total / max(self.config.ensemble_size, 1)

    @torch.no_grad()
    def encode(self, segment: TrajectorySegment) -> torch.Tensor:
        s, a = segments_to_tensors([segment])
        return cast(torch.Tensor, self.encoder(s, a).squeeze(0))

    @torch.no_grad()
    def rewards(
        self,
        segment: TrajectorySegment,
        operator_id: int,
    ) -> torch.Tensor:
        """Per-head scalar reward g_i^(m)(sigma). Shape (M,)."""
        phi = self.encode(segment).unsqueeze(0)
        z_i = self.z[operator_id]
        return torch.tensor(
            [head.utility(phi, operator_id, z_i).item() for head in self.heads]
        )

    @torch.no_grad()
    def disagreement(
        self,
        seg0: TrajectorySegment,
        seg1: TrajectorySegment,
        operator_id: int,
    ) -> float:
        """Std_m of preference logit delta across ensemble (Pillar 1 / 3 signal)."""
        phi0 = self.encode(seg0).unsqueeze(0)
        phi1 = self.encode(seg1).unsqueeze(0)
        z_i = self.z[operator_id]
        deltas = []
        for head in self.heads:
            u0 = head.utility(phi0, operator_id, z_i)
            u1 = head.utility(phi1, operator_id, z_i)
            deltas.append(float((u0 - u1).item()))
        return float(np.std(deltas))

    @torch.no_grad()
    def uncertainty(self, segment: TrajectorySegment, operator_id: int) -> float:
        """Std_m of r^(m)(sigma) across ensemble."""
        rs = self.rewards(segment, operator_id)
        return float(torch.std(rs).item())

    @torch.no_grad()
    def pooled_reward(self, segment: TrajectorySegment, operator_id: int) -> float:
        return float(self.rewards(segment, operator_id).mean().item())
