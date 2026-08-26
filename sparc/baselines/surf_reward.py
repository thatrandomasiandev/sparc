"""SURF baseline — semi-supervised reward learning with segment augmentation."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
from torch import nn

from sparc.baselines.pebble_reward import PebbleRewardModel


@dataclass
class SurfRewardModel(PebbleRewardModel):
    """
    PEBBLE reward ensemble + pseudo-labels on unlabeled segment pairs (SURF, Park et al. 2022).

    Uses ensemble agreement above ``surf_threshold`` for pseudo-labels and Gaussian
    noise augmentation on state-action segments.
    """

    surf_threshold: float = 0.95
    surf_noise_std: float = 0.1
    surf_unlabeled_batch: int = 128
    surf_pseudo_weight: float = 1.0

    def train(self) -> float:
        labeled_acc = super().train()
        pseudo_acc = self._train_pseudo_batch()
        if pseudo_acc > 0.0:
            return (labeled_acc + pseudo_acc) * 0.5
        return labeled_acc

    def _train_pseudo_batch(self) -> float:
        max_len = self.capacity if self.buffer_full else self.buffer_index
        if max_len == 0 or self.optimizer is None:
            return 0.0
        try:
            sa1, sa2, _r1, _r2 = self._sample_segment_pair()
        except RuntimeError:
            return 0.0

        take = min(self.surf_unlabeled_batch, len(sa1))
        if take == 0:
            return 0.0
        sa1, sa2 = sa1[:take], sa2[:take]

        sa1_p, sa2_p, pseudo_labels = self._pseudo_labels(sa1, sa2)
        if len(pseudo_labels) == 0:
            return 0.0

        noise = self.surf_noise_std
        sa1_aug = sa1_p + self._rng.normal(0.0, noise, sa1_p.shape).astype(np.float32)
        sa2_aug = sa2_p + self._rng.normal(0.0, noise, sa2_p.shape).astype(np.float32)

        s1 = torch.from_numpy(sa1_aug).float()
        s2 = torch.from_numpy(sa2_aug).float()
        y = torch.from_numpy(pseudo_labels).long()

        member_accs: list[float] = []
        steps = max(1, self.reward_update // 4)
        for _ in range(steps):
            self.optimizer.zero_grad()
            loss = torch.tensor(0.0)
            for model in self.ensemble:
                r1 = model(s1).sum(dim=1)
                r2 = model(s2).sum(dim=1)
                logits = torch.stack([r1, r2], dim=1)
                loss = loss + self.surf_pseudo_weight * nn.functional.cross_entropy(logits, y)
            loss.backward()
            self.optimizer.step()
            with torch.no_grad():
                r1 = self.ensemble[0](s1).sum(dim=1)
                r2 = self.ensemble[0](s2).sum(dim=1)
                logits = torch.stack([r1, r2], dim=1)
                member_accs.append(float((logits.argmax(dim=1) == y).float().mean().item()))
        return float(np.mean(member_accs)) if member_accs else 0.0

    def _pseudo_labels(
        self,
        sa1: np.ndarray,
        sa2: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return pseudo-labeled (sa1, sa2, y) triples for confident ensemble pairs."""
        picked_sa1: list[np.ndarray] = []
        picked_sa2: list[np.ndarray] = []
        labels: list[int] = []
        for i in range(len(sa1)):
            with torch.no_grad():
                x1 = torch.from_numpy(sa1[i]).float()
                x2 = torch.from_numpy(sa2[i]).float()
                member_probs = []
                for member in self.ensemble:
                    r1 = member(x1).sum()
                    r2 = member(x2).sum()
                    stacked = torch.stack([r1, r2])
                    member_probs.append(float(torch.softmax(stacked, dim=0)[0].item()))
                mean_prob = float(np.mean(member_probs))
                std_prob = float(np.std(member_probs))
                if std_prob > 1.0 - self.surf_threshold:
                    continue
                if mean_prob >= self.surf_threshold:
                    picked_sa1.append(sa1[i])
                    picked_sa2.append(sa2[i])
                    labels.append(0)
                elif mean_prob <= 1.0 - self.surf_threshold:
                    picked_sa1.append(sa1[i])
                    picked_sa2.append(sa2[i])
                    labels.append(1)
        if not labels:
            return sa1[:0], sa2[:0], np.empty(0, dtype=np.int64)
        return np.stack(picked_sa1), np.stack(picked_sa2), np.array(labels, dtype=np.int64)
