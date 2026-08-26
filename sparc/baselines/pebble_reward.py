"""PEBBLE reward model adapted from B-Pref (Lee et al. 2022)."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
from torch import nn
from torch.optim import Adam

from sparc.baselines.reward_mlp import RewardMLP, ensemble_disagreement


def _flat_action(action: np.ndarray) -> np.ndarray:
    arr = np.atleast_1d(np.asarray(action, dtype=np.float32))
    return arr.reshape(-1)


@dataclass
class PebbleRewardModel:
    """
    Ensemble of reward MLPs with disagreement-based query selection.

    Labels use oracle segment return sums (B-Pref teacher) unless overridden.
    """

    obs_dim: int
    action_dim: int
    segment_length: int = 50
    ensemble_size: int = 3
    lr: float = 3e-4
    mb_size: int = 128
    large_batch: int = 10
    activation: str = "tanh"
    capacity: int = 50_000
    train_batch_size: int = 128
    reward_update: int = 200
    teacher_gamma: float = 1.0
    teacher_eps_mistake: float = 0.0
    teacher_eps_skip: float = 0.0
    teacher_eps_equal: float = 0.0
    seed: int = 0

    sa_dim: int = field(init=False)
    ensemble: list[RewardMLP] = field(default_factory=list, init=False)
    optimizer: Adam | None = field(default=None, init=False)
    trajectories: list[np.ndarray] = field(default_factory=list, init=False)
    traj_returns: list[np.ndarray] = field(default_factory=list, init=False)
    buffer_seg1: np.ndarray = field(init=False)
    buffer_seg2: np.ndarray = field(init=False)
    buffer_label: np.ndarray = field(init=False)
    buffer_index: int = field(default=0, init=False)
    buffer_full: bool = field(default=False, init=False)
    total_labels: int = field(default=0, init=False)
    _rng: np.random.Generator = field(init=False)

    def __post_init__(self) -> None:
        self.sa_dim = self.obs_dim + self.action_dim
        self._rng = np.random.default_rng(self.seed)
        torch.manual_seed(self.seed)
        self.ensemble = [
            RewardMLP(self.sa_dim, activation=self.activation) for _ in range(self.ensemble_size)
        ]
        params = [p for m in self.ensemble for p in m.parameters()]
        self.optimizer = Adam(params, lr=self.lr)
        seg = self.segment_length
        self.buffer_seg1 = np.empty((self.capacity, seg, self.sa_dim), dtype=np.float32)
        self.buffer_seg2 = np.empty((self.capacity, seg, self.sa_dim), dtype=np.float32)
        self.buffer_label = np.empty((self.capacity, 1), dtype=np.float32)

    def start_trajectory(self) -> None:
        """Begin a new stored trajectory (call on env reset)."""
        self.trajectories.append(np.empty((0, self.sa_dim), dtype=np.float32))
        self.traj_returns.append(np.empty((0, 1), dtype=np.float32))

    def add_transition(self, obs: np.ndarray, action: np.ndarray, reward: float) -> None:
        sa = np.concatenate([obs.astype(np.float32), _flat_action(action)])
        if not self.trajectories:
            self.start_trajectory()
        self.trajectories[-1] = np.vstack([self.trajectories[-1], sa])
        self.traj_returns[-1] = np.vstack([self.traj_returns[-1], [[reward]]])

    def r_hat(self, obs: np.ndarray, action: np.ndarray) -> float:
        sa = np.concatenate([obs.astype(np.float32), _flat_action(action)])
        x = torch.from_numpy(sa).float()
        rewards = [float(m(x).item()) for m in self.ensemble]
        return float(np.mean(rewards))

    def step_disagreement(self, obs: np.ndarray, action: np.ndarray) -> float:
        """Per-step ensemble reward std (RUNE exploration signal)."""
        sa = np.concatenate([obs.astype(np.float32), _flat_action(action)])
        x = torch.from_numpy(sa).float()
        rewards = [float(m(x).item()) for m in self.ensemble]
        return float(np.std(rewards))

    def _sample_segment_pair(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if len(self.trajectories) < 2:
            raise RuntimeError("need at least two trajectories for query sampling")
        min(len(t) for t in self.trajectories if len(t) >= self.segment_length)
        valid = [i for i, t in enumerate(self.trajectories) if len(t) >= self.segment_length]
        if not valid:
            raise RuntimeError("no trajectory long enough for segment_length")
        mb = self.mb_size * self.large_batch
        idx1 = self._rng.choice(valid, size=mb, replace=True)
        idx2 = self._rng.choice(valid, size=mb, replace=True)
        sa1, r1, sa2, r2 = [], [], [], []
        for i, j in zip(idx1, idx2, strict=True):
            t1, t2 = self.trajectories[i], self.trajectories[j]
            s1 = self._rng.integers(0, len(t1) - self.segment_length + 1)
            s2 = self._rng.integers(0, len(t2) - self.segment_length + 1)
            sa1.append(t1[s1 : s1 + self.segment_length])
            sa2.append(t2[s2 : s2 + self.segment_length])
            r1.append(self.traj_returns[i][s1 : s1 + self.segment_length])
            r2.append(self.traj_returns[j][s2 : s2 + self.segment_length])
        return (
            np.stack(sa1),
            np.stack(sa2),
            np.stack(r1),
            np.stack(r2),
        )

    def _oracle_labels(
        self,
        sa1: np.ndarray,
        sa2: np.ndarray,
        r1: np.ndarray,
        r2: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        sum1 = r1.sum(axis=1).reshape(-1)
        sum2 = r2.sum(axis=1).reshape(-1)
        keep = np.ones(len(sum1), dtype=bool)
        if self.teacher_eps_skip > 0:
            keep &= np.maximum(sum1, sum2) > self.teacher_eps_skip
        if not keep.any():
            return sa1[:0], sa2[:0], r1[:0], r2[:0], np.empty(0, dtype=np.int64)
        sa1, sa2, r1, r2, sum1, sum2 = (
            sa1[keep],
            sa2[keep],
            r1[keep],
            r2[keep],
            sum1[keep],
            sum2[keep],
        )
        equal = np.abs(sum1 - sum2) < self.teacher_eps_equal
        labels = (sum2 > sum1).astype(np.int64)
        if self.teacher_eps_mistake > 0:
            flip = self._rng.random(len(labels)) < self.teacher_eps_mistake
            labels = np.where(flip, 1 - labels, labels)
        labels = labels[~equal]
        sa1, sa2, r1, r2 = sa1[~equal], sa2[~equal], r1[~equal], r2[~equal]
        return sa1, sa2, r1, r2, labels

    def _put_queries(
        self,
        sa1: np.ndarray,
        sa2: np.ndarray,
        labels: np.ndarray,
    ) -> int:
        n = len(labels)
        if n == 0:
            return 0
        next_idx = self.buffer_index + n
        if next_idx >= self.capacity:
            self.buffer_full = True
            first = self.capacity - self.buffer_index
            self.buffer_seg1[self.buffer_index : self.capacity] = sa1[:first]
            self.buffer_seg2[self.buffer_index : self.capacity] = sa2[:first]
            self.buffer_label[self.buffer_index : self.capacity] = labels[:first, None]
            remain = n - first
            if remain > 0:
                self.buffer_seg1[:remain] = sa1[first:]
                self.buffer_seg2[:remain] = sa2[first:]
                self.buffer_label[:remain] = labels[first:, None]
            self.buffer_index = remain
        else:
            self.buffer_seg1[self.buffer_index : next_idx] = sa1
            self.buffer_seg2[self.buffer_index : next_idx] = sa2
            self.buffer_label[self.buffer_index : next_idx] = labels[:, None]
            self.buffer_index = next_idx
        self.total_labels += n
        return n

    def uniform_sampling(self) -> int:
        sa1, sa2, r1, r2 = self._sample_segment_pair()
        take = min(self.mb_size, len(sa1))
        sa1, sa2, r1, r2 = sa1[:take], sa2[:take], r1[:take], r2[:take]
        sa1, sa2, r1, r2, labels = self._oracle_labels(sa1, sa2, r1, r2)
        return self._put_queries(sa1, sa2, labels)

    def disagreement_sampling(self) -> int:
        sa1, sa2, r1, r2 = self._sample_segment_pair()
        scores = [
            ensemble_disagreement(self.ensemble, sa1[i], sa2[i]) for i in range(len(sa1))
        ]
        order = np.argsort(-np.array(scores))[: self.mb_size]
        sa1, sa2, r1, r2 = sa1[order], sa2[order], r1[order], r2[order]
        sa1, sa2, r1, r2, labels = self._oracle_labels(sa1, sa2, r1, r2)
        return self._put_queries(sa1, sa2, labels)

    def aprel_batch_sampling(self) -> int:
        """APReL batch learner proxy: disagreement seed + greedy medoid diversity."""
        from sparc.baselines.aprel_sampling import (
            aprel_disagreement_scores,
            greedy_medoid_indices,
        )

        sa1, sa2, r1, r2 = self._sample_segment_pair()
        scores = aprel_disagreement_scores(self.ensemble, sa1, sa2)
        order = greedy_medoid_indices(sa1, sa2, scores, self.mb_size)
        sa1, sa2, r1, r2 = sa1[order], sa2[order], r1[order], r2[order]
        sa1, sa2, r1, r2, labels = self._oracle_labels(sa1, sa2, r1, r2)
        return self._put_queries(sa1, sa2, labels)

    def sample_preferences(self, feed_type: int) -> int:
        if feed_type == 0:
            return self.uniform_sampling()
        if feed_type == 1:
            return self.disagreement_sampling()
        if feed_type == 2:
            return self.aprel_batch_sampling()
        raise ValueError(f"unsupported feed_type={feed_type}")

    def train(self) -> float:
        max_len = self.capacity if self.buffer_full else self.buffer_index
        if max_len == 0 or self.optimizer is None:
            return 0.0
        member_accs: list[float] = []
        for _ in range(self.reward_update):
            self.optimizer.zero_grad()
            batch_indices = self._rng.choice(
                max_len, size=min(self.train_batch_size, max_len), replace=False
            )
            s1 = torch.from_numpy(self.buffer_seg1[batch_indices]).float()
            s2 = torch.from_numpy(self.buffer_seg2[batch_indices]).float()
            y = torch.from_numpy(self.buffer_label[batch_indices].flatten()).long()
            loss = torch.tensor(0.0)
            for model in self.ensemble:
                r1 = model(s1).sum(dim=1)
                r2 = model(s2).sum(dim=1)
                logits = torch.stack([r1, r2], dim=1)
                loss = loss + nn.functional.cross_entropy(logits, y)
            loss.backward()
            self.optimizer.step()
            with torch.no_grad():
                r1 = self.ensemble[0](s1).sum(dim=1)
                r2 = self.ensemble[0](s2).sum(dim=1)
                logits = torch.stack([r1, r2], dim=1)
                acc = float((logits.argmax(dim=1) == y).float().mean().item())
                member_accs.append(acc)
                if acc > 0.97:
                    break
        return float(np.mean(member_accs)) if member_accs else 0.0
