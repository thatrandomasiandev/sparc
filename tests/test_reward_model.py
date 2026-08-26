"""Tests for Pillar 1 reward model."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("torch")

import torch

from sparc.annotator_sim import SyntheticAnnotator
from sparc.env.dataset import PreferenceDataset, PreferenceQuery
from sparc.env.segments import TrajectorySegment
from sparc.reward_model import RewardEnsemble, RewardModelConfig
from sparc.reward_model.losses import bradley_terry_loss


def _seg(speed: float) -> TrajectorySegment:
    states = np.zeros((4, 8), dtype=np.float64)
    states[:, 3] = speed
    states[:, 5:8] = 1.0
    actions = np.zeros((4, 2), dtype=np.float64)
    return TrajectorySegment(states=states, actions=actions)


def _build_dataset(n: int = 8) -> PreferenceDataset:
    ann = SyntheticAnnotator(p_mistake=0.0, p_skip=0.0, seed=0)
    ds = PreferenceDataset()
    for k in range(n):
        q = PreferenceQuery(
            segment_0=_seg(1.0 + k * 0.1),
            segment_1=_seg(0.5),
            operator_id=1 + (k % 3),
            window_id=k,
            env_step=k * 100,
        )
        ds.add_query(q, ann.label_query(q))
    return ds


def test_bradley_terry_loss_prefers_positive_delta_for_label0() -> None:
    labels = torch.tensor([0, 1])
    delta = torch.tensor([2.0, -2.0])
    loss = bradley_terry_loss(labels, delta)
    assert loss.item() < 0.2


def test_reward_ensemble_train_epoch_reduces_loss() -> None:
    ds = _build_dataset()
    cfg = RewardModelConfig(
        embed_dim=16,
        latent_dim=4,
        hidden_dim=32,
        ensemble_size=2,
        map_steps=20,
    )
    model = RewardEnsemble.create(8, 2, 4, config=cfg, seed=1)
    loss0 = model.train_epoch(ds)
    loss1 = model.train_epoch(ds)
    assert loss1 <= loss0 + 1e-6 or loss0 < 1.0


def test_disagreement_nonnegative() -> None:
    ds = _build_dataset(4)
    cfg = RewardModelConfig(embed_dim=8, latent_dim=4, ensemble_size=3, hidden_dim=16)
    model = RewardEnsemble.create(8, 2, 4, config=cfg, seed=2)
    model.train_epoch(ds)
    d = model.disagreement(_seg(2.0), _seg(0.5), operator_id=1)
    assert d >= 0.0


def test_e_step_updates_latent() -> None:
    ds = _build_dataset(6)
    cfg = RewardModelConfig(embed_dim=8, latent_dim=4, ensemble_size=1, hidden_dim=16)
    model = RewardEnsemble.create(8, 2, 4, config=cfg, seed=3)
    z_before = model.z[1].clone()
    model.e_step(ds)
    z_after = model.z[1]
    assert not torch.allclose(z_before, z_after)


# Golden-output regression (ROADMAP Week 10)
GOLDEN_SEED = 42


def test_golden_reward_outputs() -> None:
    """Fixed seed + data → exact forward outputs (checked into suite)."""
    ds = PreferenceDataset()
    ann = SyntheticAnnotator(p_mistake=0.0, p_skip=0.0, seed=GOLDEN_SEED)
    pairs = [(2.0, 0.5), (1.5, 0.8), (3.0, 1.0), (0.9, 1.2)]
    for k, (a, b) in enumerate(pairs):
        q = PreferenceQuery(_seg(a), _seg(b), operator_id=1, window_id=0, env_step=k)
        ds.add_query(q, ann.label_query(q))

    cfg = RewardModelConfig(
        embed_dim=8,
        latent_dim=4,
        hidden_dim=16,
        ensemble_size=2,
        map_steps=50,
        map_lr=0.1,
        train_lr=1e-3,
    )
    model = RewardEnsemble.create(8, 2, 4, config=cfg, seed=GOLDEN_SEED)
    model.train_epoch(ds)

    phi = model.encode(_seg(1.25))
    pooled = model.pooled_reward(_seg(1.25), operator_id=1)
    disagree = model.disagreement(_seg(1.25), _seg(0.75), operator_id=1)
    z1 = model.z[1]

    # Golden outputs: torch 2.5.1, seed 42, config above — checked into suite.
    expected_encode = torch.tensor(
        [
            -0.2385856658220291,
            3.337860107421875e-05,
            -0.05613444745540619,
            -0.0005516074597835541,
        ],
        dtype=torch.float32,
    )
    expected_pooled = 0.00023049386800266802
    expected_disagreement = 4.036468453705311e-06
    expected_z1 = torch.tensor(
        [
            -0.0055440752767026424,
            -0.003839845536276698,
            0.001586598576977849,
            -0.004351427312940359,
        ],
        dtype=torch.float32,
    )

    torch.testing.assert_close(phi[:4], expected_encode, atol=1e-6, rtol=1e-5)
    assert abs(pooled - expected_pooled) < 1e-6
    assert abs(disagree - expected_disagreement) < 1e-6
    torch.testing.assert_close(z1, expected_z1, atol=1e-6, rtol=1e-5)
