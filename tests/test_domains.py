"""Domain constructor tests."""

from __future__ import annotations

import torch

from plr.domains import make_decoupled_grid, policy_fingerprint, sample_unit_rewards
from plr.likelihood import unit


def test_decoupled_rho_and_support() -> None:
    env, meta = make_decoupled_grid(d_task=4, d_decoy=4, seed=0)
    assert abs(meta["rho"] - 0.5) < 1e-9
    # Reachable cells: decoy dims ~ 0
    for s in meta["reachable"]:
        assert torch.allclose(env.phi[s, meta["d_task"] :], torch.zeros(meta["d_decoy"]))
    # Unreachable: task dims ~ 0
    for s in meta["unreachable"]:
        assert torch.allclose(env.phi[s, : meta["d_task"]], torch.zeros(meta["d_task"]))


def test_decoy_does_not_change_optimal_policy() -> None:
    """Flipping decoy coordinates of w must not change π* (structural claim)."""
    env, meta = make_decoupled_grid(d_task=3, d_decoy=5, seed=1)
    d = meta["d_task"] + meta["d_decoy"]
    w = unit(torch.randn(d))
    w2 = w.clone()
    w2[meta["d_task"] :] = unit(torch.randn(meta["d_decoy"]))  # scramble decoy block
    # Renormalize full vector
    w2 = unit(w2)
    # Stronger test: same task projection direction
    w_task = torch.zeros(d)
    w_task[: meta["d_task"]] = unit(w[: meta["d_task"]])
    w_task_alt = w_task.clone()
    w_task_alt[meta["d_task"] :] = unit(torch.randn(meta["d_decoy"]))
    w_task_alt = unit(w_task_alt)
    _, pi_a = env.value_iteration(unit(w_task))
    _, pi_b = env.value_iteration(unit(w_task_alt))
    assert policy_fingerprint(pi_a) == policy_fingerprint(pi_b)


def test_sample_unit_rewards_norm() -> None:
    w = sample_unit_rewards(10, 5, seed=0)
    assert torch.allclose(w.norm(dim=-1), torch.ones(10), atol=1e-5)
