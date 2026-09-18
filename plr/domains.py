"""Domain constructors for mechanistic experiments (E-eq, E-decouple)."""

from __future__ import annotations

import torch
from torch import Tensor

from plr.mdp import Gridworld


def make_decoupled_grid(
    n_rows: int = 6,
    n_cols: int = 6,
    d_task: int = 4,
    d_decoy: int = 4,
    seed: int = 0,
    gamma: float = 0.95,
) -> tuple[Gridworld, dict]:
    """Gridworld with task features on reachable cells and decoy features elsewhere.

    A sealed corner (top-left block) is walled off from the start/goal region.
    - Task dims: nonzero only on **reachable** cells (affect true optimal policies).
    - Decoy dims: nonzero only on **unreachable** cells (visible in query clips that
      sample those cells, but cannot change π* for any unit reward on task dims).

    ``ρ = d_decoy / (d_task + d_decoy)`` is the structural-irrelevance fraction.
    """
    if d_task < 1:
        raise ValueError("d_task must be ≥ 1")
    if d_decoy < 0:
        raise ValueError("d_decoy must be ≥ 0")

    g = torch.Generator().manual_seed(seed)
    S = n_rows * n_cols
    d = d_task + d_decoy
    goal = S - 1

    # Wall off a 2x2 (or larger) corner so those cells are unreachable from the
    # bottom-right start distribution / goal. Corridor blocked along the cut.
    cut_r, cut_c = max(2, n_rows // 3), max(2, n_cols // 3)
    walls: set[int] = set()
    # Vertical cut: block entering the corner from the right
    for r in range(cut_r):
        walls.add(r * n_cols + cut_c)
    # Horizontal cut: block entering from below
    for c in range(cut_c):
        walls.add(cut_r * n_cols + c)

    unreachable = set()
    for r in range(cut_r):
        for c in range(cut_c):
            unreachable.add(r * n_cols + c)

    # Unreachable cells are walls for dynamics (never start / never visited in-task)
    # but retain decoy features so preference queries can still show them.
    walls |= unreachable
    reachable = {s for s in range(S) if s not in walls}

    phi = torch.zeros(S, d)
    for s in reachable:
        if s == goal:
            continue
        phi[s, :d_task] = torch.rand(d_task, generator=g)
    for s in unreachable:
        if d_decoy > 0:
            phi[s, d_task:] = torch.rand(d_decoy, generator=g)

    env = Gridworld(
        n_rows=n_rows, n_cols=n_cols, phi=phi, gamma=gamma, goal=goal, walls=walls
    )
    meta = {
        "d_task": d_task,
        "d_decoy": d_decoy,
        "rho": d_decoy / d if d > 0 else 0.0,
        "n_reachable": len(reachable),
        "n_unreachable": len(unreachable),
        "n_walls": len(walls),
        "unreachable": sorted(unreachable),
        "reachable": sorted(reachable - {goal}),
    }
    return env, meta


def policy_fingerprint(pi: Tensor) -> tuple[int, ...]:
    """Hashable policy id for equivalence clustering."""
    return tuple(int(a) for a in pi.tolist())


def sample_unit_rewards(n: int, d: int, seed: int) -> Tensor:
    from plr.likelihood import unit

    g = torch.Generator().manual_seed(seed)
    return unit(torch.randn(n, d, generator=g))
