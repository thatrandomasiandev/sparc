"""Held-out reward-model pairwise accuracy (EXPERIMENTS.md metric 2)."""

from __future__ import annotations

import numpy as np

from sparc.annotator_sim.synthetic import SyntheticAnnotator
from sparc.env.segments import TrajectorySegment
from sparc.reward_model.ensemble import RewardEnsemble


def oracle_preference_label(
    segment_0: TrajectorySegment,
    segment_1: TrajectorySegment,
    annotator: SyntheticAnnotator,
    operator_id: int,
    env_step: int,
) -> int:
    """Noise-free oracle label from U_i(σ) = α_i · f̄(σ)."""
    u0 = annotator.utility(segment_0, operator_id, env_step)
    u1 = annotator.utility(segment_1, operator_id, env_step)
    return 0 if u0 >= u1 else 1


def held_out_pairwise_accuracy(
    ensemble: RewardEnsemble,
    annotator: SyntheticAnnotator,
    segments: list[TrajectorySegment],
    operator_id: int,
    env_step: int,
    n_pairs: int = 500,
    seed: int | None = None,
) -> float:
    """
    Macro accuracy on held-out segment pairs (J=500 default, EXPERIMENTS.md).

    Compares sign(r̂(σ0) - r̂(σ1)) to oracle preference without label noise.
    """
    if len(segments) < 2:
        return 0.0
    rng = np.random.default_rng(seed)
    correct = 0
    evaluated = 0
    for _ in range(n_pairs):
        i, j = rng.integers(0, len(segments), size=2)
        if i == j:
            continue
        seg0, seg1 = segments[i], segments[j]
        oracle = oracle_preference_label(seg0, seg1, annotator, operator_id, env_step)
        r0 = ensemble.pooled_reward(seg0, operator_id)
        r1 = ensemble.pooled_reward(seg1, operator_id)
        pred = 0 if r0 >= r1 else 1
        correct += int(pred == oracle)
        evaluated += 1
    return correct / evaluated if evaluated else 0.0


def per_operator_macro_accuracy(
    ensemble: RewardEnsemble,
    annotator: SyntheticAnnotator,
    segments: list[TrajectorySegment],
    operator_ids: list[int],
    env_step: int,
    n_pairs: int = 500,
    seed: int | None = None,
) -> dict[int, float]:
    """Per-operator held-out accuracy; macro-average is mean of values."""
    out: dict[int, float] = {}
    for op_id in operator_ids:
        out[op_id] = held_out_pairwise_accuracy(
            ensemble,
            annotator,
            segments,
            op_id,
            env_step,
            n_pairs=n_pairs,
            seed=None if seed is None else seed + op_id,
        )
    return out
