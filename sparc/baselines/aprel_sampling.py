"""APReL-style batch active query selection (disagreement + greedy medoids)."""

from __future__ import annotations

import numpy as np

from sparc.baselines.reward_mlp import ensemble_disagreement


def segment_pair_feature(sa1: np.ndarray, sa2: np.ndarray) -> np.ndarray:
    """Mean state-action embedding for a segment pair (APReL trajectory features)."""
    return np.asarray((sa1.mean(axis=0) + sa2.mean(axis=0)) * 0.5, dtype=np.float64)


def greedy_medoid_indices(
    sa1: np.ndarray,
    sa2: np.ndarray,
    scores: np.ndarray,
    batch_size: int,
) -> list[int]:
    """
    Greedy farthest-first medoid batch (APReL ``optimization_method='medoids'`` proxy).

    Seed with highest disagreement, then add pairs farthest from the selected set.
    """
    n = len(scores)
    if batch_size >= n:
        return list(range(n))

    features = np.stack([segment_pair_feature(sa1[i], sa2[i]) for i in range(n)])
    remaining = set(range(n))
    selected: list[int] = []

    first = int(np.argmax(scores))
    selected.append(first)
    remaining.remove(first)

    while len(selected) < batch_size and remaining:
        best_idx = -1
        best_dist = -1.0
        for i in remaining:
            fi = features[i]
            min_dist = min(float(np.linalg.norm(fi - features[j])) for j in selected)
            if min_dist > best_dist:
                best_dist = min_dist
                best_idx = i
        if best_idx < 0:
            break
        selected.append(best_idx)
        remaining.remove(best_idx)

    return selected


def aprel_disagreement_scores(
    ensemble: list,
    sa1: np.ndarray,
    sa2: np.ndarray,
) -> np.ndarray:
    return np.array(
        [ensemble_disagreement(ensemble, sa1[i], sa2[i]) for i in range(len(sa1))],
        dtype=np.float64,
    )
