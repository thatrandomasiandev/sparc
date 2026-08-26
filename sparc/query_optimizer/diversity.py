"""Query diversity and similarity penalties."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from sparc.query_optimizer.eig import QueryCandidate


def query_feature_vector(candidate: QueryCandidate) -> torch.Tensor:
    if candidate.phi0 is None or candidate.phi1 is None:
        raise RuntimeError("embeddings required — call ensure_embeddings first")
    return torch.cat([candidate.phi0, candidate.phi1], dim=0)


def similarity(a: QueryCandidate, b: QueryCandidate) -> float:
    """Cosine similarity between concatenated segment embeddings."""
    va = query_feature_vector(a)
    vb = query_feature_vector(b)
    return float(F.cosine_similarity(va.unsqueeze(0), vb.unsqueeze(0)).item())


def diversity_bonus(candidate: QueryCandidate, selected: list[QueryCandidate]) -> float:
    """Higher when candidate is dissimilar to already-selected queries."""
    if not selected:
        return 1.0
    sims = [similarity(candidate, s) for s in selected]
    return 1.0 - max(sims)
