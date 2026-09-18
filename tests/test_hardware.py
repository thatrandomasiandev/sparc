"""Hardware session logging — costs measured, never defaulted."""

from __future__ import annotations

from pathlib import Path

import torch

from plr.acquisition import QueryCosts
from plr.hardware.segments import Segment, SegmentLibrary, featurize_states
from plr.hardware.session import PreferenceSession, SessionConfig


def test_featurize_discounted_sum_shape() -> None:
    states = torch.randn(10, 5)
    phi = featurize_states(states, feature_fn="discounted_sum")
    assert phi.shape == (5,)


def test_session_writes_measured_costs(tmp_path: Path) -> None:
    segs = [
        Segment(segment_id="a", states=torch.zeros(4, 3), phi=torch.tensor([1.0, 0.0])),
        Segment(segment_id="b", states=torch.zeros(4, 3), phi=torch.tensor([0.0, 1.0])),
        Segment(segment_id="c", states=torch.zeros(4, 3), phi=torch.tensor([0.5, 0.5])),
    ]
    lib = SegmentLibrary(segments=segs)

    def answer_fn(phi, labels):  # noqa: ANN001
        return [0, 1]

    session = PreferenceSession(
        config=SessionConfig(session_id="t", platform="test", seed=0),
        library=lib,
        out_dir=tmp_path,
        answer_fn=answer_fn,
        query_pairs=[(0, 1), (1, 2)],
    )
    costs = session.run()
    assert isinstance(costs, QueryCosts)
    assert "binary" in costs.seconds
    assert (tmp_path / "meta.json").exists()
    assert (tmp_path / "queries.jsonl").exists()
    assert (tmp_path / "answers.jsonl").exists()
    assert (tmp_path / "costs.jsonl").exists()
