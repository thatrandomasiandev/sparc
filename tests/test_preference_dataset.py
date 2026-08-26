"""Tests for PreferenceDataset."""

from __future__ import annotations

import numpy as np
import pytest

from sparc.env.dataset import PreferenceDataset, PreferenceQuery, PreferenceRecord
from sparc.env.segments import TrajectorySegment


def _seg(speed: float) -> TrajectorySegment:
    states = np.zeros((4, 8), dtype=np.float64)
    states[:, 3] = speed
    states[:, 5:8] = 1.0
    actions = np.zeros((4, 2), dtype=np.float64)
    return TrajectorySegment(states=states, actions=actions)


def test_preference_dataset_labeled_filter() -> None:
    ds = PreferenceDataset()
    q = PreferenceQuery(_seg(1.0), _seg(2.0), operator_id=1, window_id=0, env_step=100)
    ds.add_query(q, label=0)
    ds.add_query(q, label=None)
    assert len(ds) == 2
    assert len(ds.labeled()) == 1


def test_preference_dataset_for_operator() -> None:
    ds = PreferenceDataset()
    q1 = PreferenceQuery(_seg(1.0), _seg(2.0), operator_id=1, window_id=0, env_step=0)
    q2 = PreferenceQuery(_seg(1.0), _seg(2.0), operator_id=2, window_id=1, env_step=0)
    ds.add_query(q1, label=0)
    ds.add_query(q2, label=1)
    assert len(ds.for_operator(2)) == 1


def test_invalid_label_raises() -> None:
    ds = PreferenceDataset()
    q = PreferenceQuery(_seg(1.0), _seg(2.0), operator_id=1, window_id=0, env_step=0)
    with pytest.raises(ValueError, match="label"):
        ds.add(PreferenceRecord(query=q, label=2))
