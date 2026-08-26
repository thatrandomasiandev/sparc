"""Tests for trajectory segments and SegmentBuffer."""

from __future__ import annotations

import numpy as np
import pytest

from sparc.env.buffer import SegmentBuffer
from sparc.env.segments import TrajectorySegment, slice_segments


def _rover_trajectory(length: int = 20) -> tuple[np.ndarray, np.ndarray]:
    states = np.zeros((length, 8), dtype=np.float64)
    states[:, 3] = np.linspace(0.5, 2.0, length)  # speed
    states[:, 5:8] = np.linspace(2.0, 0.5, length)[:, None]  # lidar
    actions = np.ones((length, 2), dtype=np.float64) * 0.1
    return states, actions


def test_slice_segments_count() -> None:
    states, actions = _rover_trajectory(10)
    segs = slice_segments(states, actions, horizon=4)
    assert len(segs) == 7


def test_rover_features_ordering() -> None:
    states, actions = _rover_trajectory(5)
    slow = TrajectorySegment(states=states.copy(), actions=actions.copy())
    states_fast = states.copy()
    states_fast[:, 3] = 5.0
    fast = TrajectorySegment(states=states_fast, actions=actions.copy())
    assert fast.features()[0] > slow.features()[0]


def test_segment_buffer_max_capacity() -> None:
    states, actions = _rover_trajectory(20)
    buf = SegmentBuffer(horizon=4, max_segments=5)
    buf.add_trajectory(states, actions)
    assert len(buf) == 5


def test_segment_buffer_sample() -> None:
    states, actions = _rover_trajectory(10)
    buf = SegmentBuffer(horizon=3)
    buf.add_trajectory(states, actions)
    rng = np.random.default_rng(0)
    sample = buf.sample(2, rng)
    assert len(sample) == 2
    assert all(isinstance(s, TrajectorySegment) for s in sample)


def test_segment_horizon_mismatch_raises() -> None:
    states, actions = _rover_trajectory(5)
    seg = TrajectorySegment(states=states, actions=actions)
    buf = SegmentBuffer(horizon=3)
    with pytest.raises(ValueError, match="horizon"):
        buf.add_segment(seg)
