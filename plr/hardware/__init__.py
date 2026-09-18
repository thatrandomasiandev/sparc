"""Hardware / human-study interfaces.

Optional path — does not import ROS or vendor SDKs at module import time.
See docs/hardware.md. Core invariants still apply: I3 (measured costs), I6 (seeds).
"""

from plr.hardware.segments import Segment, SegmentLibrary, featurize_states
from plr.hardware.session import PreferenceSession, SessionConfig

__all__ = [
    "Segment",
    "SegmentLibrary",
    "PreferenceSession",
    "SessionConfig",
    "featurize_states",
]
