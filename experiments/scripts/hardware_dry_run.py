#!/usr/bin/env python3
"""Dry-run a hardware preference session with synthetic segments (no robot required).

Use this to verify logging + QueryCosts before a lab booking.
Real robot: swap SegmentLibrary.from_state_files(...) with logged teleop clips.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from plr.hardware.segments import Segment, SegmentLibrary, featurize_states
from plr.hardware.session import PreferenceSession, SessionConfig, cli_answer_fn


def _synthetic_library(n: int, t: int, s_dim: int, d: int, seed: int) -> SegmentLibrary:
    g = torch.Generator().manual_seed(seed)
    # Random projection to d features so acquisition dim matches sim experiments.
    proj = torch.randn(d, s_dim, generator=g)
    segs = []
    for i in range(n):
        states = torch.randn(t, s_dim, generator=g)
        phi = proj @ featurize_states(states, feature_fn="discounted_sum")
        segs.append(Segment(segment_id=f"clip_{i:02d}", states=states, phi=phi))
    return SegmentLibrary(segments=segs)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-segments", type=int, default=6)
    p.add_argument("--n-queries", type=int, default=3)
    p.add_argument("--auto", action="store_true", help="Auto-answer (prefer index 0) for CI")
    p.add_argument("--out", type=Path, default=Path("experiments/runs/hardware/dry_run"))
    args = p.parse_args()

    lib = _synthetic_library(args.n_segments, t=20, s_dim=8, d=4, seed=args.seed)
    pairs = [(i, (i + 1) % args.n_segments) for i in range(args.n_queries)]

    def auto_fn(phi, labels):  # noqa: ANN001
        return [0, 1]

    session = PreferenceSession(
        config=SessionConfig(
            session_id="dry_run",
            platform="synthetic",
            strategy="schedule_fixed",
            seed=args.seed,
            notes="hardware dry-run; replace clips with robot logs",
        ),
        library=lib,
        out_dir=args.out,
        answer_fn=auto_fn if args.auto else cli_answer_fn,
        query_pairs=pairs,
    )
    costs = session.run()
    print(f"Measured QueryCosts: {costs.seconds}")
    print(f"Wrote session to {args.out}")


if __name__ == "__main__":
    main()
