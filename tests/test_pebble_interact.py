"""Regression: PEBBLE num_interact is in env steps, not learn-chunks."""

from __future__ import annotations

from sparc.baselines.loop import _maybe_query
from sparc.baselines.config import BaselineConfig
from sparc.logging.jsonl import JsonlLogger


class _FakeRM:
    total_labels = 0

    def sample_preferences(self, feed_type: int = 0) -> int:
        self.total_labels += 5
        return 5

    def train(self) -> float:
        return 0.9


def test_maybe_query_counts_env_steps_not_chunks() -> None:
    cfg = BaselineConfig(num_interact=5000, max_feedback=1000, num_seed_steps=1000, num_unsup_steps=5000)
    rm = _FakeRM()
    logger = JsonlLogger(path=None)
    # After first query, accumulate env steps across 128-step chunks
    steps_since, feedback, _, first = _maybe_query(
        effective_step=6000,
        cfg=cfg,
        reward_model=rm,  # type: ignore[arg-type]
        unsup_end=6000,
        steps_since_query=0,
        total_feedback=0,
        first_query_done=False,
        logger=logger,
        learn_steps=128,
    )
    assert first is True
    assert feedback == 5
    assert rm.total_labels == 5

    # 39 chunks * 128 = 4992 < 5000 → no second query
    for _ in range(39):
        steps_since, feedback, _, first = _maybe_query(
            effective_step=6000 + (_ + 1) * 128,
            cfg=cfg,
            reward_model=rm,  # type: ignore[arg-type]
            unsup_end=6000,
            steps_since_query=steps_since,
            total_feedback=feedback,
            first_query_done=True,
            logger=logger,
            learn_steps=128,
        )
    assert feedback == 5
    assert steps_since == 39 * 128

    # One more chunk crosses 5000 env steps → second query
    steps_since, feedback, _, first = _maybe_query(
        effective_step=6000 + 40 * 128,
        cfg=cfg,
        reward_model=rm,  # type: ignore[arg-type]
        unsup_end=6000,
        steps_since_query=steps_since,
        total_feedback=feedback,
        first_query_done=True,
        logger=logger,
        learn_steps=128,
    )
    assert feedback == 10
    assert steps_since == 0
