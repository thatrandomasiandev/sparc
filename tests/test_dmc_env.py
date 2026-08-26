"""Tests for DMControl env factory (requires optional [dmc] extra)."""

from __future__ import annotations

import pytest

from sparc.baselines.config import BaselineConfig
from sparc.env.dmc import is_dmc_available, make_dmc_env
from sparc.env.factory import make_env
from sparc.env.specs import DMC_ENV_SPECS, get_env_spec
from sparc.harness.env_defaults import apply_env_defaults

pytestmark = pytest.mark.skipif(not is_dmc_available(), reason="shimmy/dm-control not installed")


def test_env_spec_registry() -> None:
    spec = get_env_spec("walker-walk")
    assert spec is not None
    assert spec.segment_horizon == 50
    assert spec.train_steps == 500_000
    assert spec.gym_id == "dm_control/walker-walk-v0"
    assert "cheetah-run" in DMC_ENV_SPECS


def test_make_dmc_env_walker_walk() -> None:
    spec = get_env_spec("walker-walk")
    assert spec is not None
    env = make_dmc_env(spec, seed=0)
    obs, _ = env.reset()
    assert obs.shape == (24,)
    assert env.action_space.shape == (6,)
    action = env.action_space.sample()
    obs2, reward, term, trunc, _ = env.step(action)
    assert obs2.shape == (24,)
    assert isinstance(reward, float)
    assert term is False or trunc is False or term or trunc


def test_factory_canonical_id() -> None:
    env = make_env("walker-walk", seed=1)
    obs, _ = env.reset()
    assert obs.ndim == 1
    assert obs.shape[0] > 0


def test_apply_env_defaults_segment() -> None:
    cfg = BaselineConfig(env_id="walker-walk", segment_length=8)
    updated = apply_env_defaults(cfg)
    assert updated.segment_length == 50
