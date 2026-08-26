"""Week 19 harness dry-run suite tests."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from sparc.harness.dry_run import SuiteManifest, run_suite_from_manifest


def test_suite_manifest_loads() -> None:
    manifest = SuiteManifest.from_json_file(
        Path("experiments/configs/dry_run_pendulum_ci.json")
    )
    assert manifest.suite_name == "week19_pendulum_ci"
    assert "pebble" in manifest.methods
    assert "sparc" in manifest.methods


@pytest.mark.timeout(600)
def test_pendulum_ci_suite_smoke() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "suite"
        result = run_suite_from_manifest(
            "experiments/configs/dry_run_pendulum_ci.json",
            out,
            seeds_spec="0",
        )
        assert len(result.methods) == 4
        summary_path = out / "suite_summary.json"
        assert summary_path.exists()
        data = json.loads(summary_path.read_text(encoding="utf-8"))
        assert data["suite_name"] == "week19_pendulum_ci"
        assert "sparc_vs_pebble" in data["comparisons"]
        for method in ("pebble", "prefppo", "rune", "sparc"):
            assert (out / method / "summary.json").exists()
