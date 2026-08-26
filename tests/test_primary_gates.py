"""Tests for PROBLEM.md primary gate evaluation."""

from __future__ import annotations

from sparc.harness.gates import evaluate_primary_gate


def _suite(sparc_q: float, pebble_q: float, sparc_r: float, pebble_r: float) -> dict:
    return {
        "env_id": "walker-walk",
        "comparisons": {
            "sparc_vs_pebble": {
                "mean_queries_a": pebble_q,
                "mean_queries_b": sparc_q,
                "mean_r_inf_a": pebble_r,
                "mean_r_inf_b": sparc_r,
                "n_seeds_a": 10,
                "n_seeds_b": 10,
                "welch_p": 0.1,
            }
        },
    }


def test_gate_passes_when_criteria_met() -> None:
    gate = evaluate_primary_gate(_suite(150, 1000, 900, 850))
    assert gate is not None
    assert gate.query_budget_pass is True
    assert gate.non_inferiority_pass is True
    assert gate.overall_pass is True


def test_gate_fails_query_budget() -> None:
    gate = evaluate_primary_gate(_suite(250, 1000, 900, 850))
    assert gate is not None
    assert gate.query_budget_pass is False


def test_gate_passes_smoke_scale_low_pebble_queries() -> None:
    """At smoke scale PEBBLE≪1000, only absolute N_SPARC≤200 applies."""
    gate = evaluate_primary_gate(_suite(12, 16, -0.3, -22.0))
    assert gate is not None
    assert gate.query_budget_pass is True
