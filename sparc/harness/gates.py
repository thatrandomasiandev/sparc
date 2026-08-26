"""Primary success gate evaluation (PROBLEM.md / EXPERIMENTS.md)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sparc.harness.stress_defaults import N_PEBBLE_QUERIES, N_SPARC_QUERIES

DELTA_NI_WALKER = 50.0  # EXPERIMENTS.md non-inferiority margin (return points)


@dataclass(frozen=True)
class PrimaryGateResult:
    """Outcome of PROBLEM.md primary gate checks on one env comparison."""

    env_id: str
    n_seeds_sparc: int
    n_seeds_pebble: int
    sparc_mean_queries: float
    pebble_mean_queries: float
    query_budget_pass: bool
    sparc_mean_r_inf: float
    pebble_mean_r_inf: float
    return_delta: float
    non_inferiority_pass: bool
    delta_ni: float
    welch_p: float | None
    overall_pass: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "env_id": self.env_id,
            "n_seeds_sparc": self.n_seeds_sparc,
            "n_seeds_pebble": self.n_seeds_pebble,
            "sparc_mean_queries": self.sparc_mean_queries,
            "pebble_mean_queries": self.pebble_mean_queries,
            "query_budget_pass": self.query_budget_pass,
            "sparc_mean_r_inf": self.sparc_mean_r_inf,
            "pebble_mean_r_inf": self.pebble_mean_r_inf,
            "return_delta": self.return_delta,
            "non_inferiority_pass": self.non_inferiority_pass,
            "delta_ni": self.delta_ni,
            "welch_p": self.welch_p,
            "overall_pass": self.overall_pass,
        }


def evaluate_primary_gate(
    suite_summary: dict[str, Any],
    *,
    reference_method: str = "pebble",
    proposed_method: str = "sparc",
    max_sparc_queries: int = N_SPARC_QUERIES,
    delta_ni: float = DELTA_NI_WALKER,
) -> PrimaryGateResult | None:
    """
    Evaluate sample-efficiency and return non-inferiority from a suite summary.

    Stress-condition activation is config/log provenance — not inferred here.
    """
    comp_key = f"{proposed_method}_vs_{reference_method}"
    comp = suite_summary.get("comparisons", {}).get(comp_key)
    if not comp:
        return None

    sparc_q = float(comp["mean_queries_b"])
    pebble_q = float(comp["mean_queries_a"])
    sparc_r = float(comp["mean_r_inf_b"])
    pebble_r = float(comp["mean_r_inf_a"])
    query_pass = sparc_q <= max_sparc_queries
    if pebble_q >= N_PEBBLE_QUERIES * 0.5:
        query_pass = query_pass and sparc_q <= 0.2 * pebble_q
    return_delta = sparc_r - pebble_r
    ni_pass = return_delta >= -delta_ni
    welch_p = comp.get("welch_p")
    p_val = float(welch_p) if welch_p is not None and welch_p == welch_p else None

    return PrimaryGateResult(
        env_id=str(suite_summary.get("env_id", "")),
        n_seeds_sparc=int(comp.get("n_seeds_b", 0)),
        n_seeds_pebble=int(comp.get("n_seeds_a", 0)),
        sparc_mean_queries=sparc_q,
        pebble_mean_queries=pebble_q,
        query_budget_pass=query_pass,
        sparc_mean_r_inf=sparc_r,
        pebble_mean_r_inf=pebble_r,
        return_delta=return_delta,
        non_inferiority_pass=ni_pass,
        delta_ni=delta_ni,
        welch_p=p_val,
        overall_pass=query_pass and ni_pass,
    )


def evaluate_primary_gate_from_file(path: Path | str) -> PrimaryGateResult | None:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return evaluate_primary_gate(data)


def gate_summary_markdown(gate: PrimaryGateResult) -> list[str]:
    q_icon = "PASS" if gate.query_budget_pass else "FAIL"
    r_icon = "PASS" if gate.non_inferiority_pass else "FAIL"
    o_icon = "PASS" if gate.overall_pass else "FAIL"
    return [
        "## Primary gate (PROBLEM.md)",
        "",
        f"| Check | Result | Detail |",
        f"|-------|--------|--------|",
        f"| Query budget (≤{N_SPARC_QUERIES}, ≤20% PEBBLE) | **{q_icon}** | "
        f"SPARC {gate.sparc_mean_queries:.0f} vs PEBBLE {gate.pebble_mean_queries:.0f} |",
        f"| Return non-inferiority (Δ ≥ −{gate.delta_ni:.0f}) | **{r_icon}** | "
        f"Δ={gate.return_delta:.1f} (SPARC {gate.sparc_mean_r_inf:.1f} vs PEBBLE {gate.pebble_mean_r_inf:.1f}) |",
        f"| **Overall** | **{o_icon}** | {gate.n_seeds_sparc} SPARC / {gate.n_seeds_pebble} PEBBLE seeds |",
        "",
        "_Stress conditions (K=3, ΔT, drift) verified by config + JSONL — not auto-checked here._",
        "",
    ]
