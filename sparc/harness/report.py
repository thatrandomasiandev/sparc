"""Generate reports and learning curves from harness run artifacts (Week 18/19)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from sparc.harness.gates import evaluate_primary_gate, gate_summary_markdown
from sparc.harness.stress_verify import StressVerification, verify_stress_from_jsonl


@dataclass(frozen=True)
class MethodRow:
    method: str
    n_seeds: int
    mean_r_inf: float
    ci95_low: float
    ci95_high: float
    mean_true_r_inf: float
    mean_queries: float
    mean_wall_sec: float


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_method_row(summary_path: Path) -> MethodRow:
    data = _load_json(summary_path)
    s = data["summary"]
    return MethodRow(
        method=str(data.get("method", summary_path.parent.name)),
        n_seeds=int(s["n_seeds"]),
        mean_r_inf=float(s["mean_asymptotic_return"]),
        ci95_low=float(s["ci95_low"]),
        ci95_high=float(s["ci95_high"]),
        mean_true_r_inf=float(s["mean_asymptotic_true_return"]),
        mean_queries=float(s["mean_total_queries"]),
        mean_wall_sec=float(s["mean_wall_clock_sec"]),
    )


def learning_curve_points(run: dict[str, Any]) -> list[tuple[int, float, int]]:
    """Return (env_steps, mean_return, total_queries) tuples from one run."""
    out: list[tuple[int, float, int]] = []
    for pt in run.get("eval_history", []):
        out.append(
            (
                int(pt["env_steps"]),
                float(pt["mean_return"]),
                int(pt["total_queries"]),
            )
        )
    return out


def suite_table_markdown(
    suite_summary_path: Path,
    stress_verifications: dict[str, StressVerification] | None = None,
) -> str:
    """Render a markdown comparison table from ``suite_summary.json``."""
    suite = _load_json(suite_summary_path)
    rows: list[MethodRow] = []
    for entry in suite.get("methods", []):
        summary = entry.get("summary", {})
        if not summary:
            continue
        s = summary["summary"]
        rows.append(
            MethodRow(
                method=str(summary.get("method", entry.get("method", "?"))),
                n_seeds=int(s["n_seeds"]),
                mean_r_inf=float(s["mean_asymptotic_return"]),
                ci95_low=float(s["ci95_low"]),
                ci95_high=float(s["ci95_high"]),
                mean_true_r_inf=float(s["mean_asymptotic_true_return"]),
                mean_queries=float(s["mean_total_queries"]),
                mean_wall_sec=float(s["mean_wall_clock_sec"]),
            )
        )
    rows.sort(key=lambda r: r.method)

    lines = [
        f"# Harness suite: {suite.get('suite_name', 'unknown')}",
        "",
        f"**Env:** `{suite.get('env_id', '')}` · **Seeds:** {suite.get('seeds', [])}",
        "",
        "| Method | Seeds | R_∞ (learned) | 95% CI | True R_∞ | Queries | Wall (s) |",
        "|--------|-------|---------------|--------|------------|---------|----------|",
    ]
    for r in rows:
        ci = f"[{r.ci95_low:.1f}, {r.ci95_high:.1f}]"
        lines.append(
            f"| {r.method} | {r.n_seeds} | {r.mean_r_inf:.1f} | {ci} | "
            f"{r.mean_true_r_inf:.1f} | {r.mean_queries:.1f} | {r.mean_wall_sec:.0f} |"
        )

    comparisons = suite.get("comparisons", {})
    if comparisons:
        lines.extend(["", "## Primary comparison", ""])
        for name, comp in comparisons.items():
            lines.append(f"### {name}")
            lines.append("")
            lines.append(
                f"- **R_∞:** {comp.get('mean_r_inf_b', 0):.1f} vs "
                f"{comp.get('mean_r_inf_a', 0):.1f} (Welch p={comp.get('welch_p', float('nan')):.4f})"
            )
            lines.append(
                f"- **Queries:** {comp.get('mean_queries_b', 0):.1f} vs "
                f"{comp.get('mean_queries_a', 0):.1f} "
                f"({comp.get('query_reduction_pct', 0):.0f}% reduction)"
            )
            lines.append("")

    gate = evaluate_primary_gate(suite)
    if gate is not None:
        lines.extend(gate_summary_markdown(gate))

    if stress_verifications:
        lines.extend(["", "## Stress regime verification (SPARC JSONL)", ""])
        for method, ver in stress_verifications.items():
            status = "PASS" if ver.stress_active else "FAIL"
            lines.append(
                f"- **{method}:** {status} — {ver.n_windows} windows, "
                f"ops={sorted(ver.operator_ids)}, "
                f"scripted_drift={ver.scripted_drift_events}, "
                f"sprt_triggers={ver.sprt_drift_triggers}"
            )
        lines.append("")

    return "\n".join(lines) + "\n"


def _find_sparc_jsonl_logs(suite_dir: Path) -> dict[str, Path]:
    """Map method name → first seed JSONL log under suite output."""
    out: dict[str, Path] = {}
    sparc_dir = suite_dir / "sparc"
    if not sparc_dir.is_dir():
        return out
    for jsonl in sorted(sparc_dir.glob("seed_*.jsonl")):
        out["sparc"] = jsonl
        break
    return out


def write_suite_report(suite_summary_path: Path, output_path: Path | None = None) -> Path:
    """Write markdown report alongside ``suite_summary.json``."""
    src = Path(suite_summary_path)
    dst = output_path or src.parent / "SUITE_REPORT.md"
    stress: dict[str, StressVerification] = {}
    for method, log_path in _find_sparc_jsonl_logs(src.parent).items():
        if log_path.exists():
            stress[method] = verify_stress_from_jsonl(log_path)
    dst.write_text(suite_table_markdown(src, stress_verifications=stress or None), encoding="utf-8")
    return dst
