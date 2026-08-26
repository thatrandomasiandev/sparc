"""Week 19 end-to-end harness dry-run suite (all methods, one env)."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from sparc.baselines.config import BaselineConfig
from sparc.engine.config import TrainConfig
from sparc.harness.compare import ComparisonReport, compare_summaries
from sparc.harness.export_curves import export_learning_curves
from sparc.harness.gates import evaluate_primary_gate
from sparc.harness.multi_seed import MultiSeedResult, run_multi_seed_baseline
from sparc.harness.report import write_suite_report
from sparc.harness.seeds import parse_seeds
from sparc.harness.stress_defaults import apply_stress_defaults
from sparc.harness.stress_verify import verify_stress_from_jsonl


@dataclass(frozen=True)
class SuiteManifest:
    """JSON manifest listing method → config path for a harness dry run."""

    suite_name: str
    env_id: str
    methods: dict[str, str]
    seeds: list[int] = field(default_factory=lambda: [0, 1])
    reference_method: str = "pebble"
    proposed_method: str = "sparc"

    @classmethod
    def from_json_file(cls, path: Path | str) -> SuiteManifest:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            suite_name=str(data["suite_name"]),
            env_id=str(data["env_id"]),
            methods={str(k): str(v) for k, v in data["methods"].items()},
            seeds=[int(s) for s in data.get("seeds", [0, 1])],
            reference_method=str(data.get("reference_method", "pebble")),
            proposed_method=str(data.get("proposed_method", "sparc")),
        )


@dataclass
class MethodSuiteEntry:
    method: str
    config_path: str
    output_dir: str
    summary: dict[str, Any]
    wall_clock_sec: float


@dataclass
class SuiteResult:
    suite_name: str
    env_id: str
    seeds: list[int]
    methods: list[MethodSuiteEntry] = field(default_factory=list)
    comparisons: dict[str, dict[str, Any]] = field(default_factory=dict)
    total_wall_clock_sec: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "suite_name": self.suite_name,
            "env_id": self.env_id,
            "seeds": self.seeds,
            "total_wall_clock_sec": self.total_wall_clock_sec,
            "methods": [asdict(m) for m in self.methods],
            "comparisons": self.comparisons,
        }


def _is_sparc_method(method: str) -> bool:
    """SPARC and Week 25 ablations (sparc_ablate_p1, …) share TrainConfig."""
    return method == "sparc" or method.startswith("sparc_")


def _load_method_config(method: str, config_path: Path) -> BaselineConfig | TrainConfig:
    if _is_sparc_method(method):
        return apply_stress_defaults(TrainConfig.from_json_file(config_path))
    return BaselineConfig.from_json_file(config_path)


def run_suite(
    manifest: SuiteManifest,
    output_dir: Path | str,
    seeds: list[int] | None = None,
) -> SuiteResult:
    """
    Run every method in the manifest under identical seeds and write suite artifacts.

    Per method: ``{output_dir}/{method}/summary.json``
    Suite rollup: ``{output_dir}/suite_summary.json``
    """
    out_root = Path(output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    seed_list = seeds if seeds is not None else manifest.seeds

    suite_t0 = time.perf_counter()
    entries: list[MethodSuiteEntry] = []

    for method, rel_config in manifest.methods.items():
        config_path = Path(rel_config)
        if not config_path.is_absolute():
            # Resolve relative to manifest location when possible
            config_path = config_path.resolve()
        method_dir = out_root / method
        cfg = _load_method_config(method, config_path)
        t0 = time.perf_counter()
        multi: MultiSeedResult = run_multi_seed_baseline(
            cfg,
            method=method,
            seeds=seed_list,
            output_dir=method_dir,
        )
        elapsed = time.perf_counter() - t0
        summary_path = method_dir / "summary.json"
        summary_data: dict[str, Any] = json.loads(summary_path.read_text(encoding="utf-8"))
        entries.append(
            MethodSuiteEntry(
                method=method,
                config_path=str(config_path),
                output_dir=str(method_dir),
                summary=summary_data,
                wall_clock_sec=elapsed,
            )
        )

    comparisons: dict[str, dict[str, Any]] = {}
    ref = manifest.reference_method
    prop = manifest.proposed_method
    ref_summary = out_root / ref / "summary.json"
    prop_summary = out_root / prop / "summary.json"
    if ref_summary.exists() and prop_summary.exists():
        report: ComparisonReport = compare_summaries(ref_summary, prop_summary)
        comparisons[f"{prop}_vs_{ref}"] = report.to_dict()

    primary_gate: dict[str, Any] | None = None
    stress_verify: dict[str, Any] | None = None

    total_elapsed = time.perf_counter() - suite_t0
    result = SuiteResult(
        suite_name=manifest.suite_name,
        env_id=manifest.env_id,
        seeds=seed_list,
        methods=entries,
        comparisons=comparisons,
        total_wall_clock_sec=total_elapsed,
    )
    suite_dict = result.to_dict()
    gate = evaluate_primary_gate(suite_dict)
    if gate is not None:
        primary_gate = gate.to_dict()
        suite_dict["primary_gate"] = primary_gate
    sparc_log = out_root / prop / "seed_0.jsonl"
    if sparc_log.exists():
        stress_verify = verify_stress_from_jsonl(sparc_log).to_dict()
        suite_dict["stress_verification"] = stress_verify

    summary_path = out_root / "suite_summary.json"
    summary_path.write_text(json.dumps(suite_dict, indent=2), encoding="utf-8")
    write_suite_report(summary_path)
    export_learning_curves(summary_path, out_root / "curves")
    return result


def run_suite_from_manifest(
    manifest_path: Path | str,
    output_dir: Path | str,
    seeds_spec: str | None = None,
) -> SuiteResult:
    manifest = SuiteManifest.from_json_file(manifest_path)
    seed_list = parse_seeds(seeds_spec, default=manifest.seeds)
    return run_suite(manifest, output_dir, seeds=seed_list)
