"""SPARC CLI — `sparc train --config ...` (ROADMAP Week 14)."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

from sparc.baselines.config import BaselineConfig
from sparc.engine import SparcTrainer, TrainConfig
from sparc.harness.compare import compare_summaries
from sparc.harness.multi_seed import run_multi_seed_baseline
from sparc.harness.runner import run_baseline
from sparc.harness.seeds import parse_seeds
from sparc.harness.dry_run import run_suite_from_manifest
from sparc.harness.export_curves import export_learning_curves
from sparc.harness.gates import evaluate_primary_gate_from_file
from sparc.harness.log_metrics import extract_log_metrics
from sparc.harness.report import write_suite_report
from sparc.harness.stress_defaults import apply_stress_defaults


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="sparc", description="SPARC preference-based RL engine")
    sub = parser.add_subparsers(dest="command", required=True)

    train = sub.add_parser("train", help="Run integrated SPARC training loop")
    train.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to JSON training config (TrainConfig)",
    )
    train.add_argument(
        "--log",
        type=Path,
        default=None,
        help="JSONL log path (overrides config.log_path)",
    )

    benchmark = sub.add_parser("benchmark", help="Run PEBBLE / PrefPPO baseline (Phase 3)")
    benchmark.add_argument(
        "--method",
        choices=["pebble", "prefppo", "surf", "rune", "aprel", "sparc"],
        required=True,
        help="Method: pebble, prefppo, surf, rune, aprel, or sparc",
    )
    benchmark.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to JSON baseline config (BaselineConfig)",
    )
    benchmark.add_argument(
        "--log",
        type=Path,
        default=None,
        help="JSONL log path for single-seed run (overrides config.log_path)",
    )
    benchmark.add_argument(
        "--seeds",
        type=str,
        default=None,
        help="Comma-separated seeds or 'all' for EXPERIMENTS.md master list (0-9)",
    )
    benchmark.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for multi-seed summary.json and per-seed logs",
    )

    compare = sub.add_parser("compare", help="Compare two benchmark summary.json files")
    compare.add_argument("--a", type=Path, required=True, help="First summary.json (e.g. PEBBLE)")
    compare.add_argument("--b", type=Path, required=True, help="Second summary.json (e.g. SPARC)")

    suite = sub.add_parser("suite", help="Run Week 19 harness dry-run (all methods)")
    suite.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help="JSON suite manifest (method → config paths)",
    )
    suite.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for per-method summaries and suite_summary.json",
    )
    suite.add_argument(
        "--seeds",
        type=str,
        default=None,
        help="Override manifest seeds (comma-separated or 'all')",
    )

    report = sub.add_parser("report", help="Generate markdown report from suite_summary.json")
    report.add_argument(
        "--suite-summary",
        type=Path,
        required=True,
        help="Path to suite_summary.json",
    )
    report.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output markdown path (default: SUITE_REPORT.md alongside summary)",
    )

    metrics = sub.add_parser("metrics", help="Extract EXPERIMENTS.md metrics from JSONL log")
    metrics.add_argument("--log", type=Path, required=True, help="Path to JSONL training log")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "train":
        config = apply_stress_defaults(TrainConfig.from_json_file(args.config))
        if args.log is not None:
            config = replace(config, log_path=str(args.log))
        trainer = SparcTrainer(config)
        result = trainer.run()
        print(
            f"SPARC train complete: {result.windows_completed} windows, "
            f"{result.total_queries} queries, {result.wall_clock_sec:.2f}s",
            file=sys.stderr,
        )
        return 0

    if args.command == "benchmark":
        bench_cfg: BaselineConfig | TrainConfig
        if args.method == "sparc":
            bench_cfg = apply_stress_defaults(TrainConfig.from_json_file(args.config))
        else:
            bench_cfg = BaselineConfig.from_json_file(args.config)
        seed_list = parse_seeds(args.seeds, default=[bench_cfg.seed])
        if len(seed_list) > 1 or args.output_dir is not None:
            multi = run_multi_seed_baseline(
                bench_cfg,
                method=args.method,
                seeds=seed_list,
                output_dir=args.output_dir,
            )
            assert multi.summary is not None
            print(
                f"SPARC {args.method} multi-seed: {multi.summary.n_seeds} seeds, "
                f"R_inf={multi.summary.mean_asymptotic_return:.1f} "
                f"[{multi.summary.ci95_low:.1f}, {multi.summary.ci95_high:.1f}], "
                f"queries={multi.summary.mean_total_queries:.1f}, "
                f"{multi.total_wall_clock_sec:.1f}s total",
                file=sys.stderr,
            )
            return 0
        if args.log is not None:
            bench_cfg = replace(bench_cfg, log_path=str(args.log))
        bench_result = run_baseline(bench_cfg, method=args.method)
        print(
            f"SPARC {args.method} complete: {bench_result.total_env_steps} steps, "
            f"{bench_result.total_queries} queries, "
            f"R_inf={bench_result.asymptotic_return:.1f}, "
            f"{bench_result.wall_clock_sec:.2f}s",
            file=sys.stderr,
        )
        return 0

    if args.command == "compare":
        report = compare_summaries(args.a, args.b)
        print(json.dumps(report.to_dict(), indent=2))
        return 0

    if args.command == "suite":
        result = run_suite_from_manifest(
            args.manifest,
            args.output_dir,
            seeds_spec=args.seeds,
        )
        n_methods = len(result.methods)
        sparc_vs = result.comparisons.get("sparc_vs_pebble", {})
        print(
            f"SPARC suite {result.suite_name}: {n_methods} methods, "
            f"{len(result.seeds)} seeds, {result.total_wall_clock_sec:.1f}s total",
            file=sys.stderr,
        )
        if sparc_vs:
            print(
                f"  sparc vs pebble: R_inf {sparc_vs.get('mean_r_inf_b', 0):.1f} vs "
                f"{sparc_vs.get('mean_r_inf_a', 0):.1f}, "
                f"queries {sparc_vs.get('mean_queries_b', 0):.1f} vs "
                f"{sparc_vs.get('mean_queries_a', 0):.1f} "
                f"({sparc_vs.get('query_reduction_pct', 0):.0f}% reduction)",
                file=sys.stderr,
            )
        report_path = write_suite_report(args.output_dir / "suite_summary.json")
        curves = export_learning_curves(args.output_dir / "suite_summary.json", args.output_dir / "curves")
        gate = evaluate_primary_gate_from_file(args.output_dir / "suite_summary.json")
        print(f"Report written: {report_path}", file=sys.stderr)
        if curves:
            print(f"Learning curves: {len(curves)} CSV(s) in {args.output_dir / 'curves'}", file=sys.stderr)
        if gate is not None:
            print(
                f"Primary gate: {'PASS' if gate.overall_pass else 'FAIL'} "
                f"(queries={gate.sparc_mean_queries:.0f}, ΔR={gate.return_delta:.1f})",
                file=sys.stderr,
            )
        return 0

    if args.command == "report":
        out = write_suite_report(args.suite_summary, args.output)
        print(str(out))
        return 0

    if args.command == "metrics":
        m = extract_log_metrics(args.log)
        print(
            json.dumps(
                {
                    "n_windows": m.n_windows,
                    "total_queries": m.total_queries,
                    "mean_wall_clock_per_query_sec": m.mean_wall_clock_per_query_sec,
                    "mean_select_sec": m.mean_select_sec,
                },
                indent=2,
            )
        )
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
