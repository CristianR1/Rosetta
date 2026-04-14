"""
run_evaluation.py - Coordination script for running all three pipeline systems.

Collects shared parameters and runs Lotus, DocETL, and Palimpzest
pipelines sequentially with identical configuration.

Usage:
    python run_evaluation.py --null-param null_binary_explicit --noise-param 1_data_1_noise --ops filter join --limit 10
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

from rosetta_env import REPO_ROOT, evaluation_results_dir

_DEMO_DIR = Path(__file__).resolve().parent

PIPELINES = [
    ("lotus",       "lotus_pipeline.py"),
    ("docetl",      "docetl_pipeline.py"),
    ("palimpzest",  "palimpzest_pipeline.py"),
]


def _build_child_args(args: argparse.Namespace) -> list[str]:
    """Build the common CLI arguments forwarded to each pipeline script."""
    child = ["--semantic"]

    if args.null_param:
        child += ["--null-param", args.null_param]
    if args.noise_param:
        child += ["--noise-param", args.noise_param]
    if args.ops:
        child += ["--ops"] + args.ops
    if args.limit > 0:
        child += ["--limit", str(args.limit)]
    if args.num_documents != 10:
        child += ["--num_documents", str(args.num_documents)]

    return child


def _resolve_data_root(args: argparse.Namespace) -> Path:
    if args.null_param and args.noise_param:
        return _DEMO_DIR / "pipeline_data" / args.null_param / args.noise_param / "Text"
    elif args.null_param:
        return _DEMO_DIR / "pipeline_data" / args.null_param / "Text"
    return _DEMO_DIR / "pipeline_data" / "data"


def main():
    parser = argparse.ArgumentParser(
        description="Run Lotus, DocETL, and Palimpzest evaluations with shared parameters",
    )
    parser.add_argument(
        "--null-param", required=True,
        help="Null representation directory under pipeline_data/ (e.g. null_binary_explicit, null_binary_implicit)",
    )
    parser.add_argument(
        "--noise-param", default=None,
        help="Noise ratio directory (e.g. 1_data_1_noise, 1_data_0_noise). Omit if the null dir has no noise subdirectory.",
    )
    parser.add_argument(
        "--ops", nargs="*", default=[],
        help="Only run entries containing these ops: filter, join, group, agg, extract, topk",
    )
    parser.add_argument(
        "--limit", type=int, default=0,
        help="Max number of questions to run per system (0 = all matching)",
    )
    parser.add_argument(
        "--num_documents", type=int, default=10,
        help="Limit each table's SQL to N rows for ground-truth alignment",
    )
    parser.add_argument(
        "--systems", nargs="*", default=[],
        help="Subset of systems to run: lotus, docetl, palimpzest (default: all)",
    )
    args = parser.parse_args()

    data_root = _resolve_data_root(args)
    if not data_root.exists():
        print(f"[ERROR] Data root does not exist: {data_root}")
        sys.exit(1)

    child_args = _build_child_args(args)

    requested = {s.lower() for s in args.systems} if args.systems else None

    print("=" * 70)
    print("SDPS Evaluation Run")
    print("=" * 70)
    print(f"  null_param   : {args.null_param}")
    print(f"  noise_param  : {args.noise_param or '(none)'}")
    print(f"  data_root    : {data_root}")
    print(f"  ops filter   : {', '.join(args.ops) if args.ops else '(all)'}")
    print(f"  limit        : {args.limit if args.limit else '(all)'}")
    print(f"  num_documents: {args.num_documents}")
    print(f"  systems      : {', '.join(requested) if requested else 'lotus, docetl, palimpzest'}")
    print(f"  results_root : {REPO_ROOT / 'Results'}")
    for sys_name, _script in PIPELINES:
        if requested and sys_name not in requested:
            continue
        rd = evaluation_results_dir(
            sys_name, args.null_param, args.noise_param, args.num_documents
        )
        print(f"    → {sys_name}: {rd}")
    print("=" * 70)
    print()

    results: dict[str, dict] = {}

    for name, script in PIPELINES:
        if requested and name not in requested:
            continue

        script_path = _DEMO_DIR / script
        if not script_path.exists():
            print(f"[WARN] Script not found: {script_path} — skipping {name}")
            continue

        print(f"\n{'-' * 70}")
        print(f"  Running: {name}")
        print(f"{'-' * 70}\n")

        cmd = [sys.executable, "-u", str(script_path)] + child_args
        t0 = time.perf_counter()
        proc = subprocess.run(cmd, cwd=str(_DEMO_DIR))
        elapsed = time.perf_counter() - t0

        results[name] = {
            "returncode": proc.returncode,
            "elapsed_seconds": round(elapsed, 2),
        }

        status = "OK" if proc.returncode == 0 else f"FAILED (exit {proc.returncode})"
        print(f"\n  [{name}] {status} in {elapsed:.1f}s")

    print(f"\n{'=' * 70}")
    print("Summary")
    print(f"{'=' * 70}")
    for name, info in results.items():
        tag = "PASS" if info["returncode"] == 0 else "FAIL"
        print(f"  {name:15s}  {tag}  {info['elapsed_seconds']:>8.1f}s")
    print()


if __name__ == "__main__":
    main()
