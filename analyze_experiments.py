#!/usr/bin/env python3
"""
analyze_experiments.py
-----------------------
CLI entry point for the HARMONY handover post-processing tool.

Example:
    python analyze_experiments.py /path/to/HARMONY_Experiments
    python analyze_experiments.py /path/to/HARMONY_Experiments --output ./analysis_output
    python analyze_experiments.py /path/to/HARMONY_Experiments --group-by harmony_version direction scenario
    python analyze_experiments.py /path/to/HARMONY_Experiments --exclude-invalid
    python analyze_experiments.py /path/to/HARMONY_Experiments --config my_config.yaml
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from harmony_analysis.pipeline import run_analysis


def _progress_printer(msg: str, cur: int, total: int) -> None:
    if total:
        print(f"[{cur}/{total}] {msg}")
    else:
        print(msg)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Recursively analyze HARMONY indoor/outdoor handover experiment logs."
    )
    parser.add_argument("root", type=str, help="Root folder to scan recursively (e.g. HARMONY_Experiments/)")
    parser.add_argument("--output", type=str, default="analysis_output",
                         help="Output folder (default: ./analysis_output)")
    parser.add_argument("--recursive", action=argparse.BooleanOptionalAction, default=True,
                         help="Recursively scan subfolders (default: True)")
    parser.add_argument("--exclude-invalid", action="store_true",
                         help="Exclude INVALID trials from aggregate_results.csv / plots "
                              "(they still appear in per_trial_metrics.csv and validation_report.csv)")
    parser.add_argument("--group-by", nargs="+", default=None,
                         help="Override aggregation group-by columns, e.g. --group-by harmony_version direction scenario")
    parser.add_argument("--config", type=str, default=None,
                         help="Path to a custom config.yaml (default: bundled config.yaml)")
    args = parser.parse_args(argv)

    root = Path(args.root)
    if not root.exists():
        print(f"ERROR: root folder not found: {root}", file=sys.stderr)
        return 1

    result = run_analysis(
        root=root,
        output_dir=args.output,
        config_path=args.config,
        recursive=args.recursive,
        exclude_invalid=args.exclude_invalid,
        group_by=args.group_by,
        progress_cb=_progress_printer,
    )

    print()
    print("=" * 60)
    print(f"Trials found      : {result.n_trials_found}")
    print(f"  VALID           : {result.n_valid}")
    print(f"  VALID_W_WARNINGS: {result.n_warnings}")
    print(f"  INVALID         : {result.n_invalid}")
    print(f"Output written to : {Path(result.output_dir).resolve()}")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
