"""
pipeline.py
-----------
Top-level orchestration of the full analysis pipeline. This module has
NO Tkinter / argparse dependency so it can be unit-tested and reused by
both analyze_experiments.py (CLI) and gui.py.
"""
from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional

import pandas as pd
import yaml

from .aggregate import aggregate_trials, build_paper_metrics
from .discovery import scan_root
from .export import write_csvs, write_excel_report
from .metrics import compute_trial_metrics
from .pairing import TrialFiles, pair_trials
from .plotting import generate_all_plots

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.yaml"


@dataclass
class PipelineResult:
    per_trial: pd.DataFrame
    aggregate: pd.DataFrame
    paper_metrics: pd.DataFrame
    validation_report: pd.DataFrame
    event_diagnostics: pd.DataFrame
    n_trials_found: int = 0
    n_valid: int = 0
    n_warnings: int = 0
    n_invalid: int = 0
    output_dir: Optional[Path] = None


def load_config(config_path: Optional[str | Path] = None) -> dict:
    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _build_validation_report(per_trial: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in ["session_id", "harmony_version", "direction", "validation_status", "validation_notes"]
            if c in per_trial.columns]
    return per_trial[cols].copy() if cols else pd.DataFrame()


def _build_event_diagnostics(trials: Dict[str, TrialFiles]) -> pd.DataFrame:
    """Per-trial inventory of unique event names / states / sources seen,
    useful for auditing whether the taxonomy in config.yaml still matches
    a new batch of logs (spec section 13)."""
    rows = []
    for sid, trial in trials.items():
        if trial.events is None or trial.events.df is None:
            continue
        df = trial.events.df
        rows.append({
            "session_id": sid,
            "unique_events": ", ".join(sorted(df["event"].dropna().unique().astype(str))) if "event" in df else "",
            "unique_from_states": ", ".join(sorted(df["from_state"].dropna().unique().astype(str))) if "from_state" in df else "",
            "unique_to_states": ", ".join(sorted(df["to_state"].dropna().unique().astype(str))) if "to_state" in df else "",
            "unique_sources": ", ".join(sorted(df["source"].dropna().unique().astype(str))) if "source" in df else "",
            "n_events": len(df),
        })
    return pd.DataFrame(rows)


def run_analysis(root: str | Path,
                  output_dir: str | Path,
                  config_path: Optional[str | Path] = None,
                  recursive: bool = True,
                  exclude_invalid: bool = False,
                  group_by: Optional[List[str]] = None,
                  progress_cb: Optional[Callable[[str, int, int], None]] = None) -> PipelineResult:
    """Run the full pipeline end-to-end.

    Parameters
    ----------
    progress_cb : callable(message, current, total), optional
        Called periodically so a GUI can show a progress bar.
    """
    cfg = load_config(config_path)
    if group_by:
        cfg["aggregate"]["group_by"] = group_by

    def report(msg: str, cur: int = 0, total: int = 0):
        if progress_cb:
            progress_cb(msg, cur, total)

    report("Scanning folder tree...", 0, 1)
    discovered = scan_root(
        root,
        events_glob=cfg["discovery"]["events_glob"],
        samples_glob=cfg["discovery"]["samples_glob"],
        summary_glob=cfg["discovery"]["summary_glob"],
        recursive=recursive,
    )
    report(f"Found {discovered.total()} candidate CSV files. Pairing trials...", 0, 1)

    trials = pair_trials(discovered)
    total = len(trials)
    report(f"Paired into {total} trial(s). Computing metrics...", 0, total)

    rows = []
    for i, (sid, trial) in enumerate(trials.items(), start=1):
        row = compute_trial_metrics(trial, cfg)
        rows.append(row)
        report(f"Analyzed trial {i}/{total}: {sid}", i, total)

    per_trial = pd.DataFrame(rows)

    n_valid = int((per_trial["validation_status"] == "VALID").sum()) if "validation_status" in per_trial else 0
    n_warn = int((per_trial["validation_status"] == "VALID_WITH_WARNINGS").sum()) if "validation_status" in per_trial else 0
    n_invalid = int((per_trial["validation_status"] == "INVALID").sum()) if "validation_status" in per_trial else 0

    report("Aggregating results...", total, total)
    aggregate = aggregate_trials(per_trial, cfg, exclude_invalid=exclude_invalid) if len(per_trial) else pd.DataFrame()
    paper = build_paper_metrics(aggregate, cfg) if len(aggregate) else pd.DataFrame()
    validation_report = _build_validation_report(per_trial)
    event_diag = _build_event_diagnostics(trials)

    output_dir = Path(output_dir)
    report("Writing CSV outputs...", total, total)
    write_csvs(output_dir, per_trial, aggregate, paper, validation_report, event_diag)

    readme_lines = _readme_lines(cfg, per_trial, n_valid, n_warn, n_invalid)
    report("Writing Excel report...", total, total)
    write_excel_report(output_dir / "report_summary.xlsx", per_trial, aggregate, paper,
                        validation_report, readme_lines)

    report("Generating plots...", total, total)
    generate_all_plots(per_trial, aggregate, output_dir / "plots", cfg)

    report("Done.", total, total)

    return PipelineResult(
        per_trial=per_trial, aggregate=aggregate, paper_metrics=paper,
        validation_report=validation_report, event_diagnostics=event_diag,
        n_trials_found=total, n_valid=n_valid, n_warnings=n_warn, n_invalid=n_invalid,
        output_dir=output_dir,
    )


def _readme_lines(cfg: dict, per_trial: pd.DataFrame, n_valid: int, n_warn: int, n_invalid: int) -> list[str]:
    now = _dt.datetime.now().isoformat(timespec="seconds")
    lines = [
        "HARMONY Handover Analysis — report_summary.xlsx",
        f"Generated: {now}",
        "",
        f"Trials found: {len(per_trial)}  |  VALID: {n_valid}  |  VALID_WITH_WARNINGS: {n_warn}  |  INVALID: {n_invalid}",
        "",
        "Sheets:",
        " - Per Trial: one row per trial, all reconstructed KPIs + metadata (invalid rows highlighted red, warnings amber).",
        " - Aggregate: grouped by harmony_version x direction [x scenario], with N/mean/median/SD/Q1/Q3/min/max/95% CI.",
        " - Paper Metrics: the clean table for direct use in the paper (HSR, FHR, latency, source switching, ...).",
        " - Validation: validation_status + validation_notes per trial (invalid rows highlighted red).",
        " - Source Transitions / GPS Statistics / PDR Statistics / VPS Statistics: metric subsets by prefix.",
        "",
        "IMPORTANT — metrics reconstructed from events_*.csv + samples_*.csv, NOT taken blindly from summary_*.csv:",
        "  handover_success, false_handovers, hsr_percent, fhr_percent, source_toggle_count, "
        "position_jump_m, heading_jump_deg.",
        "Summary.csv values for these are kept side-by-side (suffix _summary) for cross-checking only.",
        "Where reconstructed != summary, validation_notes flags [METRIC_MISMATCH] and the reconstructed value is authoritative.",
        "",
        "Key handover-detection rule (GPS_TO_VPS): a VPS handover is counted as SUCCESS only if the FSM reaches",
        "IndoorVps/IndoorLocalized via a genuine vps_state transition, NOT via the fallback events",
        "(pdr_approximate_localization / handover_completed_approximate / pdr_approximate_handover_completed).",
        "A VPS timeout that falls back to PDR-approximate pose is scored as a FAILURE even though the FSM's",
        "to_state is 'IndoorVps'. See config.yaml -> events.vps_fallback_events to adjust.",
        "",
        "Metrics that could not be computed with confidence from the current logger are left blank/NaN with",
        "a note containing 'NOT_EVALUABLE_FROM_CURRENT_LOG' — never invented, never silently zeroed.",
    ]
    return lines
