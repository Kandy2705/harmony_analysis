"""
validation.py
--------------
Per-trial data-quality validation. Produces a `validation_status` in
{VALID, VALID_WITH_WARNINGS, INVALID, NOT_EVALUABLE} plus a list of
human-readable `validation_notes`. Per spec section 5, invalid trials
are NEVER silently dropped — they still appear in per_trial_metrics.csv
and validation_report.csv with their status and notes.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import pandas as pd

from .pairing import TrialFiles


@dataclass
class ValidationResult:
    status: str  # VALID | VALID_WITH_WARNINGS | INVALID | NOT_EVALUABLE
    notes: List[str] = field(default_factory=list)

    def add(self, note: str) -> None:
        self.notes.append(note)

    def notes_str(self) -> str:
        return " | ".join(self.notes)


def validate_trial(trial: TrialFiles, cfg: dict) -> ValidationResult:
    errors: List[str] = []
    warnings: List[str] = []

    # --- pairing-level issues collected during loading/pairing ---
    for issue in trial.pairing_issues:
        target = errors if issue.severity == "ERROR" else warnings
        target.append(f"[{issue.code}] {issue.message}")

    # --- missing files ---
    if trial.events is None:
        errors.append("[MISSING_EVENTS_FILE] No events_*.csv found for this session_id.")
    if trial.samples is None:
        errors.append("[MISSING_SAMPLES_FILE] No samples_*.csv found for this session_id.")
    if trial.summary is None:
        warnings.append("[MISSING_SUMMARY_FILE] No summary_*.csv found; cross-check skipped.")

    if trial.duplicate_events:
        warnings.append(f"[DUPLICATE_EVENTS_FILE] {len(trial.duplicate_events)} extra events file(s) ignored.")
    if trial.duplicate_samples:
        warnings.append(f"[DUPLICATE_SAMPLES_FILE] {len(trial.duplicate_samples)} extra samples file(s) ignored.")
    if trial.duplicate_summary:
        warnings.append(f"[DUPLICATE_SUMMARY_FILE] {len(trial.duplicate_summary)} extra summary file(s) ignored.")

    # If events or samples are fundamentally missing/corrupted, we cannot go further.
    if trial.events is None or not trial.events.ok:
        if trial.events is not None:
            for issue in trial.events.issues:
                (errors if issue.severity == "ERROR" else warnings).append(f"[{issue.code}] {issue.message}")
        return ValidationResult(status="INVALID", notes=errors + warnings)

    if trial.samples is None or not trial.samples.ok:
        if trial.samples is not None:
            for issue in trial.samples.issues:
                (errors if issue.severity == "ERROR" else warnings).append(f"[{issue.code}] {issue.message}")
        return ValidationResult(status="INVALID", notes=errors + warnings)

    ev_df = trial.events.df
    sm_df = trial.samples.df

    min_events = cfg["validation"]["min_events"]
    min_samples = cfg["validation"]["min_samples"]
    if len(ev_df) < min_events:
        errors.append(f"[TOO_FEW_EVENTS] Only {len(ev_df)} event rows (< {min_events}).")
    if len(sm_df) < min_samples:
        errors.append(f"[TOO_FEW_SAMPLES] Only {len(sm_df)} sample rows (< {min_samples}).")

    # --- monotonic elapsed_s ---
    if cfg["validation"].get("require_monotonic_elapsed", True):
        for name, df in (("events", ev_df), ("samples", sm_df)):
            if "elapsed_s" in df.columns:
                elapsed = df["elapsed_s"].dropna().to_numpy()
                if len(elapsed) > 1 and np.any(np.diff(elapsed) < 0):
                    warnings.append(f"[NON_MONOTONIC_ELAPSED_S] {name} elapsed_s is not monotonically increasing.")

    # --- impossible timestamps ---
    for name, df in (("events", ev_df), ("samples", sm_df)):
        if "elapsed_s" in df.columns:
            neg = (df["elapsed_s"] < 0).sum()
            if neg > 0:
                warnings.append(f"[IMPOSSIBLE_TIMESTAMP] {neg} negative elapsed_s value(s) in {name}.")

    # --- large time gaps ---
    max_gap = cfg["validation"].get("max_allowed_gap_s")
    if max_gap and "elapsed_s" in ev_df.columns:
        elapsed = ev_df["elapsed_s"].dropna().sort_values().to_numpy()
        if len(elapsed) > 1:
            gaps = np.diff(elapsed)
            big_gaps = int(np.sum(gaps > max_gap))
            if big_gaps > 0:
                warnings.append(f"[LARGE_TIME_GAP] {big_gaps} gap(s) in events elapsed_s exceed {max_gap}s "
                                 f"(app pause/resume or logging interruption).")

    # --- session_id consistency across files ---
    sids = set()
    for loaded in (trial.events, trial.samples, trial.summary):
        if loaded is not None and loaded.df is not None and "session_id" in loaded.df.columns:
            vals = loaded.df["session_id"].dropna().unique().tolist()
            sids.update(str(v) for v in vals)
    if len(sids) > 1:
        errors.append(f"[MISMATCHED_SESSION_ID] Files disagree on session_id: {sorted(sids)}")

    # --- aborted / incomplete trial heuristics ---
    trial_started_ev = cfg["events"]["trial_started"]
    if trial_started_ev not in ev_df["event"].values:
        warnings.append("[NO_TRIAL_STARTED_EVENT] trial_started event missing.")

    if trial.summary is not None and trial.summary.df is not None and len(trial.summary.df) > 0:
        srow = trial.summary.df.iloc[0]
        completed = srow.get("completed")
        end_reason = srow.get("end_reason")
        if pd.isna(completed) or completed in (0, "0", False, "False"):
            if pd.isna(end_reason) or str(end_reason).strip() == "":
                warnings.append("[TRIAL_NOT_COMPLETED] summary.completed=0 and no end_reason logged; "
                                 "trial may have been aborted, reset, or app-closed mid-run.")
            else:
                warnings.append(f"[TRIAL_NOT_COMPLETED] summary.completed=0, end_reason={end_reason}.")

    status = "VALID"
    if errors:
        status = "INVALID"
    elif warnings:
        status = "VALID_WITH_WARNINGS"

    return ValidationResult(status=status, notes=errors + warnings)


def mark_not_evaluable(result: ValidationResult, reason: str) -> ValidationResult:
    """Downgrade a VALID/VALID_WITH_WARNINGS result to NOT_EVALUABLE for the
    purposes of a *specific* KPI subset (handled at the metrics layer via
    per-metric NaN + note, not by discarding the whole trial). This helper
    exists for the (rarer) case where the ENTIRE trial has no evaluable
    handover at all.
    """
    if result.status == "INVALID":
        return result
    result.status = "NOT_EVALUABLE"
    result.add(reason)
    return result
