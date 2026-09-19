"""
pairing.py
----------
Groups discovered events_*.csv / samples_*.csv / summary_*.csv files into
"trials" using the `session_id` column found INSIDE each CSV — never the
folder name or file name — per spec section 1.5/1.6.

A trial is considered a candidate as soon as at least one of the three
files for a given session_id is found; missing files are reported as
validation issues rather than silently dropped (spec section 5).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from .discovery import DiscoveredFiles
from .loader import LoadedCsv, LoadIssue, load_csv


@dataclass
class TrialFiles:
    session_id: str
    events: Optional[LoadedCsv] = None
    samples: Optional[LoadedCsv] = None
    summary: Optional[LoadedCsv] = None
    # any extra files that matched the same session_id (duplicates)
    duplicate_events: List[Path] = field(default_factory=list)
    duplicate_samples: List[Path] = field(default_factory=list)
    duplicate_summary: List[Path] = field(default_factory=list)
    pairing_issues: List[LoadIssue] = field(default_factory=list)


def _extract_session_id(loaded: LoadedCsv) -> Optional[str]:
    """Best-effort extraction of a single session_id from a loaded CSV.

    Returns None if the file couldn't be loaded or has no session_id
    column/value. Reports (via loaded.issues) if MULTIPLE distinct
    session_ids are found in one file, which would indicate a corrupted
    or concatenated export.
    """
    if loaded.df is None or "session_id" not in loaded.df.columns:
        return None
    ids = loaded.df["session_id"].dropna().unique().tolist()
    if len(ids) == 0:
        return None
    if len(ids) > 1:
        loaded.issues.append(LoadIssue(
            "ERROR", "MULTIPLE_SESSION_IDS",
            f"File contains {len(ids)} distinct session_id values, expected 1: {ids[:5]}...",
            str(loaded.path),
        ))
        # We still return the first — the ERROR above will mark the trial INVALID.
    return str(ids[0])


def pair_trials(discovered: DiscoveredFiles) -> Dict[str, TrialFiles]:
    """Load every discovered file and group them into TrialFiles by session_id."""
    trials: Dict[str, TrialFiles] = {}

    def _register(loaded: LoadedCsv, kind: str) -> None:
        sid = _extract_session_id(loaded)
        if sid is None:
            # Can't pair this file to any trial; still need to surface the failure
            # to the user rather than silently discarding it.
            orphan_key = f"__UNPAIRABLE__::{loaded.path}"
            trial = trials.setdefault(orphan_key, TrialFiles(session_id="UNKNOWN"))
            trial.pairing_issues.extend(loaded.issues or [
                LoadIssue("ERROR", "NO_SESSION_ID",
                          f"Could not determine session_id for {kind} file", str(loaded.path))
            ])
            setattr(trial, kind, loaded)
            return

        trial = trials.setdefault(sid, TrialFiles(session_id=sid))
        existing = getattr(trial, kind)
        if existing is not None:
            # Duplicate file for the same (kind, session_id)
            getattr(trial, f"duplicate_{kind}").append(loaded.path)
            trial.pairing_issues.append(LoadIssue(
                "WARNING", "DUPLICATE_FILE",
                f"Multiple {kind} files found for session_id={sid}: "
                f"{existing.path} and {loaded.path} (first kept as primary)",
                str(loaded.path),
            ))
        else:
            setattr(trial, kind, loaded)
        trial.pairing_issues.extend(loaded.issues)

    for p in discovered.events_files:
        _register(load_csv(p, "events"), "events")
    for p in discovered.samples_files:
        _register(load_csv(p, "samples"), "samples")
    for p in discovered.summary_files:
        _register(load_csv(p, "summary"), "summary")

    return trials
