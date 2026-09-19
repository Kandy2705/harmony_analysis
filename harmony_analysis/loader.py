"""
loader.py
---------
Safe CSV loading for events / samples / summary files.

Design goals (per spec section 5 & 11):
  * Never silently swallow a bad file — every failure becomes a
    structured LoadIssue that flows into the validation report.
  * Never invent data: missing/duplicate/corrupted files are reported,
    not guessed at.
  * Read-only: this module never writes to the source files.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import pandas as pd


@dataclass
class LoadIssue:
    severity: str        # "ERROR" | "WARNING"
    code: str            # short machine-readable code
    message: str
    file: Optional[str] = None


@dataclass
class LoadedCsv:
    path: Path
    df: Optional[pd.DataFrame]
    issues: List[LoadIssue] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.df is not None and not any(i.severity == "ERROR" for i in self.issues)


REQUIRED_COLUMNS = {
    "events": {"session_id", "elapsed_s", "event"},
    "samples": {"session_id", "elapsed_s", "state", "source"},
    "summary": {"session_id"},
}


def load_csv(path: Path, kind: str) -> LoadedCsv:
    """Load a single CSV file with defensive error handling.

    `kind` is one of "events", "samples", "summary" and controls which
    required-column set is checked.
    """
    issues: List[LoadIssue] = []

    if not path.exists():
        issues.append(LoadIssue("ERROR", "FILE_MISSING", f"File does not exist: {path}", str(path)))
        return LoadedCsv(path=path, df=None, issues=issues)

    if path.stat().st_size == 0:
        issues.append(LoadIssue("ERROR", "EMPTY_FILE", f"File is empty (0 bytes): {path}", str(path)))
        return LoadedCsv(path=path, df=None, issues=issues)

    try:
        df = pd.read_csv(path, encoding="utf-8-sig")
    except pd.errors.EmptyDataError:
        issues.append(LoadIssue("ERROR", "EMPTY_CSV", f"CSV has no columns/rows: {path}", str(path)))
        return LoadedCsv(path=path, df=None, issues=issues)
    except Exception as exc:  # noqa: BLE001 - we want to capture ANY parse failure
        issues.append(LoadIssue("ERROR", "CORRUPTED_CSV", f"Failed to parse CSV ({exc}): {path}", str(path)))
        return LoadedCsv(path=path, df=None, issues=issues)

    if df.shape[0] == 0:
        issues.append(LoadIssue("WARNING", "NO_ROWS", f"CSV has headers but 0 data rows: {path}", str(path)))

    required = REQUIRED_COLUMNS.get(kind, set())
    missing = required - set(df.columns)
    if missing:
        issues.append(LoadIssue(
            "ERROR", "MISSING_REQUIRED_COLUMNS",
            f"Missing required columns {sorted(missing)} in {kind} file: {path}",
            str(path),
        ))

    # Duplicate-row detection (exact duplicate rows) — warning only.
    dup_count = int(df.duplicated().sum())
    if dup_count > 0:
        issues.append(LoadIssue(
            "WARNING", "DUPLICATE_ROWS",
            f"{dup_count} exact duplicate row(s) found in {kind} file: {path}",
            str(path),
        ))

    return LoadedCsv(path=path, df=df, issues=issues)
