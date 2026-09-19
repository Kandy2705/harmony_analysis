"""
discovery.py
------------
Recursively scans a root experiment folder for events_*.csv, samples_*.csv
and summary_*.csv files, WITHOUT relying on folder naming conventions.

Folder structure is treated as opaque; only used later (in pairing.py) as
a *fallback* label if a trial's CSVs are missing scenario/direction metadata.
"""
from __future__ import annotations

import fnmatch
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List


@dataclass
class DiscoveredFiles:
    """Raw file discovery result, before any pairing/validation."""
    events_files: List[Path] = field(default_factory=list)
    samples_files: List[Path] = field(default_factory=list)
    summary_files: List[Path] = field(default_factory=list)

    def total(self) -> int:
        return len(self.events_files) + len(self.samples_files) + len(self.summary_files)


def scan_root(root: str | Path,
              events_glob: str = "events_*.csv",
              samples_glob: str = "samples_*.csv",
              summary_glob: str = "summary_*.csv",
              recursive: bool = True) -> DiscoveredFiles:
    """Walk `root` and collect every file matching the three glob patterns.

    Parameters
    ----------
    root : str | Path
        Root folder to scan (e.g. HARMONY_Experiments/).
    recursive : bool
        If False, only scan the top-level of `root` (mainly for testing).
    """
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"Root folder does not exist: {root}")

    result = DiscoveredFiles()

    if recursive:
        walker = os.walk(root)
    else:
        walker = [(str(root), [], [p.name for p in root.iterdir() if p.is_file()])]

    for dirpath, _dirnames, filenames in walker:
        for fname in filenames:
            full = Path(dirpath) / fname
            if fnmatch.fnmatch(fname, events_glob):
                result.events_files.append(full)
            elif fnmatch.fnmatch(fname, samples_glob):
                result.samples_files.append(full)
            elif fnmatch.fnmatch(fname, summary_glob):
                result.summary_files.append(full)

    return result
