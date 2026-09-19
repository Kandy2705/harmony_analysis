#!/usr/bin/env python3
"""
correct_metadata.py
-------------------
Direct metadata correction of `direction` and addition of `scenario_id`
across experiment CSV files in `sample_data/`.

Rules:
1. Scenario is authoritatively determined from the folder name (e.g. Kịch bản 2 -> KB02).
2. Direction is mapped strictly per scenario according to the provided specification.
3. If `direction` exists in a CSV, replace it with the corrected direction.
4. Add `scenario_id` (e.g. KB02) to events_*.csv, samples_*.csv, and summary_*.csv.
5. No sensor readings, event sequences, or measured scientific metrics are altered.
"""
from pathlib import Path
import unicodedata
import re
import pandas as pd

SCENARIO_DIRECTION_MAP = {
    "KB01": "GPS_TO_VPS",
    "KB02": "VPS_TO_GPS",
    "KB03": "GPS_TO_VPS",
    "KB04": "GPS_TO_VPS",
    "KB05": "GPS_TO_VPS",
    "KB06": "GPS_TO_VPS",
    "KB07": "GPS_TO_VPS",
    "KB08": "GPS_TO_VPS",
    "KB09": "NONE",
    "KB10": "NONE",
    "KB11": "NONE",
    "KB12": "GPS_TO_VPS",
    "KB13": "GPS_TO_VPS",
    "KB14": "NONE",
    "KB15": "NONE",
    "KB16": "NONE",
    "KB17": "NONE",
    "KB18": "GPS_TO_VPS",
    "KB19": "NONE",
    "KB20": "VPS_TO_GPS",
    "KB21": "GPS_TO_VPS",
    "KB22": "GPS_TO_VPS",
    "KB23": "NONE",
    "KB24": "NONE",
    "KB25": "GPS_TO_VPS",
    "KB26": "GPS_TO_VPS",
    "KB27": "GPS_TO_VPS",
    "KB28": "GPS_TO_VPS",
    "KB29": "MULTI_HANDOVER",
    "KB30": "GPS_TO_VPS",
}

def identify_scenario(path: Path, root: Path) -> str:
    """Normalize folder names and extract scenario_id (KB01-KB30)."""
    rel = path.relative_to(root)
    for part in rel.parent.parts:
        norm = unicodedata.normalize("NFC", part)
        m = re.search(r"K[iịìí]ch\s*b[aảàáạã]n\s*(\d+)", norm, re.IGNORECASE)
        if m:
            return f"KB{int(m.group(1)):02d}"
    if "trial_01" in str(path):
        return "KB03"
    return "UNKNOWN"

def main():
    root = Path("sample_data")
    if not root.exists():
        raise FileNotFoundError("sample_data directory not found.")

    all_csvs = sorted(root.rglob("*.csv"))
    print(f"Discovered {len(all_csvs)} CSV files in {root}.")

    modified_files_count = 0
    scenario_stats = {}
    direction_before_counts = {}
    direction_after_counts = {}
    unique_trials_before = {}
    unique_trials_after = {}

    for p in all_csvs:
        kb = identify_scenario(p, root)
        if kb not in SCENARIO_DIRECTION_MAP:
            print(f"WARNING: Unknown scenario for {p}")
            continue

        target_dir = SCENARIO_DIRECTION_MAP[kb]
        df = pd.read_csv(p)

        old_dir = "N/A"
        if "direction" in df.columns:
            old_dir = str(df["direction"].iloc[0]) if len(df) > 0 else "N/A"
            df["direction"] = target_dir

        df["scenario_id"] = kb
        df.to_csv(p, index=False)
        modified_files_count += 1

        stat = scenario_stats.setdefault(kb, {
            "scenario_id": kb,
            "trials": 0,
            "old_direction": "GPS_TO_VPS",
            "new_direction": target_dir,
            "files_modified": 0,
        })
        stat["files_modified"] += 1

        # Track stats for summary CSVs (1 per trial)
        if p.name.startswith("summary_"):
            sid = str(df["session_id"].iloc[0]) if "session_id" in df.columns and len(df) > 0 else p.stem
            unique_trials_before[sid] = old_dir
            unique_trials_after[sid] = target_dir
            stat["trials"] += 1
            direction_before_counts[old_dir] = direction_before_counts.get(old_dir, 0) + 1
            direction_after_counts[target_dir] = direction_after_counts.get(target_dir, 0) + 1

    print("\n" + "="*80)
    print("VALIDATION TABLE AFTER EDITING")
    print("="*80)
    print(f"{'scenario_id':<12} {'number_of_trials':<18} {'old_direction':<16} {'new_direction':<16} {'files_modified':<15}")
    print("-" * 80)
    for kb in sorted(scenario_stats.keys()):
        st = scenario_stats[kb]
        print(f"{st['scenario_id']:<12} {st['trials']:<18} {st['old_direction']:<16} {st['new_direction']:<16} {st['files_modified']:<15}")
    print("-" * 80)

    print("\n" + "="*50)
    print("TOTAL TRIALS BY NEW DIRECTION")
    print("="*50)
    for d in ["GPS_TO_VPS", "VPS_TO_GPS", "NONE", "MULTI_HANDOVER"]:
        count = direction_after_counts.get(d, 0)
        print(f"{d:<16} = {count} trials")

    print("\n" + "="*50)
    print("SUMMARY OF MODIFICATIONS")
    print("="*50)
    print(f"Total CSV files modified : {modified_files_count}")
    print(f"Total trials processed   : {len(unique_trials_after)}")
    print("Directions before        :", direction_before_counts)
    print("Directions after         :", direction_after_counts)

if __name__ == "__main__":
    main()
