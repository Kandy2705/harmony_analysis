#!/usr/bin/env python3
"""
step8_kb06_replacement.py
---------------------------
KB06 has 7 REAL recorded trials, but all 7 crashed mid-logging (1-3 event
rows each, no completion) and were correctly excluded as INVALID during
Task 1's cleaning pass (see excluded_trials.csv / _excluded_invalid/KB06).
That leaves KB06 as the only one of the 30 KB scenarios with zero trials
passing validation at all. Per the user's explicit choice (consistent with
the step7 override), this generates a synthetic replacement set for KB06
using the exact same construction as step6's KB01 (KB06 is GPS_TO_VPS per
the original DIRECTION_MAP, a standard baseline handover, no special
condition) -- real per-version distributions, real template geometry, no
invented columns or event types.
"""
import numpy as np
import pandas as pd
from pathlib import Path

from step4_synthetic_reverse import dwell_config_for_version
from step6_synthetic_missing_scenarios import load_forward_stats
from step7_synthetic_expanded_scenarios import _gps_to_vps_core, _finish_indoor_walk, finalize_trial, _mk_ev

CLEAN = Path("sample_data_clean")
OUT = CLEAN / "_synthetic_expanded_scenarios"  # same bucket as the other override-generated scenarios
SEED = 20260920
VERSIONS = ["V1", "V2", "V3", "V4", "V5", "BQ", "BT"]


def build_kb06(version, stats, cfg, rng):
    seed_tag = int(rng.integers(1000, 9999))
    session_id = f"SYNTH_KB06_{version}_{seed_tag}_FWD"
    ev_rows, rows, scanning_s, entry_indoor_row, dest_indoor_row, indoor_rows, add, t = _gps_to_vps_core(0.0, stats, cfg, rng)
    dt = stats["sampling_dt"]
    vps_attempt = 1
    ev_rows.append(_mk_ev(scanning_s, "vps_state", to_state="StartingVps", source="Pdr", vps_attempt=vps_attempt))
    ev_rows.append(_mk_ev(scanning_s + 0.3, "vps_state", to_state="Scanning", source="Pdr", vps_attempt=vps_attempt))
    t = scanning_s
    for i in range(3):
        t += dt
        frac = (i + 1) / 3
        add(t, state="EnteringWithPdr", source="Pdr", active_source="Pdr",
            campus_x=entry_indoor_row.get("campus_x"), campus_y=entry_indoor_row.get("campus_y"),
            heading_deg=entry_indoor_row.get("heading_deg"), vps_valid=1, vps_state="Scanning", vps_attempt=vps_attempt,
            vps_reliability=round(float(np.clip(rng.normal(stats["vps_rel_mean"], 0.03), 0.5, 0.98)), 4),
            vps_confidence=round(float(np.clip(cfg["vpsEnterReliability"] * frac + 0.1, 0.3, 0.95)), 4),
            vps_confidence_available=1, vps_threshold_passed=int(frac >= 1.0))
    t += 0.8
    accept_s = round(t, 3)
    ev_rows.append(_mk_ev(accept_s, "vps_state", to_state="IndoorLocalized", source="Vps", note="VPS localization accepted", vps_attempt=vps_attempt))
    ev_rows.append(_mk_ev(accept_s, "reliability_state", "VpsScanning", "IndoorVps", "Vps", "VPS localization accepted"))
    ev_rows.append(_mk_ev(accept_s, "indoor_state", to_state="Navigating", source="Vps"))
    t = accept_s + dt * 0.1
    add(t, state="IndoorVps", source="Vps", active_source="Vps",
        campus_x=entry_indoor_row.get("campus_x"), campus_y=entry_indoor_row.get("campus_y"), campus_z=entry_indoor_row.get("campus_z"),
        heading_deg=entry_indoor_row.get("heading_deg"), latitude=entry_indoor_row.get("latitude"), longitude=entry_indoor_row.get("longitude"),
        vps_valid=1, vps_state="IndoorLocalized", vps_attempt=vps_attempt,
        vps_reliability=round(float(np.clip(rng.normal(stats["vps_rel_mean"], 0.03), 0.5, 0.98)), 4),
        vps_confidence=round(float(np.clip(rng.normal(stats["vps_conf_mean"], 0.03), 0.5, 0.98)), 4),
        vps_confidence_available=1, vps_map_id_available=1, vps_map_matches=1,
        indoor_state="Navigating", indoor_pose_source="Vps")
    t = accept_s + dt
    t_end, dest, _ = _finish_indoor_walk(ev_rows, rows, add, t, indoor_rows, dt)
    ev_df, sm_df, su_df = finalize_trial(
        session_id, "KB06", version, cfg, "GPS_TO_VPS", ev_rows, rows, dest,
        completed=1, handover_success=1, destination_arrived=1, final_state="IndoorVps",
        end_reason="destination_reached", handover_attempts_total=1, handover_attempts_evaluable=1,
        successful_handovers=1, false_handovers=0, incomplete_handovers=0,
    )
    grounding = ("STRONG: identical construction to step6's KB01 (standard baseline GPS_TO_VPS handover, real "
                 "per-version distributions + real template geometry). Generated specifically to replace KB06's "
                 "7 real trials, which all crashed mid-logging (1-3 event rows, no completion) and were excluded "
                 "as INVALID in Task 1 -- see excluded_trials.csv. This is a REPLACEMENT for corrupted real "
                 "recordings, not a stand-in for an unrepresentable condition.")
    return session_id, ev_df, sm_df, su_df, stats["template_sid"], grounding


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows_manifest = []
    rng_master = np.random.default_rng(SEED)
    for version in VERSIONS:
        stats = load_forward_stats(version)
        cfg = dwell_config_for_version(version)
        rng = np.random.default_rng(rng_master.integers(0, 2**31 - 1))
        session_id, ev_df, sm_df, su_df, template_used, grounding = build_kb06(version, stats, cfg, rng)
        dest = OUT / "KB06" / session_id
        dest.mkdir(parents=True, exist_ok=True)
        ev_df.to_csv(dest / f"events_{session_id}.csv", index=False)
        sm_df.to_csv(dest / f"samples_{session_id}.csv", index=False)
        su_df.to_csv(dest / f"summary_{session_id}.csv", index=False)
        rows_manifest.append({
            "session_id": session_id, "scenario_id": "KB06", "harmony_version": version,
            "data_provenance": "SYNTHETIC_SUPPORT", "synthetic": 1, "generation_seed": SEED,
            "template_session_id_used": template_used, "validation_status": "PENDING_PIPELINE_CHECK",
            "grounding_strength": grounding,
        })
    mdf = pd.DataFrame(rows_manifest)
    mdf.to_csv("kb06_replacement_manifest.csv", index=False)
    print(f"Generated {len(mdf)} KB06 replacement trials")


if __name__ == "__main__":
    main()
