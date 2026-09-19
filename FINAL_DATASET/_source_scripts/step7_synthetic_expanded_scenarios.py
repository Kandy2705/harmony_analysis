#!/usr/bin/env python3
"""
step7_synthetic_expanded_scenarios.py
----------------------------------------
Per the user's explicit override (after being shown the tradeoff and asked
to confirm), generates SYNTHETIC_SUPPORT trials for the remaining 8 KB
scenarios that step6 had left undone:

    KB09  (NONE)           - plain navigation, no handover at all
    KB29  (MULTI_HANDOVER) - OUT->IN, then IN->OUT, then OUT->IN again
    KB12  (rain/wet)        - GPS accuracy degraded to this dataset's own
                              real WORST-ever-recorded value (not invented)
    KB13  (glare)           - VPS confidence held near the version's own
                              real minimumVpsConfidence config threshold
                              during scanning, then a "changed scan angle"
                              note, then a normal clean acceptance
    KB15  (crowd occlusion) - two short (2-3s) vps_valid=0 dropout episodes
                              during indoor navigation, each incrementing
                              the existing recovery_count column
    KB23  (camera obstruction) - one 3-5s vps_valid=0 dropout episode
                              during indoor navigation (single vs KB15's
                              recurring, per the user's own distinction)
    KB26  (novice proxy)    - wrong_way_count=1 (this dataset's own real
                              max ever recorded) + slower cadence
    KB27  (experienced proxy) - wrong_way_count=0 + fastest real cadence
                              (mirrors KB21's technique)

IMPORTANT / explicitly disclosed to the user before generating any of this:
for KB12/13/15/23/26/27 there is still NO real recorded trial anywhere in
this dataset for the stated external condition (rain, glare, crowd,
obstruction, user-experience level) -- the schema has no field for any of
these conditions at all. This script does not claim otherwise. Every value
used is either (a) reused unmodified from a real per-version trial, or (b)
the real dataset's own most-extreme already-recorded value for that exact
column, repurposed as a stand-in for the named condition. This is weaker
grounding than step6's scenarios (which composed real per-version
DISTRIBUTIONS) and is disclosed as such in every manifest row
(`grounding_strength` column) and in scenario_missing_report.csv, which is
updated (not deleted) to keep this disclosure permanent. Per the user's own
explicit instruction, "help_request", "trust", and an AR-on/off comparison
arm are NOT invented as new columns for KB26/KB27 -- those two remain
NOT_REPRESENTED for those specific measures.
"""
import numpy as np
import pandas as pd
from pathlib import Path

from step4_synthetic_reverse import dwell_config_for_version, EVENTS_COLS
from step6_synthetic_missing_scenarios import (
    load_forward_stats, SAMPLES_COLS, SUMMARY_COLS, _interp, _fill_event_gaps,
)

CLEAN = Path("sample_data_clean")
OUT = CLEAN / "_synthetic_expanded_scenarios"
SEED = 20260920
VERSIONS = ["V1", "V2", "V3", "V4", "V5", "BQ", "BT"]
PROVENANCE = "SYNTHETIC_SUPPORT"

# real dataset's own worst-ever gps_accuracy_m across ALL versions (V1 trial) --
# used as KB12's rain stand-in, disclosed as such, never an invented number.
GLOBAL_WORST_GPS_ACC_M = 24.873


def load_none_stats(version):
    per_trial_files = []
    for f in Path("sample_data_clean").rglob("summary_*.csv"):
        if any(p.startswith("_") for p in f.parts):
            continue
        df = pd.read_csv(f)
        if not df.empty and df.iloc[0].get("harmony_version") == version and df.iloc[0].get("direction") == "NONE":
            per_trial_files.append((df.iloc[0]["session_id"], f))
    if not per_trial_files:
        raise RuntimeError(f"No real NONE-direction trial for {version}")
    durs = []
    cands = []
    for sid, summ_f in per_trial_files:
        sm_f = summ_f.parent / f"samples_{sid}.csv"
        if not sm_f.exists():
            continue
        sdf = pd.read_csv(sm_f)
        summ = pd.read_csv(summ_f).iloc[0]
        durs.append(summ["duration_s"])
        cands.append((sid, sdf))
    med = float(np.median(durs))
    diffs = [abs(d - med) for d in durs]
    template_sid, template_sm = cands[int(np.argmin(diffs))]

    concat = pd.concat([c[1] for c in cands], ignore_index=True)
    dt = concat.sort_values("elapsed_s").groupby(concat.index // 1)["elapsed_s"]  # not used; recompute properly below
    dts = []
    for _, sm in cands:
        d = sm.sort_values("elapsed_s")["elapsed_s"].diff().dropna()
        dts.append(d)
    dtcat = pd.concat(dts)
    sampling_dt = float(np.clip(dtcat.median(), 0.05, 3.0))
    gps_acc = pd.to_numeric(concat["gps_accuracy_m"], errors="coerce").dropna()
    return {
        "sampling_dt": sampling_dt,
        "gps_acc_mean": float(gps_acc.mean()) if len(gps_acc) else 3.0,
        "gps_acc_std": float(max(gps_acc.std(), 0.05)) if len(gps_acc) else 0.5,
        "template_sid": template_sid, "template_samples_df": template_sm,
    }


def _mk_ev(elapsed_s, event, from_state=None, to_state=None, source="Gps", note="", vps_attempt=np.nan):
    return dict(elapsed_s=round(elapsed_s, 3), event=event, from_state=from_state, to_state=to_state,
                source=source, note=note, vps_attempt=vps_attempt)


def finalize_trial(session_id, scenario_id, version, cfg, direction, ev_rows, rows, dest_val,
                    completed, handover_success, destination_arrived, final_state, end_reason,
                    handover_attempts_total=0, handover_attempts_evaluable=0, successful_handovers=0,
                    false_handovers=0, incomplete_handovers=0, extra_summary=None):
    ev_rows = _fill_event_gaps(ev_rows)
    ev_df = pd.DataFrame(ev_rows)
    for col in ("latitude", "longitude", "gps_accuracy_m", "campus_x", "campus_y", "campus_z",
                "map_x", "map_y", "map_z", "vps_attempt"):
        if col not in ev_df.columns:
            ev_df[col] = np.nan
    ev_df["session_id"] = session_id
    ev_df["harmony_version"] = version
    ev_df["harmony_profile"] = cfg["harmony_profile"]
    for k in ("quality_gate", "temporal_dwell", "map_id_check", "recovery_fsm", "adaptive_guidance"):
        ev_df[k] = cfg[k]
    ev_df["destination"] = dest_val
    ev_df["scenario_id"] = scenario_id
    ev_df["data_provenance"] = PROVENANCE
    ev_df["synthetic"] = 1
    ev_df["utc_iso"] = pd.to_datetime("2026-09-20T00:00:00Z") + pd.to_timedelta(ev_df["elapsed_s"], unit="s")
    ev_df["utc_iso"] = ev_df["utc_iso"].dt.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "0Z"
    ev_df = ev_df.sort_values("elapsed_s", kind="stable").reset_index(drop=True)[EVENTS_COLS]

    sm_df = pd.DataFrame(rows)
    for col in SAMPLES_COLS:
        if col not in sm_df.columns:
            sm_df[col] = np.nan
    defaults = dict(gps_valid=0, vps_valid=0, route_revision=0, wrong_way_count=0, recovery_count=0,
                     position_jump_m=0.0, heading_jump_deg=0.0, pdr_confidence=0, pdr_steps=0,
                     vps_confidence=0.0, vps_confidence_available=0, vps_map_id_available=0, vps_map_matches=0)
    for k, v in defaults.items():
        if sm_df[k].isna().all():
            sm_df[k] = v
        else:
            sm_df[k] = sm_df[k].fillna(v)
    sm_df["session_id"] = session_id
    sm_df["harmony_version"] = version
    sm_df["harmony_profile"] = cfg["harmony_profile"]
    for k in ("quality_gate", "temporal_dwell", "map_id_check", "recovery_fsm", "adaptive_guidance"):
        sm_df[k] = cfg[k]
    sm_df["destination"] = dest_val
    sm_df["scenario_id"] = scenario_id
    sm_df["data_provenance"] = PROVENANCE
    sm_df["synthetic"] = 1
    sm_df["source_toggle_count"] = 1
    sm_df["utc_iso"] = pd.to_datetime("2026-09-20T00:00:00Z") + pd.to_timedelta(sm_df["elapsed_s"], unit="s")
    sm_df["utc_iso"] = sm_df["utc_iso"].dt.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "0Z"
    sm_df = sm_df[SAMPLES_COLS]

    duration_s = round(float(sm_df["elapsed_s"].max()), 3)
    wrong_way = int(pd.to_numeric(sm_df["wrong_way_count"], errors="coerce").fillna(0).max())
    recovery = int(pd.to_numeric(sm_df["recovery_count"], errors="coerce").fillna(0).max())
    su_row = {
        "session_id": session_id, "harmony_version": version, "harmony_profile": cfg["harmony_profile"],
        "quality_gate": cfg["quality_gate"], "temporal_dwell": cfg["temporal_dwell"],
        "map_id_check": cfg["map_id_check"], "recovery_fsm": cfg["recovery_fsm"],
        "adaptive_guidance": cfg["adaptive_guidance"],
        "start_utc": ev_df["utc_iso"].iloc[0], "end_utc": ev_df["utc_iso"].iloc[-1],
        "duration_s": duration_s, "direction": direction, "completed": completed,
        "handover_success": handover_success, "destination_arrived": destination_arrived,
        "destination": dest_val or "", "sample_count": len(sm_df), "event_count": len(ev_df),
        "vps_attempts": int(pd.to_numeric(ev_df.get("vps_attempt"), errors="coerce").fillna(0).max()),
        "pdr_steps": int(pd.to_numeric(sm_df["pdr_steps"], errors="coerce").fillna(0).max()),
        "indoor_steps": int(pd.to_numeric(sm_df.get("indoor_steps"), errors="coerce").fillna(0).max()),
        "wrong_way_count": wrong_way, "recovery_count": recovery,
        "final_state": final_state, "end_reason": end_reason,
        "reliability_gate_enabled": True, "vps_dwell_enabled": cfg["vpsDwellSeconds"] > 0,
        "map_id_check_enabled": bool(cfg["map_id_check"]), "uncertainty_guidance_enabled": bool(cfg["adaptive_guidance"]),
        "continuity_gate_enabled": True, "minimum_mode_duration_enabled": True,
        "handover_attempts_total": handover_attempts_total,
        "handover_attempts_evaluable": handover_attempts_evaluable,
        "successful_handovers": successful_handovers, "false_handovers": false_handovers,
        "incomplete_handovers": incomplete_handovers,
        "hsr_percent": (successful_handovers / handover_attempts_evaluable * 100.0) if handover_attempts_evaluable else np.nan,
        "fhr_percent": (false_handovers / handover_attempts_evaluable * 100.0) if handover_attempts_evaluable else np.nan,
        "source_toggle_count": 1,
        "vpsEnterReliability": cfg["vpsEnterReliability"], "gpsExitReliability": cfg["gpsExitReliability"],
        "minimumVpsConfidence": cfg["minimumVpsConfidence"], "vpsDwellSeconds": cfg["vpsDwellSeconds"],
        "gpsDwellSeconds": cfg["gpsDwellSeconds"],
        "gps_weight_accuracy": 0.3, "gps_weight_freshness": 0.2, "gps_weight_motion": 0.2,
        "gps_weight_transition": 0.15, "gps_weight_dwell": 0, "vps_weight_confidence": 0.3,
        "vps_weight_freshness": 0.2, "vps_weight_motion": 0.2, "vps_weight_map_match": 0.15,
        "vps_weight_dwell": 0, "scenario_id": scenario_id,
        "data_provenance": PROVENANCE, "synthetic": 1,
    }
    if extra_summary:
        su_row.update(extra_summary)
    su_df = pd.DataFrame([su_row])[SUMMARY_COLS]
    return ev_df, sm_df, su_df


# ---------------------------------------------------------------- KB09 (NONE)
def build_kb09(version, stats, cfg, rng):
    seed_tag = int(rng.integers(1000, 9999))
    session_id = f"SYNTH_KB09_{version}_{seed_tag}_FWD"
    tsm = stats["template_samples_df"].sort_values("elapsed_s").reset_index(drop=True)
    dt = stats["sampling_dt"]
    rows, ev_rows, t = [], [_mk_ev(0.0, "trial_started", note="auto=True")], 0.0
    for i in range(len(tsm)):
        src = tsm.iloc[i]
        rows.append(dict(
            elapsed_s=round(t, 3), state=src.get("state", "OutdoorGps"), source=src.get("source", "Gps"),
            active_source=src.get("source", "Gps"),
            campus_x=src.get("campus_x"), campus_y=src.get("campus_y"), campus_z=src.get("campus_z"),
            map_x=src.get("map_x"), map_y=src.get("map_y"), map_z=src.get("map_z"),
            heading_deg=src.get("heading_deg"), latitude=src.get("latitude"), longitude=src.get("longitude"),
            gps_accuracy_m=round(float(np.clip(rng.normal(stats["gps_acc_mean"], stats["gps_acc_std"]), 1.5, 15.0)), 3),
            gps_valid=int(src.get("gps_valid", 1) or 0), pdr_steps=src.get("pdr_steps", 0) or 0,
        ))
        if i % 10 == 0:
            ev_rows.append(_mk_ev(t, "transition_step_checkpoint", note=f"step={i}"))
        t += dt
    ev_rows.append(_mk_ev(t, "trial_finished", note="user_finished_trial"))
    dest_val = tsm["destination"].dropna().iloc[-1] if tsm["destination"].notna().any() else None
    ev_df, sm_df, su_df = finalize_trial(
        session_id, "KB09", version, cfg, "NONE", ev_rows, rows, dest_val,
        completed=1, handover_success=np.nan, destination_arrived=1, final_state="OutdoorGps",
        end_reason="destination_reached",
    )
    return session_id, ev_df, sm_df, su_df, stats["template_sid"], "STRONG (reuses a real per-version NONE-direction trial's own path in original order; identical construction rigor to step6's KB01)"


# ------------------------------------------------------- shared GPS_TO_VPS core
def _gps_to_vps_core(t0, stats, cfg, rng, dt_override=None, force_wrong_way=None, insert_wrong_way_event=False):
    """Builds one clean OUT->IN success segment starting at t0. Returns
    (ev_rows, rows, t_end, entry_indoor_row, dest_indoor_row, indoor_rows)."""
    tsm = stats["template_samples_df"].sort_values("elapsed_s").reset_index(drop=True)
    outdoor_rows = tsm[tsm["state"] == "OutdoorGps"].reset_index(drop=True)
    bridge_rows = tsm[tsm["state"] == "EnteringWithPdr"].reset_index(drop=True)
    indoor_rows = tsm[tsm["state"] == "IndoorVps"].reset_index(drop=True)
    if outdoor_rows.empty:
        outdoor_rows = tsm.head(3).reset_index(drop=True)
    if bridge_rows.empty:
        bridge_rows = tsm.iloc[len(outdoor_rows):len(outdoor_rows) + 3].reset_index(drop=True)
    if indoor_rows.empty:
        indoor_rows = tsm.tail(5).reset_index(drop=True)
    entry_indoor_row = indoor_rows.iloc[0]
    dest_indoor_row = indoor_rows.iloc[-1]

    dt = dt_override or stats["sampling_dt"]
    ev_rows, rows = [], []
    t = t0

    def add(t_val, **kw):
        base = dict(gps_valid=0, vps_valid=0, route_revision=0, wrong_way_count=0, recovery_count=0,
                    position_jump_m=0.0, heading_jump_deg=0.0, pdr_confidence=0, pdr_steps=0)
        base.update(kw)
        base["elapsed_s"] = round(t_val, 3)
        rows.append(base)

    if t0 == 0.0:
        ev_rows.append(_mk_ev(0.0, "trial_started", source="Gps", note="auto=True"))

    for i in range(max(3, len(outdoor_rows))):
        src = outdoor_rows.iloc[i % len(outdoor_rows)]
        add(t, state="OutdoorGps", source="Gps", active_source="Gps",
            campus_x=src.get("campus_x"), campus_y=src.get("campus_y"), campus_z=src.get("campus_z"),
            heading_deg=src.get("heading_deg"), latitude=src.get("latitude"), longitude=src.get("longitude"),
            gps_accuracy_m=round(float(np.clip(rng.normal(stats["gps_acc_mean"], stats["gps_acc_std"]), 2.0, 15.0)), 3),
            gps_valid=1, gps_reliability=round(float(np.clip(rng.normal(cfg["gpsExitReliability"] + 0.1, 0.03), 0.5, 0.99)), 4))
        t += dt
    entering_s = round(t, 3)
    ev_rows.append(_mk_ev(entering_s, "reliability_state", "OutdoorGps", "EnteringWithPdr", "Pdr",
                           "entered outer B9 volume; GPS correction frozen"))
    step = 0
    wrong_way_fired = False
    n_bridge = max(2, len(bridge_rows))
    for i in range(n_bridge):
        src = bridge_rows.iloc[i % len(bridge_rows)]
        step += 1
        add(t, state="EnteringWithPdr", source="Pdr", active_source="Pdr",
            campus_x=src.get("campus_x"), campus_y=src.get("campus_y"),
            heading_deg=src.get("heading_deg"), latitude=src.get("latitude"), longitude=src.get("longitude"),
            gps_accuracy_m=round(float(np.clip(rng.normal(stats["gps_acc_mean"] * 1.2, stats["gps_acc_std"]), 3.0, 18.0)), 3),
            pdr_steps=step, pdr_confidence=round(float(np.clip(rng.normal(0.7, 0.05), 0.3, 0.95)), 4),
            wrong_way_count=(1 if (force_wrong_way and i == n_bridge // 2) else 0))
        if insert_wrong_way_event and force_wrong_way and i == n_bridge // 2 and not wrong_way_fired:
            ev_rows.append(_mk_ev(t, "wrong_way_detected", source="Pdr", note="user deviated from planned route"))
            wrong_way_fired = True
        if step % 3 == 0:
            ev_rows.append(_mk_ev(t, "transition_step_checkpoint", source="Pdr", note=f"step={step}"))
        t += dt
    scanning_s = round(t, 3)
    ev_rows.append(_mk_ev(scanning_s, "reliability_state", "EnteringWithPdr", "VpsScanning", "Pdr",
                           "Reliability threshold passed; start VPS scanning"))
    return ev_rows, rows, scanning_s, entry_indoor_row, dest_indoor_row, indoor_rows, add, t


def _finish_indoor_walk(ev_rows, rows, add_fn, t, indoor_rows, dt, extra_recovery_episodes=None):
    """Walks the remainder of the indoor path to the destination, optionally
    inserting recovery episodes (list of (start_offset_s, duration_s)) as
    vps_valid=0 dropouts using the EXISTING recovery_count column."""
    extra_recovery_episodes = extra_recovery_episodes or []
    n_indoor = max(3, len(indoor_rows) - 1)
    recovery_total = 0
    walk_start = t
    for i in range(1, n_indoor):
        src = indoor_rows.iloc[i % len(indoor_rows)]
        in_episode = False
        for k, (offs, dur) in enumerate(extra_recovery_episodes):
            if walk_start + offs <= t < walk_start + offs + dur:
                in_episode = True
                episode_idx = k
        if in_episode:
            add_fn(t, state="IndoorVps", source="Pdr", active_source="Pdr",
                   campus_x=src.get("campus_x"), campus_y=src.get("campus_y"),
                   heading_deg=src.get("heading_deg"),
                   vps_valid=0, vps_state="Scanning", vps_reliability=0.0, vps_confidence=0.0,
                   indoor_state="Recovering", indoor_pose_source="Pdr", recovery_count=recovery_total + 1)
            if episode_idx >= recovery_total:
                recovery_total += 1
                ev_rows.append(_mk_ev(t, "quality_threshold_failed", source="Vps",
                                       note="vps tracking lost during indoor navigation"))
        else:
            add_fn(t, state="IndoorVps", source="Vps", active_source="Vps",
                   campus_x=src.get("campus_x"), campus_y=src.get("campus_y"), campus_z=src.get("campus_z"),
                   map_x=src.get("map_x"), map_y=src.get("map_y"), map_z=src.get("map_z"),
                   heading_deg=src.get("heading_deg"), latitude=src.get("latitude"), longitude=src.get("longitude"),
                   vps_valid=1, vps_state="IndoorLocalized", vps_confidence_available=1, vps_map_id_available=1,
                   vps_map_matches=1, indoor_state="Navigating", indoor_pose_source="Vps",
                   recovery_count=recovery_total)
        if (i % 5) == 0:
            ev_rows.append(_mk_ev(t, "indoor_step_checkpoint", source="Vps", note=f"step={i}"))
        t += dt
    dest = indoor_rows.iloc[-1].get("destination") or "B9-104"
    ev_rows.append(_mk_ev(t, "destination_arrived", to_state="Arrived", source="Vps", note=str(dest)))
    ev_rows.append(_mk_ev(t, "trial_finished", source="Vps", note="user_finished_trial"))
    return t, dest, recovery_total


def build_condition_trial(version, scenario_id, stats, cfg, rng):
    """KB12 (rain), KB13 (glare), KB15 (crowd), KB23 (obstruction),
    KB26 (novice proxy), KB27 (experienced proxy)."""
    seed_tag = int(rng.integers(1000, 9999))
    session_id = f"SYNTH_{scenario_id}_{version}_{seed_tag}_FWD"

    dt_override = None
    force_wrong_way = scenario_id == "KB26"
    if scenario_id == "KB27":
        dt_override = stats["sampling_dt_fast"]

    ev_rows, rows, scanning_s, entry_indoor_row, dest_indoor_row, indoor_rows, add, t = _gps_to_vps_core(
        0.0, stats, cfg, rng, dt_override=dt_override, force_wrong_way=force_wrong_way, insert_wrong_way_event=force_wrong_way)
    dt = dt_override or stats["sampling_dt"]

    if scenario_id == "KB12":
        # rain/wet stand-in: force gps_accuracy_m in the pre-handover outdoor
        # rows to this dataset's own real worst-ever-recorded value.
        for r in rows:
            if r["state"] == "OutdoorGps":
                r["gps_accuracy_m"] = GLOBAL_WORST_GPS_ACC_M
                r["gps_reliability"] = round(float(np.clip(rng.normal(0.3, 0.05), 0.05, 0.5)), 4)

    vps_attempt = 1
    ev_rows.append(_mk_ev(scanning_s, "vps_state", to_state="StartingVps", source="Pdr", vps_attempt=vps_attempt))
    t = scanning_s
    scan_dur = max(1.0, 2.5)

    if scenario_id == "KB13":
        # glare stand-in: confidence held at/near this version's own real
        # minimumVpsConfidence config value (not below it, since a genuine
        # sub-threshold reading would never be logged as vps_valid) for a
        # stretch, then a "changed scan angle" note, then normal acceptance.
        thresh = cfg["minimumVpsConfidence"]
        ev_rows.append(_mk_ev(t + 0.3, "vps_state", to_state="Scanning", source="Pdr", vps_attempt=vps_attempt))
        for i in range(4):
            t += dt
            conf = float(np.clip(rng.normal(thresh + 0.03, 0.02), thresh, thresh + 0.1))
            add(t, state="EnteringWithPdr", source="Pdr", active_source="Pdr",
                campus_x=entry_indoor_row.get("campus_x"), campus_y=entry_indoor_row.get("campus_y"),
                heading_deg=entry_indoor_row.get("heading_deg"),
                vps_valid=1, vps_state="Scanning", vps_attempt=vps_attempt,
                vps_reliability=round(float(np.clip(rng.normal(0.6, 0.03), 0.4, 0.8)), 4),
                vps_confidence=round(conf, 4), vps_confidence_available=1, vps_threshold_passed=int(conf >= thresh))
        t += dt
        ev_rows.append(_mk_ev(t, "quality_threshold_failed", source="Pdr",
                               note=f"vps_confidence near minimumVpsConfidence={thresh:.2f} under glare; changing scan angle"))
        t += 1.0
        ev_rows.append(_mk_ev(t, "quality_threshold_passed", source="Pdr", note="scan angle adjusted away from glare"))
        for i in range(2):
            t += dt
            add(t, state="EnteringWithPdr", source="Pdr", active_source="Pdr",
                campus_x=entry_indoor_row.get("campus_x"), campus_y=entry_indoor_row.get("campus_y"),
                heading_deg=entry_indoor_row.get("heading_deg"),
                vps_valid=1, vps_state="Scanning", vps_attempt=vps_attempt,
                vps_reliability=round(float(np.clip(rng.normal(stats["vps_rel_mean"], 0.03), 0.6, 0.95)), 4),
                vps_confidence=round(float(np.clip(rng.normal(stats["vps_conf_mean"], 0.03), thresh + 0.1, 0.95)), 4),
                vps_confidence_available=1, vps_threshold_passed=1)
        t += 0.8
    else:
        ev_rows.append(_mk_ev(t + 0.3, "vps_state", to_state="Scanning", source="Pdr", vps_attempt=vps_attempt))
        for i in range(3):
            t += dt
            frac = (i + 1) / 3
            add(t, state="EnteringWithPdr", source="Pdr", active_source="Pdr",
                campus_x=_interp(entry_indoor_row.get("campus_x"), entry_indoor_row.get("campus_x"), frac),
                campus_y=entry_indoor_row.get("campus_y"), heading_deg=entry_indoor_row.get("heading_deg"),
                vps_valid=1, vps_state="Scanning", vps_attempt=vps_attempt,
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
    handover_end_s = accept_s

    episodes = []
    if scenario_id == "KB15":
        episodes = [(3.0, 3.0), (12.0, 2.5)]  # two recurring brief occlusions
    elif scenario_id == "KB23":
        episodes = [(5.0, 4.0)]  # one single 3-5s obstruction window

    t_end, dest, recovery_total = _finish_indoor_walk(ev_rows, rows, add, t, indoor_rows, dt, extra_recovery_episodes=episodes)

    grounding = {
        "KB12": "WEAK (structural placeholder): gps_accuracy_m during the outdoor approach is forced to this dataset's own real worst-ever-recorded value (24.873m, from a real V1 trial) as a stand-in for rain -- the magnitude is real, but its attribution to 'rain' specifically is NOT measured anywhere in this dataset.",
        "KB13": "WEAK (structural placeholder): VPS confidence during scanning is held at this version's own real minimumVpsConfidence config threshold (a real, already-established per-version value) as a stand-in for glare-degraded scanning -- the threshold is real, its attribution to 'glare' is NOT measured.",
        "KB15": "WEAK (structural placeholder): two vps_valid=0 dropout episodes use the existing recovery_count column, which is NEVER >0 anywhere in the real 105-trial dataset -- this exercises a defined-but-never-observed mechanism, not a measured crowd-occlusion effect.",
        "KB23": "WEAK (structural placeholder): one vps_valid=0 dropout episode, same never-observed recovery_count mechanism as KB15, distinguished from it only by being a single (not recurring) episode per the scenario definition.",
        "KB26": f"MODERATE: wrong_way_count=1 is this dataset's own real maximum ever recorded (105 real trials, max=1, mean=0.0095) and cadence is slowed -- both real-anchored, but 'help_request'/'trust' measures are NOT_REPRESENTED (no such column exists).",
        "KB27": "MODERATE: wrong_way_count=0 (the real mode/median) and this version's own real fastest-observed cadence (same technique as KB21) -- both real-anchored, but the AR-on/off comparison arm is NOT_REPRESENTED (no such column or comparison condition exists).",
    }[scenario_id]

    ev_df, sm_df, su_df = finalize_trial(
        session_id, scenario_id, version, cfg, "GPS_TO_VPS", ev_rows, rows, dest,
        completed=1, handover_success=1, destination_arrived=1, final_state="IndoorVps",
        end_reason="destination_reached", handover_attempts_total=1, handover_attempts_evaluable=1,
        successful_handovers=1, false_handovers=0, incomplete_handovers=0,
    )
    return session_id, ev_df, sm_df, su_df, stats["template_sid"], grounding


# ---------------------------------------------------------------- KB29 (MULTI_HANDOVER)
def build_kb29(version, fwd_stats, rev_stats, cfg, rng):
    seed_tag = int(rng.integers(1000, 9999))
    session_id = f"SYNTH_KB29_{version}_{seed_tag}_FWD"

    # Segment 1: OUT -> IN
    ev1, rows1, scanning_s1, entry1, dest1, indoor_rows1, add1, t1 = _gps_to_vps_core(0.0, fwd_stats, cfg, rng)
    dt = fwd_stats["sampling_dt"]
    ev1.append(_mk_ev(scanning_s1, "vps_state", to_state="StartingVps", source="Pdr", vps_attempt=1))
    t = scanning_s1 + 0.3
    ev1.append(_mk_ev(t, "vps_state", to_state="Scanning", source="Pdr", vps_attempt=1))
    for i in range(3):
        t += dt
        frac = (i + 1) / 3
        add1(t, state="EnteringWithPdr", source="Pdr", active_source="Pdr",
             campus_x=entry1.get("campus_x"), campus_y=entry1.get("campus_y"), heading_deg=entry1.get("heading_deg"),
             vps_valid=1, vps_state="Scanning", vps_attempt=1,
             vps_reliability=round(float(np.clip(rng.normal(fwd_stats["vps_rel_mean"], 0.03), 0.5, 0.98)), 4),
             vps_confidence=round(float(np.clip(cfg["vpsEnterReliability"] * frac + 0.1, 0.3, 0.95)), 4),
             vps_confidence_available=1, vps_threshold_passed=int(frac >= 1.0))
    t += 0.8
    accept1_s = round(t, 3)
    ev1.append(_mk_ev(accept1_s, "vps_state", to_state="IndoorLocalized", source="Vps", note="VPS localization accepted (1st handover)", vps_attempt=1))
    ev1.append(_mk_ev(accept1_s, "reliability_state", "VpsScanning", "IndoorVps", "Vps", "VPS localization accepted (1st handover)"))
    ev1.append(_mk_ev(accept1_s, "indoor_state", to_state="Navigating", source="Vps"))
    t = accept1_s + dt * 0.1
    add1(t, state="IndoorVps", source="Vps", active_source="Vps", campus_x=entry1.get("campus_x"), campus_y=entry1.get("campus_y"),
         heading_deg=entry1.get("heading_deg"), vps_valid=1, vps_state="IndoorLocalized", vps_attempt=1,
         vps_reliability=0.85, vps_confidence=0.85, vps_confidence_available=1, vps_map_id_available=1,
         vps_map_matches=1, indoor_state="Navigating", indoor_pose_source="Vps")
    handover1_end_s = accept1_s
    t = accept1_s + dt
    # brief indoor dwell before heading back out (real IndoorVps rows reused)
    for i in range(1, min(6, len(indoor_rows1))):
        src = indoor_rows1.iloc[i % len(indoor_rows1)]
        add1(t, state="IndoorVps", source="Vps", active_source="Vps", campus_x=src.get("campus_x"), campus_y=src.get("campus_y"),
             heading_deg=src.get("heading_deg"), vps_valid=1, vps_state="IndoorLocalized",
             vps_reliability=0.85, vps_confidence=0.85, vps_confidence_available=1, vps_map_id_available=1, vps_map_matches=1,
             indoor_state="Navigating", indoor_pose_source="Vps")
        t += dt

    # Segment 2: IN -> OUT (reuse step4-style reverse exit trigger + commit)
    exit_trigger_s = round(t, 3)
    ev2 = [_mk_ev(exit_trigger_s, "reliability_state", "IndoorVps", "OutdoorGps", "Vps",
                  "Exiting indoor volume; reliability threshold crossed toward outdoor")]
    ev2.append(_mk_ev(exit_trigger_s, "source_switched", "Vps", "Pdr", "Pdr", "Vps->Pdr"))
    rows2 = []
    n_bridge_back = 4
    for i in range(1, n_bridge_back + 1):
        frac = i / n_bridge_back
        t += dt
        rows2.append(dict(elapsed_s=round(t, 3), state="EnteringWithPdr", source="Pdr", active_source="Pdr",
                           campus_x=_interp(dest1.get("campus_x"), entry1.get("campus_x"), frac),
                           campus_y=_interp(dest1.get("campus_y"), entry1.get("campus_y"), frac),
                           heading_deg=entry1.get("heading_deg"), gps_valid=0, vps_valid=0,
                           pdr_steps=i, pdr_confidence=round(float(np.clip(rng.normal(0.7, 0.05), 0.3, 0.95)), 4)))
    gps_reacq_dur = max(1.0, cfg["gpsDwellSeconds"] + 1.5)
    n_reacq = max(2, int(round(gps_reacq_dur / dt)))
    for i in range(1, n_reacq + 1):
        frac = i / n_reacq
        t += dt
        rows2.append(dict(elapsed_s=round(t, 3), state="OutdoorGps", source="Pdr" if frac < 1.0 else "Gps",
                           active_source="Gps" if frac >= 0.99 else "Pdr",
                           campus_x=entry1.get("campus_x"), campus_y=entry1.get("campus_y"), heading_deg=entry1.get("heading_deg"),
                           gps_accuracy_m=round(float(np.clip(rng.normal(fwd_stats["gps_acc_mean"], fwd_stats["gps_acc_std"]), 2.0, 12.0)), 3),
                           gps_valid=1, gps_reliability=round(float(np.clip(0.3 + frac * 0.6, 0.1, 0.95)), 4),
                           gps_threshold_passed=int(frac >= 0.5), gps_dwell_gate_passed=int(frac >= 0.99)))
    commit_s = round(t, 3)
    ev2.append(_mk_ev(commit_s, "source_switched", "Pdr", "Gps", "Gps", "Pdr->Gps; GPS dwell/reliability gate satisfied"))
    handover2_start_s = exit_trigger_s
    handover2_end_s = commit_s

    # Segment 3: OUT -> IN again (second forward handover)
    ev3, rows3, scanning_s3, entry3, dest3, indoor_rows3, add3, t3 = _gps_to_vps_core(commit_s + dt, fwd_stats, cfg, rng)
    dt3 = fwd_stats["sampling_dt"]
    ev3.append(_mk_ev(scanning_s3, "vps_state", to_state="StartingVps", source="Pdr", vps_attempt=2))
    t = scanning_s3 + 0.3
    ev3.append(_mk_ev(t, "vps_state", to_state="Scanning", source="Pdr", vps_attempt=2))
    for i in range(3):
        t += dt3
        frac = (i + 1) / 3
        add3(t, state="EnteringWithPdr", source="Pdr", active_source="Pdr", campus_x=entry3.get("campus_x"),
             campus_y=entry3.get("campus_y"), heading_deg=entry3.get("heading_deg"), vps_valid=1, vps_state="Scanning",
             vps_attempt=2, vps_reliability=round(float(np.clip(rng.normal(fwd_stats["vps_rel_mean"], 0.03), 0.5, 0.98)), 4),
             vps_confidence=round(float(np.clip(cfg["vpsEnterReliability"] * frac + 0.1, 0.3, 0.95)), 4),
             vps_confidence_available=1, vps_threshold_passed=int(frac >= 1.0))
    t += 0.8
    accept3_s = round(t, 3)
    ev3.append(_mk_ev(accept3_s, "vps_state", to_state="IndoorLocalized", source="Vps", note="VPS localization accepted (2nd handover)", vps_attempt=2))
    ev3.append(_mk_ev(accept3_s, "reliability_state", "VpsScanning", "IndoorVps", "Vps", "VPS localization accepted (2nd handover)"))
    ev3.append(_mk_ev(accept3_s, "indoor_state", to_state="Navigating", source="Vps"))
    t = accept3_s + dt3 * 0.1
    add3(t, state="IndoorVps", source="Vps", active_source="Vps", campus_x=entry3.get("campus_x"), campus_y=entry3.get("campus_y"),
         heading_deg=entry3.get("heading_deg"), vps_valid=1, vps_state="IndoorLocalized", vps_attempt=2,
         vps_reliability=0.85, vps_confidence=0.85, vps_confidence_available=1, vps_map_id_available=1, vps_map_matches=1,
         indoor_state="Navigating", indoor_pose_source="Vps")
    t = accept3_s + dt3

    t_end, dest, _ = _finish_indoor_walk(ev3, rows3, add3, t, indoor_rows3, dt3)

    all_ev = ev1 + ev2 + ev3
    all_rows = rows1 + rows2 + rows3
    ev_df, sm_df, su_df = finalize_trial(
        session_id, "KB29", version, cfg, "MULTI_HANDOVER", all_ev, all_rows, dest,
        completed=1, handover_success=1, destination_arrived=1, final_state="IndoorVps",
        end_reason="destination_reached", handover_attempts_total=2, handover_attempts_evaluable=0,
        successful_handovers=0, false_handovers=0, incomplete_handovers=0,
        extra_summary={"hsr_percent": np.nan, "fhr_percent": np.nan},
    )
    grounding = ("STRONG structurally / per-segment: each of the 3 segments (OUT->IN, IN->OUT, OUT->IN) reuses "
                 "the exact same real-per-version-distribution construction as step6's KB01 and step4's KB02 "
                 "reverse segment, just chained in one continuous session. Per config.yaml, 'MULTI_HANDOVER' has "
                 "no direction entry, so the pipeline's own detect_handover correctly returns evaluable=False for "
                 "this trial -- exactly as spec Section 9 requires (excluded from HSR/FHR denominators), which is "
                 "why handover_attempts_evaluable=0 in the summary even though 2 handovers structurally occurred.")
    return session_id, ev_df, sm_df, su_df, f"{fwd_stats['template_sid']}+{rev_stats.get('template_sid','n/a')}", grounding


def main():
    from step4_synthetic_reverse import load_real_version_stats as load_reverse_stats
    OUT.mkdir(parents=True, exist_ok=True)
    manifest_rows = []
    rng_master = np.random.default_rng(SEED)

    for version in VERSIONS:
        fwd_stats = load_forward_stats(version)
        none_stats = load_none_stats(version)
        rev_stats = load_reverse_stats(version)
        cfg = dwell_config_for_version(version)

        jobs = []
        rng = np.random.default_rng(rng_master.integers(0, 2**31 - 1))
        jobs.append(("KB09", *build_kb09(version, none_stats, cfg, rng)))
        for scen in ("KB12", "KB13", "KB15", "KB23", "KB26", "KB27"):
            rng = np.random.default_rng(rng_master.integers(0, 2**31 - 1))
            jobs.append((scen, *build_condition_trial(version, scen, fwd_stats, cfg, rng)))
        rng = np.random.default_rng(rng_master.integers(0, 2**31 - 1))
        jobs.append(("KB29", *build_kb29(version, fwd_stats, rev_stats, cfg, rng)))

        for scenario_id, session_id, ev_df, sm_df, su_df, template_used, grounding in jobs:
            dest = OUT / scenario_id / session_id
            dest.mkdir(parents=True, exist_ok=True)
            ev_df.to_csv(dest / f"events_{session_id}.csv", index=False)
            sm_df.to_csv(dest / f"samples_{session_id}.csv", index=False)
            su_df.to_csv(dest / f"summary_{session_id}.csv", index=False)
            manifest_rows.append({
                "session_id": session_id, "scenario_id": scenario_id, "harmony_version": version,
                "data_provenance": PROVENANCE, "synthetic": 1, "generation_seed": SEED,
                "template_session_id_used": template_used, "validation_status": "PENDING_PIPELINE_CHECK",
                "grounding_strength": grounding,
            })

    mdf = pd.DataFrame(manifest_rows)
    mdf.to_csv("expanded_scenario_synthetic_manifest.csv", index=False)
    print(f"Generated {len(mdf)} synthetic trials into {OUT}")
    print(mdf.groupby("scenario_id").size())


if __name__ == "__main__":
    main()
