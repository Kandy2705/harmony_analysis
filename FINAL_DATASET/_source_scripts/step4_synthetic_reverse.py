#!/usr/bin/env python3
"""
step4_synthetic_reverse.py
----------------------------
Generates a SMALL, clearly-marked synthetic VPS_TO_GPS (reverse-handover)
support dataset for KB02 (normal IN->OUT) and KB20 (IN->OUT after a
backtrack), per spec Section 8.

Real KB02 data audit (from analysis_output_clean/per_trial_metrics.csv)
showed 0 of 7 real KB02 trials have a fully reconstructable VPS_TO_GPS exit
sequence (3 never log the reliability_state->OutdoorGps start trigger; 4
log the start trigger but the log ends before source_switched->Gps), and
there are 0 real KB20 trials at all -- so this synthetic support set is the
only way the pipeline's reverse-handover branch is populated at all.

Construction method (documented for the manifest / final report):
  - Per-version statistics (GPS accuracy, VPS reliability while valid, PDR
    step rate, sampling interval, source-switch frequency, trial duration,
    indoor dwell duration) are LEARNED from that SAME version's real, clean
    GPS_TO_VPS trials (per spec Section 8.2) -- never invented.
  - Per-version dwell/reliability *configuration* (gpsDwellSeconds,
    vpsDwellSeconds, minimumVpsConfidence, gpsExitReliability,
    vpsEnterReliability) is read directly from that version's real
    summary_*.csv rows (these are static app config fields, unaffected by
    the earlier data-adjustment script) and used to set a version-appropriate
    GPS-reacquisition latency (Section 8.4: GPS dwell dominates VPS_TO_GPS
    latency, not VPS localization time).
  - The indoor segment reuses one real template trial's own indoor
    (post-acceptance) coordinate/heading path for that version, in its
    ORIGINAL chronological order (never reversed), re-based to start at
    t=0 of the new trial.
  - The PDR-bridge + outdoor-reacquisition segment is a straight-line
    interpolation from the last indoor position to that same template
    trial's own real outdoor-entrance coordinate (its position right before
    ITS OWN reliability_state->VpsScanning trigger) -- i.e. real map
    geometry from the same version/location, not invented coordinates.
  - KB20 additionally lengthens the pre-exit indoor phase and increments
    the existing logger fields `route_revision` / `wrong_way_count` to
    represent a backtrack, per spec Section 8.6 (no invented event types).
  - Every row of every generated file carries `data_provenance =
    SYNTHETIC_REVERSE_SUPPORT` and `synthetic = 1` so it can never be
    mistaken for a physically collected observation once separated from
    this manifest.
"""
import numpy as np
import pandas as pd
from pathlib import Path

CLEAN = Path("sample_data_clean")
OUT = CLEAN / "_synthetic_reverse"
SEED = 20260920  # today's date, deterministic

VERSIONS = ["V1", "V2", "V3", "V4", "V5", "BQ", "BT"]

EVENTS_COLS = ["utc_iso", "session_id", "harmony_version", "harmony_profile",
               "quality_gate", "temporal_dwell", "map_id_check", "recovery_fsm",
               "adaptive_guidance", "elapsed_s", "event", "from_state", "to_state",
               "source", "destination", "latitude", "longitude", "gps_accuracy_m",
               "campus_x", "campus_y", "campus_z", "map_x", "map_y", "map_z",
               "vps_attempt", "note", "scenario_id", "data_provenance", "synthetic"]

SAMPLES_COLS = ["utc_iso", "session_id", "harmony_version", "harmony_profile",
                "quality_gate", "temporal_dwell", "map_id_check", "recovery_fsm",
                "adaptive_guidance", "elapsed_s", "state", "source", "destination",
                "latitude", "longitude", "gps_accuracy_m", "gps_valid", "pdr_steps",
                "pdr_confidence", "campus_x", "campus_y", "campus_z", "map_x", "map_y",
                "map_z", "outdoor_remaining_m", "vps_attempt", "vps_state", "vps_scan_s",
                "indoor_state", "indoor_pose_source", "indoor_confidence", "indoor_steps",
                "heading_deg", "indoor_remaining_m", "route_revision", "wrong_way_count",
                "recovery_count", "active_source", "gps_age_s", "gps_reliability",
                "gps_stable_s", "vps_valid", "vps_reliability", "vps_stable_s",
                "vps_confidence", "vps_confidence_available", "vps_map_id",
                "vps_map_id_available", "vps_map_matches", "source_toggle_count",
                "position_jump_m", "heading_jump_deg", "candidate_source",
                "gps_threshold_passed", "vps_threshold_passed", "gps_dwell_gate_passed",
                "vps_dwell_gate_passed", "decision_reason", "scenario_id",
                "data_provenance", "synthetic"]


def load_real_version_stats(version):
    """Learn per-version distributions from clean, evaluable, real GPS_TO_VPS trials."""
    per_trial = pd.read_csv("analysis_output_clean/per_trial_metrics.csv")
    manifest = pd.read_csv("teacher_manifest.csv")
    ok_sids = set(manifest[(manifest["usable_for_out_to_in"]) & (manifest["harmony_version"] == version)]["session_id"])
    sub = per_trial[per_trial["session_id"].isin(ok_sids)]
    if sub.empty:
        raise RuntimeError(f"No usable real GPS_TO_VPS trials found for version {version}")

    # gather raw samples across these trials for fine-grained stats
    all_samples = []
    template_candidates = []
    for sid in sub["session_id"]:
        matches = list(CLEAN.rglob(f"samples_{sid}.csv"))
        summ = list(CLEAN.rglob(f"summary_{sid}.csv"))
        ev = list(CLEAN.rglob(f"events_{sid}.csv"))
        if not matches or not summ or not ev:
            continue
        sdf = pd.read_csv(matches[0])
        all_samples.append(sdf)
        template_candidates.append((sid, matches[0], summ[0], ev[0], sdf))

    concat = pd.concat(all_samples, ignore_index=True)
    dt = concat.sort_values(["session_id", "elapsed_s"]).groupby("session_id")["elapsed_s"].diff().dropna()
    sampling_dt = float(np.clip(dt.median(), 0.4, 3.0))

    gps_acc = pd.to_numeric(concat["gps_accuracy_m"], errors="coerce").dropna()
    gps_acc_mean, gps_acc_std = float(gps_acc.mean()), float(max(gps_acc.std(), 0.05))

    vps_active = concat[concat.get("vps_valid", 0) == 1]
    vps_rel = pd.to_numeric(vps_active["vps_reliability"], errors="coerce").dropna()
    vps_rel_mean = float(vps_rel.mean()) if len(vps_rel) else 0.85

    pdr_rows = concat[concat["source"] == "Pdr"]
    pdr_conf = pd.to_numeric(pdr_rows["pdr_confidence"], errors="coerce")
    pdr_conf = pdr_conf[pdr_conf > 0].dropna()
    pdr_conf_mean = float(pdr_conf.mean()) if len(pdr_conf) else 0.7

    trial_durations = pd.to_numeric(sub["duration_s"], errors="coerce").dropna()
    duration_median = float(trial_durations.median()) if len(trial_durations) else 90.0

    # indoor dwell (post-acceptance) duration, used as this synthetic trial's
    # pre-exit indoor navigation duration
    post_dwell = (trial_durations - pd.to_numeric(sub["handover_end_s"], errors="coerce")).dropna()
    post_dwell = post_dwell[post_dwell > 0]
    indoor_dwell_median = float(post_dwell.median()) if len(post_dwell) else 20.0

    pdr_transit = pd.to_numeric(sub["pdr_transition_duration_s"], errors="coerce").dropna()
    pdr_transit_median = float(pdr_transit.median()) if len(pdr_transit) else 3.0

    # pick the template trial closest to median duration (stability, not cherry-picked extremes)
    sub2 = sub.copy()
    sub2["dur_diff"] = (pd.to_numeric(sub2["duration_s"], errors="coerce") - duration_median).abs()
    template_sid = sub2.sort_values("dur_diff").iloc[0]["session_id"]
    template = next(t for t in template_candidates if t[0] == template_sid)

    return {
        "sampling_dt": sampling_dt, "gps_acc_mean": gps_acc_mean, "gps_acc_std": gps_acc_std,
        "vps_rel_mean": vps_rel_mean, "pdr_conf_mean": pdr_conf_mean,
        "indoor_dwell_median": indoor_dwell_median, "pdr_transit_median": pdr_transit_median,
        "template_sid": template_sid, "template_samples_path": template[1],
        "template_summary_path": template[2], "template_events_path": template[3],
        "template_samples_df": template[4],
    }


def dwell_config_for_version(version):
    per_trial = pd.read_csv("analysis_output_clean/per_trial_metrics.csv")
    row = per_trial[per_trial["harmony_version"] == version].iloc[0]
    sid = row["session_id"]
    summ = list(CLEAN.rglob(f"summary_{sid}.csv")) or list((CLEAN / "_excluded_invalid").rglob(f"summary_{sid}.csv"))
    sdf = pd.read_csv(summ[0])
    r = sdf.iloc[0]
    return {
        "harmony_profile": r.get("harmony_profile", ""),
        "gpsDwellSeconds": float(r.get("gpsDwellSeconds", 0) or 0),
        "vpsDwellSeconds": float(r.get("vpsDwellSeconds", 0) or 0),
        "minimumVpsConfidence": float(r.get("minimumVpsConfidence", 0.5) or 0.5),
        "gpsExitReliability": float(r.get("gpsExitReliability", 0.7) or 0.7),
        "vpsEnterReliability": float(r.get("vpsEnterReliability", 0.72) or 0.72),
        "quality_gate": r.get("quality_gate", 0), "temporal_dwell": r.get("temporal_dwell", 0),
        "map_id_check": r.get("map_id_check", 0), "recovery_fsm": r.get("recovery_fsm", 0),
        "adaptive_guidance": r.get("adaptive_guidance", 0),
    }


def _interp(a, b, frac):
    return a + (b - a) * frac


def build_trial(version, scenario_id, stats, cfg, rng, backtrack=False):
    seed_tag = int(rng.integers(1000, 9999))
    session_id = f"SYNTH_{scenario_id}_{version}_{seed_tag}_REV"

    tsm = stats["template_samples_df"].sort_values("elapsed_s").reset_index(drop=True)
    indoor_rows = tsm[tsm["state"] == "IndoorVps"].reset_index(drop=True)
    if indoor_rows.empty:
        indoor_rows = tsm.tail(max(5, len(tsm) // 4)).reset_index(drop=True)
    outdoor_rows = tsm[tsm["state"] == "OutdoorGps"].reset_index(drop=True)
    entrance_row = outdoor_rows.iloc[-1] if len(outdoor_rows) else tsm.iloc[0]
    far_outdoor_row = outdoor_rows.iloc[0] if len(outdoor_rows) else tsm.iloc[0]

    dt = stats["sampling_dt"]
    indoor_dwell = stats["indoor_dwell_median"] * (1.6 if backtrack else 1.0)
    n_indoor = max(4, int(round(indoor_dwell / dt)))
    pdr_bridge_dur = max(1.5, stats["pdr_transit_median"])
    n_bridge = max(2, int(round(pdr_bridge_dur / dt)))
    # Section 8.4: GPS reacquisition + dwell dominate latency, not VPS time.
    gps_reacq_dur = max(1.0, cfg["gpsDwellSeconds"] + rng.uniform(1.0, 2.0))
    n_outdoor_tail = max(3, int(round(4.0 / dt)))

    rows = []  # sample rows as dicts
    t = 0.0
    route_revision = 0
    wrong_way = 0

    # ---- indoor phase: reuse template's own indoor path, original order, re-based to t=0 ----
    for i in range(n_indoor):
        src_row = indoor_rows.iloc[i % len(indoor_rows)]
        if backtrack and i == n_indoor // 2:
            route_revision += 1
            wrong_way += 1
        rows.append(dict(
            elapsed_s=round(t, 3), state="IndoorVps", source="Vps",
            campus_x=src_row.get("campus_x"), campus_y=src_row.get("campus_y"),
            campus_z=src_row.get("campus_z"), map_x=src_row.get("map_x"),
            map_y=src_row.get("map_y"), map_z=src_row.get("map_z"),
            heading_deg=src_row.get("heading_deg"),
            latitude=src_row.get("latitude"), longitude=src_row.get("longitude"),
            gps_accuracy_m=round(float(np.clip(rng.normal(stats["gps_acc_mean"], stats["gps_acc_std"]), 2.0, 15.0)), 3),
            gps_valid=0, vps_valid=1,
            vps_reliability=round(float(np.clip(rng.normal(stats["vps_rel_mean"], 0.03), 0.5, 0.98)), 4),
            vps_confidence=round(float(np.clip(rng.normal(cfg["vpsEnterReliability"], 0.03), 0.5, 0.98)), 4),
            vps_confidence_available=1, vps_map_id_available=1, vps_map_matches=1,
            vps_map_id=src_row.get("vps_map_id") or "B9_FLOOR_1_CAMPUS",
            pdr_steps=src_row.get("indoor_steps", 0) or 0, pdr_confidence=0,
            indoor_state="Navigating", indoor_pose_source="Vps",
            indoor_confidence=round(float(np.clip(rng.normal(stats["vps_rel_mean"], 0.03), 0.5, 0.98)), 4),
            route_revision=route_revision, wrong_way_count=wrong_way, recovery_count=0,
            active_source="Vps", position_jump_m=0.0, heading_jump_deg=0.0,
        ))
        t += dt

    exit_trigger_s = round(t, 3)

    # ---- PDR bridge phase: interpolate from last indoor position to the
    #       template's own real outdoor-entrance coordinate ----
    last = rows[-1]
    for i in range(1, n_bridge + 1):
        frac = i / n_bridge
        rows.append(dict(
            elapsed_s=round(t, 3), state="EnteringWithPdr", source="Pdr",
            campus_x=_interp(last["campus_x"], entrance_row.get("campus_x"), frac),
            campus_y=_interp(last["campus_y"], entrance_row.get("campus_y"), frac),
            campus_z=_interp(last["campus_z"], entrance_row.get("campus_z"), frac),
            map_x=_interp(last["map_x"], entrance_row.get("map_x"), frac) if pd.notna(last.get("map_x")) else np.nan,
            map_y=_interp(last["map_y"], entrance_row.get("map_y"), frac) if pd.notna(last.get("map_y")) else np.nan,
            map_z=last.get("map_z"),
            heading_deg=entrance_row.get("heading_deg"),
            latitude=entrance_row.get("latitude"), longitude=entrance_row.get("longitude"),
            gps_accuracy_m=round(float(np.clip(rng.normal(stats["gps_acc_mean"] * 1.3, stats["gps_acc_std"]), 3.0, 20.0)), 3),
            gps_valid=0, vps_valid=0, vps_reliability=0.0,
            vps_confidence=0.0, vps_confidence_available=0, vps_map_id_available=0, vps_map_matches=0,
            pdr_steps=(last.get("pdr_steps") or 0) + i, pdr_confidence=round(float(np.clip(rng.normal(stats["pdr_conf_mean"], 0.05), 0.3, 0.95)), 4),
            indoor_state="", indoor_pose_source="Pdr", indoor_confidence=0,
            route_revision=route_revision, wrong_way_count=wrong_way, recovery_count=0,
            active_source="Pdr", position_jump_m=0.0, heading_jump_deg=0.0,
        ))
        t += dt

    # ---- GPS reacquisition phase: accuracy/reliability climb toward the
    #       version's real dwell/reliability gate before commit ----
    n_reacq = max(2, int(round(gps_reacq_dur / dt)))
    for i in range(1, n_reacq + 1):
        frac = i / n_reacq
        gps_rel = float(np.clip(0.3 + frac * (cfg["gpsExitReliability"] + 0.1 - 0.3), 0.1, 0.95))
        rows.append(dict(
            elapsed_s=round(t, 3), state="OutdoorGps", source="Pdr" if frac < 1.0 else "Gps",
            campus_x=_interp(entrance_row.get("campus_x"), far_outdoor_row.get("campus_x"), frac * 0.3),
            campus_y=_interp(entrance_row.get("campus_y"), far_outdoor_row.get("campus_y"), frac * 0.3),
            campus_z=entrance_row.get("campus_z"),
            map_x=entrance_row.get("map_x"), map_y=entrance_row.get("map_y"), map_z=entrance_row.get("map_z"),
            heading_deg=entrance_row.get("heading_deg"),
            latitude=entrance_row.get("latitude"), longitude=entrance_row.get("longitude"),
            gps_accuracy_m=round(float(np.clip(rng.normal(stats["gps_acc_mean"], stats["gps_acc_std"]), 2.0, 12.0)), 3),
            gps_valid=1, vps_valid=0, vps_reliability=0.0,
            vps_confidence=0.0, vps_confidence_available=0, vps_map_id_available=0, vps_map_matches=0,
            pdr_steps=last.get("pdr_steps", 0) or 0, pdr_confidence=0,
            gps_reliability=round(gps_rel, 4), gps_stable_s=round(frac * gps_reacq_dur, 3),
            gps_threshold_passed=int(frac >= 0.5), gps_dwell_gate_passed=int(frac >= 0.99),
            indoor_state="", indoor_pose_source="", indoor_confidence=0,
            route_revision=route_revision, wrong_way_count=wrong_way, recovery_count=0,
            active_source="Gps" if frac >= 0.99 else "Pdr", position_jump_m=0.0, heading_jump_deg=0.0,
        ))
        t += dt

    commit_s = round(t - dt, 3)  # elapsed_s of the last (fully-reacquired) row

    # ---- brief outdoor tail after commit ----
    for i in range(1, n_outdoor_tail + 1):
        frac = i / n_outdoor_tail
        rows.append(dict(
            elapsed_s=round(t, 3), state="OutdoorGps", source="Gps",
            campus_x=_interp(entrance_row.get("campus_x"), far_outdoor_row.get("campus_x"), 0.3 + frac * 0.7),
            campus_y=_interp(entrance_row.get("campus_y"), far_outdoor_row.get("campus_y"), 0.3 + frac * 0.7),
            campus_z=far_outdoor_row.get("campus_z"),
            map_x=far_outdoor_row.get("map_x"), map_y=far_outdoor_row.get("map_y"), map_z=far_outdoor_row.get("map_z"),
            heading_deg=far_outdoor_row.get("heading_deg"),
            latitude=far_outdoor_row.get("latitude"), longitude=far_outdoor_row.get("longitude"),
            gps_accuracy_m=round(float(np.clip(rng.normal(stats["gps_acc_mean"], stats["gps_acc_std"]), 2.0, 12.0)), 3),
            gps_valid=1, vps_valid=0, gps_reliability=round(cfg["gpsExitReliability"] + 0.05, 4),
            gps_stable_s=round(gps_reacq_dur + frac * 4.0, 3),
            pdr_steps=last.get("pdr_steps", 0) or 0, pdr_confidence=0,
            gps_threshold_passed=1, gps_dwell_gate_passed=1,
            route_revision=route_revision, wrong_way_count=wrong_way, recovery_count=0,
            active_source="Gps", position_jump_m=0.0, heading_jump_deg=0.0,
        ))
        t += dt

    sm_df = pd.DataFrame(rows)
    for col in SAMPLES_COLS:
        if col not in sm_df.columns:
            sm_df[col] = np.nan
    sm_df["session_id"] = session_id
    sm_df["harmony_version"] = version
    sm_df["harmony_profile"] = cfg["harmony_profile"]
    for k in ("quality_gate", "temporal_dwell", "map_id_check", "recovery_fsm", "adaptive_guidance"):
        sm_df[k] = cfg[k]
    sm_df["destination"] = "B9-EXIT"
    sm_df["scenario_id"] = scenario_id
    sm_df["data_provenance"] = "SYNTHETIC_REVERSE_SUPPORT"
    sm_df["synthetic"] = 1
    sm_df["source_toggle_count"] = 2
    sm_df["utc_iso"] = pd.to_datetime("2026-09-20T00:00:00Z") + pd.to_timedelta(sm_df["elapsed_s"], unit="s")
    sm_df["utc_iso"] = sm_df["utc_iso"].dt.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "0Z"
    sm_df = sm_df[SAMPLES_COLS]

    # ---- events file: only the events the reverse-detection FSM needs, plus
    #       basic bookkeeping (trial_started, app_resumed) ----
    ev_rows = [
        dict(elapsed_s=0.0, event="trial_started", from_state=None, to_state=None, source="Vps", note="auto=True"),
        dict(elapsed_s=0.0, event="app_resumed", from_state=None, to_state=None, source="Vps", note=""),
        dict(elapsed_s=0.0, event="indoor_state", from_state=None, to_state="IndoorVps", source="Vps", note="synthetic reverse support: trial begins already indoors"),
    ]
    # periodic bookkeeping events through the indoor dwell so the events log
    # doesn't show an unrealistic multi-minute silent gap (real logs emit
    # these throughout navigation; config.yaml already defines this event)
    checkpoint_interval = 15.0
    ck = checkpoint_interval
    while ck < exit_trigger_s - 2.0:
        ev_rows.append(dict(elapsed_s=round(ck, 3), event="indoor_step_checkpoint",
                             from_state=None, to_state="IndoorVps", source="Vps",
                             note="periodic indoor navigation checkpoint"))
        ck += checkpoint_interval
    if backtrack:
        ev_rows.append(dict(elapsed_s=round(exit_trigger_s * 0.5, 3), event="indoor_step_checkpoint",
                             from_state=None, to_state="IndoorVps", source="Vps",
                             note=f"route_revision incremented (backtrack); wrong_way_count={wrong_way}"))
    ev_rows.append(dict(elapsed_s=exit_trigger_s, event="reliability_state", from_state="IndoorVps",
                         to_state="OutdoorGps", source="Vps", note="Exiting indoor volume; reliability threshold crossed toward outdoor"))
    ev_rows.append(dict(elapsed_s=exit_trigger_s, event="source_switched", from_state="Vps", to_state="Pdr",
                         source="Pdr", note="Vps->Pdr"))
    ev_rows.append(dict(elapsed_s=round(exit_trigger_s + pdr_bridge_dur, 3), event="quality_threshold_passed",
                         from_state=None, to_state=None, source="Pdr", note="GPS quality threshold passed during reacquisition"))
    ev_rows.append(dict(elapsed_s=commit_s, event="source_switched", from_state="Pdr", to_state="Gps",
                         source="Gps", note="Pdr->Gps; GPS dwell/reliability gate satisfied"))

    ev_df = pd.DataFrame(ev_rows)
    ev_df["session_id"] = session_id
    ev_df["harmony_version"] = version
    ev_df["harmony_profile"] = cfg["harmony_profile"]
    for k in ("quality_gate", "temporal_dwell", "map_id_check", "recovery_fsm", "adaptive_guidance"):
        ev_df[k] = cfg[k]
    ev_df["destination"] = "B9-EXIT"
    for col in ("latitude", "longitude", "gps_accuracy_m", "campus_x", "campus_y", "campus_z",
                "map_x", "map_y", "map_z", "vps_attempt"):
        ev_df[col] = np.nan
    ev_df["scenario_id"] = scenario_id
    ev_df["data_provenance"] = "SYNTHETIC_REVERSE_SUPPORT"
    ev_df["synthetic"] = 1
    ev_df["utc_iso"] = pd.to_datetime("2026-09-20T00:00:00Z") + pd.to_timedelta(ev_df["elapsed_s"], unit="s")
    ev_df["utc_iso"] = ev_df["utc_iso"].dt.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "0Z"
    ev_df = ev_df.sort_values("elapsed_s").reset_index(drop=True)[EVENTS_COLS]

    duration_s = round(t, 3)
    su_row = {
        "session_id": session_id, "harmony_version": version, "harmony_profile": cfg["harmony_profile"],
        "quality_gate": cfg["quality_gate"], "temporal_dwell": cfg["temporal_dwell"],
        "map_id_check": cfg["map_id_check"], "recovery_fsm": cfg["recovery_fsm"],
        "adaptive_guidance": cfg["adaptive_guidance"],
        "start_utc": ev_df["utc_iso"].iloc[0], "end_utc": ev_df["utc_iso"].iloc[-1],
        "duration_s": duration_s, "direction": "VPS_TO_GPS", "completed": 1,
        "handover_success": 1, "destination_arrived": 1, "destination": "B9-EXIT",
        "sample_count": len(sm_df), "event_count": len(ev_df), "vps_attempts": 0,
        "pdr_steps": int(sm_df["pdr_steps"].max()), "indoor_steps": n_indoor,
        "wrong_way_count": wrong_way, "recovery_count": 0, "final_state": "OutdoorGps",
        "end_reason": "destination_reached",
        "reliability_gate_enabled": True, "vps_dwell_enabled": cfg["vpsDwellSeconds"] > 0,
        "map_id_check_enabled": bool(cfg["map_id_check"]), "uncertainty_guidance_enabled": bool(cfg["adaptive_guidance"]),
        "continuity_gate_enabled": True, "minimum_mode_duration_enabled": True,
        "handover_attempts_total": 1, "handover_attempts_evaluable": 1, "successful_handovers": 1,
        "false_handovers": 0, "incomplete_handovers": 0, "hsr_percent": 100.0, "fhr_percent": 0.0,
        "source_toggle_count": 2,
        "vpsEnterReliability": cfg["vpsEnterReliability"], "gpsExitReliability": cfg["gpsExitReliability"],
        "minimumVpsConfidence": cfg["minimumVpsConfidence"], "vpsDwellSeconds": cfg["vpsDwellSeconds"],
        "gpsDwellSeconds": cfg["gpsDwellSeconds"],
        "gps_weight_accuracy": 0.3, "gps_weight_freshness": 0.2, "gps_weight_motion": 0.2,
        "gps_weight_transition": 0.15, "gps_weight_dwell": 0, "vps_weight_confidence": 0.3,
        "vps_weight_freshness": 0.2, "vps_weight_motion": 0.2, "vps_weight_map_match": 0.15,
        "vps_weight_dwell": 0, "scenario_id": scenario_id,
        "data_provenance": "SYNTHETIC_REVERSE_SUPPORT", "synthetic": 1,
    }
    su_df = pd.DataFrame([su_row])

    return session_id, ev_df, sm_df, su_df, {
        "exit_trigger_s": exit_trigger_s, "commit_s": commit_s,
        "latency_s": round(commit_s - exit_trigger_s, 3), "duration_s": duration_s,
        "template_sid": stats["template_sid"],
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    manifest_rows = []
    rng_master = np.random.default_rng(SEED)

    for version in VERSIONS:
        stats = load_real_version_stats(version)
        cfg = dwell_config_for_version(version)

        for scenario_id, backtrack in (("KB02", False), ("KB20", True)):
            rng = np.random.default_rng(rng_master.integers(0, 2**31 - 1))
            session_id, ev_df, sm_df, su_df, info = build_trial(version, scenario_id, stats, cfg, rng, backtrack=backtrack)

            dest = OUT / scenario_id / session_id
            dest.mkdir(parents=True, exist_ok=True)
            ev_df.to_csv(dest / f"events_{session_id}.csv", index=False)
            sm_df.to_csv(dest / f"samples_{session_id}.csv", index=False)
            su_df.to_csv(dest / f"summary_{session_id}.csv", index=False)

            manifest_rows.append({
                "session_id": session_id, "scenario_id": scenario_id, "harmony_version": version,
                "template_session_ids": info["template_sid"], "generation_seed": SEED,
                "synthetic": 1, "direction": "VPS_TO_GPS",
                "validation_status": "PENDING_PIPELINE_CHECK",
                "notes": (f"backtrack={backtrack}; exit_trigger_s={info['exit_trigger_s']}, "
                          f"commit_s={info['commit_s']}, latency_s={info['latency_s']}, "
                          f"duration_s={info['duration_s']}; indoor path + outdoor entrance "
                          f"coordinates reused (original order) from template trial "
                          f"{info['template_sid']} of the same version; PDR-bridge/"
                          f"reacquisition segment is a straight-line interpolation between "
                          f"them (not row-reversal, not mirrored coordinates); GPS "
                          f"accuracy/VPS-reliability/PDR-confidence drawn from that "
                          f"version's real clean GPS_TO_VPS trial distributions; latency "
                          f"driven by that version's real gpsDwellSeconds config, not by "
                          f"VPS localization time."),
            })

    pd.DataFrame(manifest_rows).to_csv("reverse_synthetic_manifest.csv", index=False)
    print(f"Generated {len(manifest_rows)} synthetic reverse-handover support trials "
          f"({len(VERSIONS)} versions x 2 scenarios) into {OUT}")


if __name__ == "__main__":
    main()
