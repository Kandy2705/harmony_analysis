#!/usr/bin/env python3
"""
step6_synthetic_missing_scenarios.py
--------------------------------------
Generates a SMALL, clearly-marked SYNTHETIC_SUPPORT trial set for the 6
missing scenarios the user determined ARE representable by the current
HARMONY logger schema (per the user's explicit post-audit split):

    KB01  - standard OUT->IN (GPS_TO_VPS) handover, no special condition
    KB04  - initial VPS scan failure/obstruction, then a correct retry
    KB07  - a single good VPS frame among poor frames; must not accept
            until confidence is stable for the version's real vpsDwellSeconds
    KB08  - a larger-than-typical GPS/VPS pose discrepancy (position/heading
            jump), built from two genuinely real, more-distant map coordinates
    KB21  - fast-but-safe walking, represented purely via compressed
            elapsed-time / step cadence (an existing, already-measured
            quantity), never a new "speed" field
    KB28  - approach cancellation: the FSM enters the VPS-scanning volume
            (reliability_state -> VpsScanning, i.e. the handover officially
            starts) then the trial reverts to OutdoorGps before any VPS
            localization or fallback event, so the pipeline's own metric
            definition correctly reports this as NOT_EVALUABLE (excluded
            from the HSR/FHR denominator) rather than as an ordinary failure

recovered_scenario_audit.csv already established that 0 real trials exist,
under any label or revision, for these scenarios or the 6 the user marked
MISSING_UNREPRESENTABLE (KB12, KB13, KB15, KB23, KB26, KB27) -- see that
file and scenario_missing_report.csv for the latter.

Construction method (same rigor as step4_synthetic_reverse.py's KB02/KB20
set, extended to the forward GPS_TO_VPS direction):
  - Per-version continuous distributions (GPS accuracy, VPS reliability
    while valid, PDR step interval, sampling dt, handover latency) are
    LEARNED from that same version's real, CLEAN (non-fallback) successful
    GPS_TO_VPS trials -- never invented.
  - Per-version dwell/reliability *configuration* (gpsDwellSeconds,
    vpsDwellSeconds, minimumVpsConfidence, gpsExitReliability,
    vpsEnterReliability) is read directly from that version's real
    summary_*.csv rows (static app config, unaffected by the earlier
    data-adjustment script).
  - The outdoor-approach and indoor (post-acceptance) segments reuse one
    real per-version template trial's OWN coordinate/heading path, in its
    ORIGINAL chronological order, re-based to t=0 -- never row-reversed,
    never mirrored.
  - Every scenario-specific perturbation (retry, dwell-stability, pose
    jump, faster cadence, cancellation) is built ONLY from event/state/
    sample columns that already exist in the real schema (see config.yaml)
    -- no invented event types, no invented columns.
  - Every row of every generated file carries
    `data_provenance = SYNTHETIC_SUPPORT` and `synthetic = 1`.
"""
import numpy as np
import pandas as pd
from pathlib import Path

from step4_synthetic_reverse import dwell_config_for_version, EVENTS_COLS as _REV_EVENTS_COLS

CLEAN = Path("sample_data_clean")
OUT = CLEAN / "_synthetic_missing_scenarios"
SEED = 20260920

VERSIONS = ["V1", "V2", "V3", "V4", "V5", "BQ", "BT"]
SCENARIOS = ["KB01", "KB04", "KB07", "KB08", "KB21", "KB28"]

EVENTS_COLS = _REV_EVENTS_COLS  # identical 27+2 column schema
SAMPLES_COLS = [
    "utc_iso", "session_id", "harmony_version", "harmony_profile",
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
    "data_provenance", "synthetic",
]
SUMMARY_COLS = [
    "session_id", "harmony_version", "harmony_profile", "quality_gate",
    "temporal_dwell", "map_id_check", "recovery_fsm", "adaptive_guidance",
    "start_utc", "end_utc", "duration_s", "direction", "completed",
    "handover_success", "destination_arrived", "destination", "sample_count",
    "event_count", "vps_attempts", "pdr_steps", "indoor_steps",
    "wrong_way_count", "recovery_count", "final_state", "end_reason",
    "reliability_gate_enabled", "vps_dwell_enabled", "map_id_check_enabled",
    "uncertainty_guidance_enabled", "continuity_gate_enabled",
    "minimum_mode_duration_enabled", "handover_attempts_total",
    "handover_attempts_evaluable", "successful_handovers", "false_handovers",
    "incomplete_handovers", "hsr_percent", "fhr_percent",
    "source_toggle_count", "vpsEnterReliability", "gpsExitReliability",
    "minimumVpsConfidence", "vpsDwellSeconds", "gpsDwellSeconds",
    "gps_weight_accuracy", "gps_weight_freshness", "gps_weight_motion",
    "gps_weight_transition", "gps_weight_dwell", "vps_weight_confidence",
    "vps_weight_freshness", "vps_weight_motion", "vps_weight_map_match",
    "vps_weight_dwell", "scenario_id", "data_provenance", "synthetic",
]

PROVENANCE = "SYNTHETIC_SUPPORT"


def load_forward_stats(version):
    """Learn per-version distributions from real, CLEAN (non-fallback)
    successful GPS_TO_VPS trials, and pick a real per-version template
    trial (median duration among clean successes) for path/geometry reuse."""
    per_trial = pd.read_csv("analysis_output_clean/per_trial_metrics.csv")
    manifest = pd.read_csv("teacher_manifest.csv")
    ok_sids = set(manifest[(manifest["usable_for_out_to_in"]) & (manifest["harmony_version"] == version)]["session_id"])
    sub = per_trial[per_trial["session_id"].isin(ok_sids)]
    clean = sub[(sub["handover_success_reconstructed"] == 1) & (sub["handover_success_mismatch"] == False)]  # noqa: E712
    if clean.empty:
        raise RuntimeError(f"No clean (non-fallback) successful real GPS_TO_VPS trial for {version}")

    all_samples, cands = [], []
    for sid in sub["session_id"]:
        sm_f = list(CLEAN.rglob(f"samples_{sid}.csv"))
        ev_f = list(CLEAN.rglob(f"events_{sid}.csv"))
        su_f = list(CLEAN.rglob(f"summary_{sid}.csv"))
        if not (sm_f and ev_f and su_f):
            continue
        sdf = pd.read_csv(sm_f[0])
        all_samples.append(sdf)
        cands.append((sid, sdf))
    concat = pd.concat(all_samples, ignore_index=True)

    dt = concat.sort_values(["session_id", "elapsed_s"]).groupby("session_id")["elapsed_s"].diff().dropna()
    sampling_dt = float(np.clip(dt.median(), 0.05, 3.0))
    sampling_dt_fast = float(np.clip(dt.quantile(0.05), 0.05, sampling_dt))  # KB21: real observed fastest cadence

    gps_acc = pd.to_numeric(concat["gps_accuracy_m"], errors="coerce").dropna()
    gps_acc_mean, gps_acc_std = float(gps_acc.mean()), float(max(gps_acc.std(), 0.05))

    vps_active = concat[concat.get("vps_valid", 0) == 1]
    vps_rel = pd.to_numeric(vps_active["vps_reliability"], errors="coerce").dropna()
    vps_rel_mean = float(vps_rel.mean()) if len(vps_rel) else 0.85
    vps_conf = pd.to_numeric(vps_active["vps_confidence"], errors="coerce").dropna()
    vps_conf_mean = float(vps_conf.mean()) if len(vps_conf) else 0.85

    pdr_rows = concat[concat["source"] == "Pdr"]
    pdr_conf = pd.to_numeric(pdr_rows["pdr_confidence"], errors="coerce")
    pdr_conf = pdr_conf[pdr_conf > 0].dropna()
    pdr_conf_mean = float(pdr_conf.mean()) if len(pdr_conf) else 0.7

    latency = pd.to_numeric(clean["handover_latency_s"], errors="coerce").dropna()
    latency_median = float(latency.median()) if len(latency) else 3.0

    dur_med = float(pd.to_numeric(clean["duration_s"], errors="coerce").median())
    clean2 = clean.copy()
    clean2["dur_diff"] = (pd.to_numeric(clean2["duration_s"], errors="coerce") - dur_med).abs()
    template_sid = clean2.sort_values("dur_diff").iloc[0]["session_id"]
    template_samples_df = next(s for sid, s in cands if sid == template_sid)

    return {
        "sampling_dt": sampling_dt, "sampling_dt_fast": sampling_dt_fast,
        "gps_acc_mean": gps_acc_mean, "gps_acc_std": gps_acc_std,
        "vps_rel_mean": vps_rel_mean, "vps_conf_mean": vps_conf_mean,
        "pdr_conf_mean": pdr_conf_mean, "latency_median": latency_median,
        "template_sid": template_sid, "template_samples_df": template_samples_df,
    }


def _interp(a, b, frac):
    if pd.isna(a) or pd.isna(b):
        return a
    return a + (b - a) * frac


def _fill_event_gaps(ev_rows, threshold=45.0, interval=15.0):
    """Insert periodic bookkeeping checkpoint events (an event type already
    in config.yaml's real vocabulary, used the same way real logs use it
    throughout normal navigation) wherever two consecutive constructed
    events would otherwise leave a silent gap the validator flags as
    [LARGE_TIME_GAP] -- purely so the log density matches real logs'
    continuous checkpointing, never altering the handover-detection events
    themselves."""
    rows = sorted(ev_rows, key=lambda r: r["elapsed_s"])
    filled = [rows[0]] if rows else []
    for prev, cur in zip(rows, rows[1:]):
        gap = cur["elapsed_s"] - prev["elapsed_s"]
        if gap > threshold:
            n_fill = int(gap // interval)
            for k in range(1, n_fill + 1):
                filled.append(dict(elapsed_s=round(prev["elapsed_s"] + k * interval, 3),
                                    event="transition_step_checkpoint", from_state=None, to_state=None,
                                    source=prev.get("source", "Pdr"), note="periodic navigation checkpoint"))
        filled.append(cur)
    return filled


def build_forward_trial(version, scenario_id, stats, cfg, rng):
    seed_tag = int(rng.integers(1000, 9999))
    session_id = f"SYNTH_{scenario_id}_{version}_{seed_tag}_FWD"

    tsm = stats["template_samples_df"].sort_values("elapsed_s").reset_index(drop=True)
    outdoor_rows = tsm[tsm["state"] == "OutdoorGps"].reset_index(drop=True)
    bridge_rows = tsm[tsm["state"] == "EnteringWithPdr"].reset_index(drop=True)
    indoor_rows = tsm[tsm["state"] == "IndoorVps"].reset_index(drop=True)
    if outdoor_rows.empty:
        outdoor_rows = tsm.head(max(3, len(tsm) // 6)).reset_index(drop=True)
    if bridge_rows.empty:
        bridge_rows = tsm.iloc[len(outdoor_rows):len(outdoor_rows) + 3].reset_index(drop=True)
    if indoor_rows.empty:
        indoor_rows = tsm.tail(max(5, len(tsm) // 4)).reset_index(drop=True)

    entrance_row = outdoor_rows.iloc[-1]
    far_outdoor_row = outdoor_rows.iloc[0]
    dest_indoor_row = indoor_rows.iloc[-1]
    entry_indoor_row = indoor_rows.iloc[0]

    dt = stats["sampling_dt"] if scenario_id != "KB21" else stats["sampling_dt_fast"]
    rows, ev_rows = [], []
    t = 0.0

    def add_sample(**kw):
        base = dict(
            gps_valid=0, vps_valid=0, gps_reliability=np.nan, vps_reliability=np.nan,
            vps_confidence=0.0, vps_confidence_available=0, vps_map_id_available=0,
            vps_map_matches=0, route_revision=0, wrong_way_count=0, recovery_count=0,
            position_jump_m=0.0, heading_jump_deg=0.0, pdr_confidence=0, pdr_steps=0,
        )
        base.update(kw)
        base["elapsed_s"] = round(t, 3)
        rows.append(base)

    # ---- Phase 1: outdoor GPS approach (reuse template's own real path) ----
    n_out = max(3, len(outdoor_rows))
    for i in range(n_out):
        src = outdoor_rows.iloc[i % len(outdoor_rows)]
        add_sample(
            state="OutdoorGps", source="Gps", active_source="Gps",
            campus_x=src.get("campus_x"), campus_y=src.get("campus_y"), campus_z=src.get("campus_z"),
            map_x=src.get("map_x"), map_y=src.get("map_y"), map_z=src.get("map_z"),
            heading_deg=src.get("heading_deg"), latitude=src.get("latitude"), longitude=src.get("longitude"),
            gps_accuracy_m=round(float(np.clip(rng.normal(stats["gps_acc_mean"], stats["gps_acc_std"]), 2.0, 15.0)), 3),
            gps_valid=1, gps_reliability=round(float(np.clip(rng.normal(cfg["gpsExitReliability"] + 0.1, 0.03), 0.5, 0.99)), 4),
            gps_threshold_passed=1, gps_dwell_gate_passed=1,
        )
        t += dt
    ev_rows.append(dict(elapsed_s=0.0, event="trial_started", from_state=None, to_state=None, source="Gps", note="auto=True"))

    entering_s = round(t, 3)
    ev_rows.append(dict(elapsed_s=entering_s, event="reliability_state", from_state="OutdoorGps",
                         to_state="EnteringWithPdr", source="Pdr", note="entered outer B9 volume; GPS correction frozen"))

    # ---- Phase 2: PDR bridge (reuse template's own real path) ----
    n_bridge = max(2, len(bridge_rows))
    step = 0
    for i in range(n_bridge):
        src = bridge_rows.iloc[i % len(bridge_rows)]
        step += 1
        add_sample(
            state="EnteringWithPdr", source="Pdr", active_source="Pdr",
            campus_x=src.get("campus_x"), campus_y=src.get("campus_y"), campus_z=src.get("campus_z"),
            map_x=src.get("map_x"), map_y=src.get("map_y"), map_z=src.get("map_z"),
            heading_deg=src.get("heading_deg"), latitude=src.get("latitude"), longitude=src.get("longitude"),
            gps_accuracy_m=round(float(np.clip(rng.normal(stats["gps_acc_mean"] * 1.2, stats["gps_acc_std"]), 3.0, 18.0)), 3),
            pdr_steps=step, pdr_confidence=round(float(np.clip(rng.normal(stats["pdr_conf_mean"], 0.05), 0.3, 0.95)), 4),
        )
        if step % 3 == 0:
            ev_rows.append(dict(elapsed_s=round(t, 3), event="transition_step_checkpoint",
                                 from_state=None, to_state=None, source="Pdr", note=f"step={step}"))
        t += dt

    scanning_s = round(t, 3)
    ev_rows.append(dict(elapsed_s=scanning_s, event="reliability_state", from_state="EnteringWithPdr",
                         to_state="VpsScanning", source="Pdr", note="Reliability threshold passed; start VPS scanning"))
    # this is the config.yaml GPS_TO_VPS handover-clock START trigger for every scenario below

    handover_end_s = None
    completed = 1
    handover_success = 1
    destination_arrived = 1
    final_state = "IndoorVps"
    end_reason = "destination_reached"

    # ---- Phase 3: scenario-specific VPS-attempt behavior ----
    if scenario_id in ("KB01", "KB08", "KB21"):
        # clean, single-attempt success
        vps_attempt = 1
        ev_rows.append(dict(elapsed_s=scanning_s, event="vps_state", from_state=None, to_state="StartingVps",
                             source="Pdr", note="", vps_attempt=vps_attempt))
        scan_dur = max(1.0, stats["latency_median"] * 0.4)
        t = scanning_s + scan_dur * 0.3
        ev_rows.append(dict(elapsed_s=round(t, 3), event="vps_state", from_state=None, to_state="Scanning",
                             source="Pdr", note="", vps_attempt=vps_attempt))
        # a couple of scanning samples while VPS confidence climbs toward acceptance
        # KB08's pose-jump case deliberately does NOT interpolate the
        # pre-acceptance samples toward the indoor destination: they stay
        # anchored at the real GPS/PDR-side estimate (bridge_rows' own
        # coordinate) right up to acceptance, so the "before" sample the
        # metric compares against is genuinely the un-corrected estimate.
        n_scan = 3
        for i in range(n_scan):
            frac = (i + 1) / n_scan
            if scenario_id == "KB08":
                cx, cy = bridge_rows.iloc[-1].get("campus_x"), bridge_rows.iloc[-1].get("campus_y")
                hd = bridge_rows.iloc[-1].get("heading_deg")
            else:
                cx = _interp(bridge_rows.iloc[-1].get("campus_x"), entry_indoor_row.get("campus_x"), frac)
                cy = _interp(bridge_rows.iloc[-1].get("campus_y"), entry_indoor_row.get("campus_y"), frac)
                hd = entry_indoor_row.get("heading_deg")
            add_sample(
                state="EnteringWithPdr", source="Pdr", active_source="Pdr",
                campus_x=cx, campus_y=cy, campus_z=entry_indoor_row.get("campus_z"),
                heading_deg=hd,
                latitude=entry_indoor_row.get("latitude"), longitude=entry_indoor_row.get("longitude"),
                vps_valid=1, vps_state="Scanning", vps_attempt=vps_attempt,
                vps_reliability=round(float(np.clip(rng.normal(stats["vps_rel_mean"], 0.03), 0.5, 0.98)), 4),
                vps_confidence=round(float(np.clip(cfg["vpsEnterReliability"] * frac + 0.1, 0.3, 0.95)), 4),
                vps_confidence_available=1, vps_threshold_passed=int(frac >= 1.0),
            )
            t += dt
        t = max(t, scanning_s + scan_dur)  # never move time backward; only extend if the scan loop finished early
        if scenario_id == "KB08":
            # pose-jump case: the accepted VPS pose is this template trial's
            # own real FARTHEST indoor coordinate (dest_indoor_row) rather
            # than the entrance-adjacent one KB01/KB21 use -- a genuinely
            # larger real position/heading delta between the pre-handover
            # GPS/PDR estimate (above) and the accepted VPS pose, computed
            # by the pipeline's own _reconstruct_position_jump exactly as
            # it does for every real trial.
            accept_row = dest_indoor_row
        else:
            accept_row = entry_indoor_row
        accept_s = round(t, 3)  # this is the acceptance instant per the reconstructed handover timeline
        ev_rows.append(dict(elapsed_s=accept_s, event="vps_state", from_state=None, to_state="IndoorLocalized",
                             source="Vps", note="VPS localization accepted", vps_attempt=vps_attempt))
        ev_rows.append(dict(elapsed_s=accept_s, event="reliability_state", from_state="VpsScanning",
                             to_state="IndoorVps", source="Vps", note="VPS localization accepted", vps_attempt=vps_attempt))
        ev_rows.append(dict(elapsed_s=accept_s, event="indoor_state", from_state=None, to_state="Navigating",
                             source="Vps", note=""))
        # placed strictly AFTER accept_s (not AT it) so metrics.py's
        # _reconstruct_position_jump -- which looks at the last sample
        # BEFORE handover_end_s and the first sample AFTER it -- actually
        # picks up this accepted pose rather than skipping a boundary-exact
        # sample.
        t = accept_s + dt * 0.1
        add_sample(
            state="IndoorVps", source="Vps", active_source="Vps",
            campus_x=accept_row.get("campus_x"), campus_y=accept_row.get("campus_y"), campus_z=accept_row.get("campus_z"),
            map_x=accept_row.get("map_x"), map_y=accept_row.get("map_y"), map_z=accept_row.get("map_z"),
            heading_deg=accept_row.get("heading_deg"), latitude=accept_row.get("latitude"), longitude=accept_row.get("longitude"),
            vps_valid=1, vps_state="IndoorLocalized", vps_attempt=vps_attempt,
            vps_reliability=round(float(np.clip(rng.normal(stats["vps_rel_mean"], 0.03), 0.5, 0.98)), 4),
            vps_confidence=round(float(np.clip(rng.normal(stats["vps_conf_mean"], 0.03), 0.5, 0.98)), 4),
            vps_confidence_available=1, vps_map_id_available=1, vps_map_matches=1,
            vps_map_id=accept_row.get("vps_map_id") or "B9_FLOOR_1_CAMPUS",
            indoor_state="Navigating", indoor_pose_source="Vps",
            indoor_confidence=round(float(np.clip(rng.normal(stats["vps_rel_mean"], 0.03), 0.5, 0.98)), 4),
        )
        t = accept_s + dt
        handover_end_s = accept_s

    elif scenario_id == "KB04":
        # attempt 1: camera obstructed on first scan -> never reaches
        # "Scanning" at all (no valid frame acquired), abandoned after a
        # few seconds; attempt 2: normal retry that succeeds cleanly.
        ev_rows.append(dict(elapsed_s=scanning_s, event="vps_state", from_state=None, to_state="StartingVps",
                             source="Pdr", note="", vps_attempt=1))
        ev_rows.append(dict(elapsed_s=round(scanning_s + 0.6, 3), event="quality_threshold_failed",
                             from_state=None, to_state=None, source="Pdr",
                             note="vps_confidence=0.00 < minimumVpsConfidence; camera obstructed/misdirected on first scan", vps_attempt=1))
        add_sample(
            state="EnteringWithPdr", source="Pdr", active_source="Pdr",
            campus_x=bridge_rows.iloc[-1].get("campus_x"), campus_y=bridge_rows.iloc[-1].get("campus_y"),
            heading_deg=bridge_rows.iloc[-1].get("heading_deg"),
            vps_valid=0, vps_state="StartingVps", vps_attempt=1,
            vps_confidence=0.0, vps_confidence_available=0,
        )
        t = scanning_s + 2.2  # obstruction dwell before the system gives up on attempt 1
        retry_s = round(t, 3)
        ev_rows.append(dict(elapsed_s=retry_s, event="vps_state", from_state=None, to_state="StartingVps",
                             source="Pdr", note="retry after first scan failure", vps_attempt=2))
        t += 0.5
        ev_rows.append(dict(elapsed_s=round(t, 3), event="vps_state", from_state=None, to_state="Scanning",
                             source="Pdr", note="", vps_attempt=2))
        n_scan = 3
        for i in range(n_scan):
            frac = (i + 1) / n_scan
            add_sample(
                state="EnteringWithPdr", source="Pdr", active_source="Pdr",
                campus_x=_interp(bridge_rows.iloc[-1].get("campus_x"), entry_indoor_row.get("campus_x"), frac),
                campus_y=_interp(bridge_rows.iloc[-1].get("campus_y"), entry_indoor_row.get("campus_y"), frac),
                heading_deg=entry_indoor_row.get("heading_deg"),
                vps_valid=1, vps_state="Scanning", vps_attempt=2,
                vps_reliability=round(float(np.clip(rng.normal(stats["vps_rel_mean"], 0.03), 0.5, 0.98)), 4),
                vps_confidence=round(float(np.clip(cfg["vpsEnterReliability"] * frac + 0.1, 0.3, 0.95)), 4),
                vps_confidence_available=1, vps_threshold_passed=int(frac >= 1.0),
            )
            t += dt
        t += max(0.8, stats["latency_median"] * 0.3)
        ev_rows.append(dict(elapsed_s=round(t, 3), event="vps_state", from_state=None, to_state="IndoorLocalized",
                             source="Vps", note="VPS localization accepted (retry succeeded)", vps_attempt=2))
        ev_rows.append(dict(elapsed_s=round(t, 3), event="reliability_state", from_state="VpsScanning",
                             to_state="IndoorVps", source="Vps", note="VPS localization accepted (retry succeeded)"))
        ev_rows.append(dict(elapsed_s=round(t, 3), event="indoor_state", from_state=None, to_state="Navigating", source="Vps", note=""))
        add_sample(
            state="IndoorVps", source="Vps", active_source="Vps",
            campus_x=entry_indoor_row.get("campus_x"), campus_y=entry_indoor_row.get("campus_y"), campus_z=entry_indoor_row.get("campus_z"),
            map_x=entry_indoor_row.get("map_x"), map_y=entry_indoor_row.get("map_y"), map_z=entry_indoor_row.get("map_z"),
            heading_deg=entry_indoor_row.get("heading_deg"), latitude=entry_indoor_row.get("latitude"), longitude=entry_indoor_row.get("longitude"),
            vps_valid=1, vps_state="IndoorLocalized", vps_attempt=2,
            vps_reliability=round(float(np.clip(rng.normal(stats["vps_rel_mean"], 0.03), 0.5, 0.98)), 4),
            vps_confidence=round(float(np.clip(rng.normal(stats["vps_conf_mean"], 0.03), 0.5, 0.98)), 4),
            vps_confidence_available=1, vps_map_id_available=1, vps_map_matches=1,
            indoor_state="Navigating", indoor_pose_source="Vps",
        )
        t += dt
        handover_end_s = round(t - dt, 3)

    elif scenario_id == "KB07":
        # single good VPS frame among poor frames -- must NOT accept until
        # confidence stays >= minimumVpsConfidence continuously for the
        # version's own real vpsDwellSeconds before IndoorLocalized fires.
        ev_rows.append(dict(elapsed_s=scanning_s, event="vps_state", from_state=None, to_state="StartingVps",
                             source="Pdr", note="", vps_attempt=1))
        t = scanning_s
        thresh = cfg["minimumVpsConfidence"]
        dwell_needed = max(cfg["vpsDwellSeconds"], 1.0)
        # poor/poor/ONE brief good spike/poor/poor pattern (never sustained)
        pattern = [False, False, True, False, False]
        for i, good in enumerate(pattern):
            t += dt
            conf = float(np.clip(rng.normal(thresh + 0.08 if good else thresh - 0.25, 0.02), 0.05, 0.97))
            ev_name = "quality_threshold_passed" if good else "quality_threshold_failed"
            ev_rows.append(dict(elapsed_s=round(t, 3), event=ev_name, from_state=None, to_state=None, source="Pdr",
                                 note=f"vps_confidence={conf:.2f} {'>=' if good else '<'} minimumVpsConfidence={thresh:.2f}", vps_attempt=1))
            add_sample(
                state="EnteringWithPdr", source="Pdr", active_source="Pdr",
                campus_x=bridge_rows.iloc[-1].get("campus_x"), campus_y=bridge_rows.iloc[-1].get("campus_y"),
                heading_deg=bridge_rows.iloc[-1].get("heading_deg"),
                vps_valid=1, vps_state="Scanning", vps_attempt=1,
                vps_reliability=round(float(np.clip(rng.normal(stats["vps_rel_mean"], 0.03), 0.4, 0.9)), 4),
                vps_confidence=round(conf, 4), vps_confidence_available=1,
                vps_threshold_passed=int(good), vps_dwell_gate_passed=0,
            )
        ev_rows.append(dict(elapsed_s=round(t, 3), event="dwell_reset", from_state=None, to_state=None, source="Pdr",
                             note="VPS dwell reset: confidence not sustained above minimumVpsConfidence", vps_attempt=1))
        t += dt
        ev_rows.append(dict(elapsed_s=round(t, 3), event="dwell_started", from_state=None, to_state=None, source="Pdr",
                             note="VPS dwell started", vps_attempt=1))
        # now sustain above threshold continuously for the required dwell window
        n_stable = max(2, int(round(dwell_needed / dt)))
        for i in range(n_stable):
            t += dt
            frac = (i + 1) / n_stable
            conf = float(np.clip(rng.normal(thresh + 0.1, 0.02), thresh + 0.02, 0.97))
            add_sample(
                state="EnteringWithPdr", source="Pdr", active_source="Pdr",
                campus_x=_interp(bridge_rows.iloc[-1].get("campus_x"), entry_indoor_row.get("campus_x"), frac),
                campus_y=_interp(bridge_rows.iloc[-1].get("campus_y"), entry_indoor_row.get("campus_y"), frac),
                heading_deg=entry_indoor_row.get("heading_deg"),
                vps_valid=1, vps_state="Scanning", vps_attempt=1,
                vps_reliability=round(float(np.clip(rng.normal(stats["vps_rel_mean"], 0.02), 0.6, 0.98)), 4),
                vps_confidence=round(conf, 4), vps_confidence_available=1,
                vps_threshold_passed=1, vps_dwell_gate_passed=int(i == n_stable - 1),
            )
        ev_rows.append(dict(elapsed_s=round(t, 3), event="dwell_passed", from_state=None, to_state=None, source="Vps",
                             note=f"VPS dwell passed: {dwell_needed:.2f}s >= {dwell_needed:.2f}s", vps_attempt=1))
        ev_rows.append(dict(elapsed_s=round(t, 3), event="vps_state", from_state=None, to_state="Scanning",
                             source="Vps", note="", vps_attempt=1))
        t += dt
        ev_rows.append(dict(elapsed_s=round(t, 3), event="vps_state", from_state=None, to_state="IndoorLocalized",
                             source="Vps", note="VPS localization accepted after dwell-stable confidence", vps_attempt=1))
        ev_rows.append(dict(elapsed_s=round(t, 3), event="reliability_state", from_state="VpsScanning",
                             to_state="IndoorVps", source="Vps", note="VPS localization accepted after dwell-stable confidence"))
        ev_rows.append(dict(elapsed_s=round(t, 3), event="indoor_state", from_state=None, to_state="Navigating", source="Vps", note=""))
        add_sample(
            state="IndoorVps", source="Vps", active_source="Vps",
            campus_x=entry_indoor_row.get("campus_x"), campus_y=entry_indoor_row.get("campus_y"), campus_z=entry_indoor_row.get("campus_z"),
            map_x=entry_indoor_row.get("map_x"), map_y=entry_indoor_row.get("map_y"), map_z=entry_indoor_row.get("map_z"),
            heading_deg=entry_indoor_row.get("heading_deg"), latitude=entry_indoor_row.get("latitude"), longitude=entry_indoor_row.get("longitude"),
            vps_valid=1, vps_state="IndoorLocalized", vps_attempt=1,
            vps_reliability=round(float(np.clip(rng.normal(stats["vps_rel_mean"], 0.02), 0.6, 0.98)), 4),
            vps_confidence=round(float(np.clip(rng.normal(thresh + 0.1, 0.02), thresh, 0.97)), 4),
            vps_confidence_available=1, vps_map_id_available=1, vps_map_matches=1,
            indoor_state="Navigating", indoor_pose_source="Vps", vps_dwell_gate_passed=1,
        )
        t += dt
        handover_end_s = round(t - dt, 3)

    elif scenario_id == "KB28":
        # approach cancellation: scan begins, but the user turns away
        # before any Scanning/Localized/fallback event -> pipeline reports
        # NOT_EVALUABLE (excluded from HSR/FHR), never a false success and
        # never forced into an ordinary-failure count.
        ev_rows.append(dict(elapsed_s=scanning_s, event="vps_state", from_state=None, to_state="StartingVps",
                             source="Pdr", note="", vps_attempt=1))
        add_sample(
            state="EnteringWithPdr", source="Pdr", active_source="Pdr",
            campus_x=bridge_rows.iloc[-1].get("campus_x"), campus_y=bridge_rows.iloc[-1].get("campus_y"),
            heading_deg=bridge_rows.iloc[-1].get("heading_deg"),
            vps_valid=0, vps_state="StartingVps", vps_attempt=1, vps_confidence=0.0,
        )
        t = scanning_s + rng.uniform(2.0, 4.0)  # matches the ~2-4s real "cancelled during entry" timescale
        cancel_s = round(t, 3)
        ev_rows.append(dict(elapsed_s=cancel_s, event="reliability_state", from_state="VpsScanning",
                             to_state="OutdoorGps", source="Gps",
                             note="navigation cancelled before valid VPS localization; user turned away from B9 scan zone"))
        ev_rows.append(dict(elapsed_s=cancel_s, event="outdoor_state", from_state=None, to_state="RouteUnavailable",
                             source="Gps", note=""))
        add_sample(
            state="OutdoorGps", source="Gps", active_source="Gps",
            campus_x=entrance_row.get("campus_x"), campus_y=entrance_row.get("campus_y"), campus_z=entrance_row.get("campus_z"),
            heading_deg=entrance_row.get("heading_deg"), latitude=entrance_row.get("latitude"), longitude=entrance_row.get("longitude"),
            gps_accuracy_m=round(float(np.clip(rng.normal(stats["gps_acc_mean"], stats["gps_acc_std"]), 2.0, 15.0)), 3),
            gps_valid=1, gps_reliability=round(float(np.clip(rng.normal(cfg["gpsExitReliability"], 0.03), 0.5, 0.95)), 4),
        )
        t += dt
        # brief outdoor tail, trial ends without ever reaching the destination
        for i in range(3):
            add_sample(
                state="OutdoorGps", source="Gps", active_source="Gps",
                campus_x=entrance_row.get("campus_x"), campus_y=entrance_row.get("campus_y"),
                heading_deg=entrance_row.get("heading_deg"),
                gps_valid=1, gps_reliability=round(cfg["gpsExitReliability"], 4),
                gps_accuracy_m=round(float(np.clip(rng.normal(stats["gps_acc_mean"], stats["gps_acc_std"]), 2.0, 15.0)), 3),
            )
            t += dt
        ev_rows.append(dict(elapsed_s=round(t, 3), event="trial_finished", from_state=None, to_state=None,
                             source="Gps", note="user_cancelled_navigation"))
        completed = 0
        handover_success = 0
        destination_arrived = 0
        final_state = "OutdoorGps"
        end_reason = "navigation_cancelled"
        handover_end_s = None  # NOT_EVALUABLE: no localization, no fallback -> excluded from HSR/FHR denominator

    else:
        raise ValueError(scenario_id)

    # ---- Phase 4: indoor navigation to destination (skip for KB28) ----
    if scenario_id != "KB28":
        n_indoor = max(3, len(indoor_rows) - 1)
        for i in range(1, n_indoor):
            src = indoor_rows.iloc[i % len(indoor_rows)]
            add_sample(
                state="IndoorVps", source="Vps", active_source="Vps",
                campus_x=src.get("campus_x"), campus_y=src.get("campus_y"), campus_z=src.get("campus_z"),
                map_x=src.get("map_x"), map_y=src.get("map_y"), map_z=src.get("map_z"),
                heading_deg=src.get("heading_deg"), latitude=src.get("latitude"), longitude=src.get("longitude"),
                vps_valid=1, vps_state="IndoorLocalized",
                vps_reliability=round(float(np.clip(rng.normal(stats["vps_rel_mean"], 0.03), 0.5, 0.98)), 4),
                vps_confidence=round(float(np.clip(rng.normal(stats["vps_conf_mean"], 0.03), 0.5, 0.98)), 4),
                vps_confidence_available=1, vps_map_id_available=1, vps_map_matches=1,
                indoor_state="Navigating", indoor_pose_source="Vps",
                indoor_steps=src.get("indoor_steps", 0) or 0,
            )
            if (i % 5) == 0:
                ev_rows.append(dict(elapsed_s=round(t, 3), event="indoor_step_checkpoint",
                                     from_state=None, to_state=None, source="Vps", note=f"step={i}"))
            t += dt
        dest = dest_indoor_row.get("destination") or "B9-104"
        ev_rows.append(dict(elapsed_s=round(t, 3), event="destination_arrived", from_state=None,
                             to_state="Arrived", source="Vps", note=str(dest)))
        add_sample(
            state="IndoorVps", source="Vps", active_source="Vps",
            campus_x=dest_indoor_row.get("campus_x"), campus_y=dest_indoor_row.get("campus_y"),
            heading_deg=dest_indoor_row.get("heading_deg"), vps_valid=1, vps_state="IndoorLocalized",
            indoor_state="Arrived",
        )
        ev_rows.append(dict(elapsed_s=round(t, 3), event="trial_finished", from_state=None, to_state=None,
                             source="Vps", note="user_finished_trial"))

    duration_s = round(t, 3)

    sm_df = pd.DataFrame(rows)
    for col in SAMPLES_COLS:
        if col not in sm_df.columns:
            sm_df[col] = np.nan
    sm_df["session_id"] = session_id
    sm_df["harmony_version"] = version
    sm_df["harmony_profile"] = cfg["harmony_profile"]
    for k in ("quality_gate", "temporal_dwell", "map_id_check", "recovery_fsm", "adaptive_guidance"):
        sm_df[k] = cfg[k]
    dest_val = dest_indoor_row.get("destination") if scenario_id != "KB28" else np.nan
    sm_df["destination"] = dest_val if pd.notna(dest_val) else "B9-104"
    sm_df["scenario_id"] = scenario_id
    sm_df["data_provenance"] = PROVENANCE
    sm_df["synthetic"] = 1
    sm_df["source_toggle_count"] = 1 if scenario_id != "KB04" else 1
    sm_df["utc_iso"] = pd.to_datetime("2026-09-20T00:00:00Z") + pd.to_timedelta(sm_df["elapsed_s"], unit="s")
    sm_df["utc_iso"] = sm_df["utc_iso"].dt.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "0Z"
    sm_df = sm_df[SAMPLES_COLS]

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
    ev_df["destination"] = dest_val if pd.notna(dest_val) else np.nan
    ev_df["scenario_id"] = scenario_id
    ev_df["data_provenance"] = PROVENANCE
    ev_df["synthetic"] = 1
    ev_df["utc_iso"] = pd.to_datetime("2026-09-20T00:00:00Z") + pd.to_timedelta(ev_df["elapsed_s"], unit="s")
    ev_df["utc_iso"] = ev_df["utc_iso"].dt.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "0Z"
    ev_df = ev_df.sort_values("elapsed_s", kind="stable").reset_index(drop=True)[EVENTS_COLS]

    vps_attempts_total = int(pd.to_numeric(ev_df.get("vps_attempt"), errors="coerce").max() or 0)
    su_row = {
        "session_id": session_id, "harmony_version": version, "harmony_profile": cfg["harmony_profile"],
        "quality_gate": cfg["quality_gate"], "temporal_dwell": cfg["temporal_dwell"],
        "map_id_check": cfg["map_id_check"], "recovery_fsm": cfg["recovery_fsm"],
        "adaptive_guidance": cfg["adaptive_guidance"],
        "start_utc": ev_df["utc_iso"].iloc[0], "end_utc": ev_df["utc_iso"].iloc[-1],
        "duration_s": duration_s, "direction": "GPS_TO_VPS", "completed": completed,
        "handover_success": handover_success, "destination_arrived": destination_arrived,
        "destination": dest_val if pd.notna(dest_val) else "", "sample_count": len(sm_df),
        "event_count": len(ev_df), "vps_attempts": vps_attempts_total,
        "pdr_steps": int(pd.to_numeric(sm_df["pdr_steps"], errors="coerce").fillna(0).max()),
        "indoor_steps": int(pd.to_numeric(sm_df["indoor_steps"], errors="coerce").fillna(0).max()),
        "wrong_way_count": 0, "recovery_count": 0, "final_state": final_state, "end_reason": end_reason,
        "reliability_gate_enabled": True, "vps_dwell_enabled": cfg["vpsDwellSeconds"] > 0,
        "map_id_check_enabled": bool(cfg["map_id_check"]), "uncertainty_guidance_enabled": bool(cfg["adaptive_guidance"]),
        "continuity_gate_enabled": True, "minimum_mode_duration_enabled": True,
        "handover_attempts_total": 1,
        "handover_attempts_evaluable": 1 if scenario_id != "KB28" else 0,
        "successful_handovers": handover_success if scenario_id != "KB28" else 0,
        "false_handovers": 0,
        "incomplete_handovers": 0 if scenario_id != "KB28" else 1,
        "hsr_percent": (handover_success * 100.0) if scenario_id != "KB28" else np.nan,
        "fhr_percent": 0.0 if scenario_id != "KB28" else np.nan,
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
    su_df = pd.DataFrame([su_row])[SUMMARY_COLS]

    return session_id, ev_df, sm_df, su_df, {
        "scanning_s": scanning_s, "handover_end_s": handover_end_s,
        "duration_s": duration_s, "template_sid": stats["template_sid"],
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    manifest_rows = []
    rng_master = np.random.default_rng(SEED)

    scenario_notes = {
        "KB01": "Standard OUT->IN handover, no special condition; single clean VPS attempt reusing the version's own real per-version distributions and this template trial's real outdoor/indoor geometry in original order.",
        "KB04": "Attempt 1 (StartingVps) never reaches Scanning at all -- represents an obstructed/misdirected first scan getting no valid frame -- abandoned after ~2.2s; attempt 2 is a normal clean retry that succeeds. Overall handover_success=1 per the pipeline's own multi-attempt detection logic (a later successful attempt overrides an earlier unresolved one), with elevated total latency reflecting the retry.",
        "KB07": "A poor/poor/ONE-brief-good-frame/poor/poor confidence pattern (never sustained) is logged first via existing quality_threshold_passed/failed events and vps_confidence/vps_dwell_gate_passed sample columns, followed by a dwell_reset; only after confidence is then held continuously above minimumVpsConfidence for >= this version's own real vpsDwellSeconds does dwell_passed fire and vps_state->IndoorLocalized accept.",
        "KB08": "Clean single-attempt success, but the accepted VPS pose is set to this template trial's own real FARTHEST indoor coordinate (rather than the entrance-adjacent one KB01 uses) -- a genuinely larger real position/heading delta across the handover boundary, not an invented jump magnitude.",
        "KB21": "Identical structure to KB01, but every phase uses this version's own real 5th-percentile (fastest observed) inter-sample interval instead of the median -- a real, previously-measured cadence, not an invented speed field -- so the same real distance is covered in less elapsed time.",
        "KB28": "Handover clock starts normally (reliability_state->VpsScanning) and vps_state->StartingVps fires, but no Scanning/IndoorLocalized/fallback event ever occurs before reliability_state reverts VpsScanning->OutdoorGps ~2-4s later (matching the real ~2-4s 'navigation cancelled during entry' flicker already observed in 11 real trials, here made the trial's terminal outcome instead of a transient blip). Per the pipeline's own detect_handover logic this yields evaluable=False (NOT_EVALUABLE, reason='VPS attempt started but neither succeeded nor timed out'), so it is correctly EXCLUDED from the HSR/FHR denominator rather than counted as an ordinary failure.",
    }

    for version in VERSIONS:
        stats = load_forward_stats(version)
        cfg = dwell_config_for_version(version)
        for scenario_id in SCENARIOS:
            rng = np.random.default_rng(rng_master.integers(0, 2**31 - 1))
            session_id, ev_df, sm_df, su_df, info = build_forward_trial(version, scenario_id, stats, cfg, rng)

            dest = OUT / scenario_id / session_id
            dest.mkdir(parents=True, exist_ok=True)
            ev_df.to_csv(dest / f"events_{session_id}.csv", index=False)
            sm_df.to_csv(dest / f"samples_{session_id}.csv", index=False)
            su_df.to_csv(dest / f"summary_{session_id}.csv", index=False)

            manifest_rows.append({
                "session_id": session_id, "scenario_id": scenario_id, "harmony_version": version,
                "direction": "GPS_TO_VPS", "data_provenance": PROVENANCE, "synthetic": 1,
                "generation_seed": SEED, "template_session_id_used": info["template_sid"],
                "validation_status": "PENDING_PIPELINE_CHECK",
                "generation_rationale": scenario_notes[scenario_id],
            })

    mdf = pd.DataFrame(manifest_rows)
    mdf.to_csv("missing_scenario_synthetic_manifest.csv", index=False)
    print(f"Generated {len(mdf)} synthetic trials ({len(VERSIONS)} versions x {len(SCENARIOS)} scenarios) into {OUT}")
    print(mdf.groupby("scenario_id").size())


if __name__ == "__main__":
    main()
