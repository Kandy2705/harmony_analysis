"""
metrics.py
----------
Computes the full per-trial KPI set (spec section 3), reconstructing the
"do-not-trust-blindly" metrics from events_*.csv / samples_*.csv and only
using summary_*.csv as a cross-check (spec section 2).

Every metric that cannot be computed with confidence from the current
logger is emitted as NaN with an explanatory string in the sibling
`<metric>_note` column — never invented, never silently zeroed.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from .events import TrialTimeline, build_timeline
from .pairing import TrialFiles
from .validation import ValidationResult, validate_trial

NOT_EVALUABLE = "NOT_EVALUABLE_FROM_CURRENT_LOG"

# Metrics the spec explicitly says NOT to trust blindly from summary.csv
CROSS_CHECK_FIELDS = [
    "handover_success", "false_handovers", "hsr_percent", "fhr_percent",
    "source_toggle_count", "position_jump_m", "heading_jump_deg",
]


def _safe_first(df: Optional[pd.DataFrame], col: str, default=np.nan):
    if df is None or col not in df.columns or len(df) == 0:
        return default
    val = df.iloc[0][col]
    return val if not pd.isna(val) else default


def _stats_block(series: pd.Series, prefix: str) -> Dict[str, Any]:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if len(s) == 0:
        return {
            f"{prefix}_n": 0, f"{prefix}_mean": np.nan, f"{prefix}_median": np.nan,
            f"{prefix}_sd": np.nan, f"{prefix}_min": np.nan, f"{prefix}_max": np.nan,
        }
    return {
        f"{prefix}_n": int(len(s)),
        f"{prefix}_mean": round(float(s.mean()), 4),
        f"{prefix}_median": round(float(s.median()), 4),
        f"{prefix}_sd": round(float(s.std(ddof=1)), 4) if len(s) > 1 else 0.0,
        f"{prefix}_min": round(float(s.min()), 4),
        f"{prefix}_max": round(float(s.max()), 4),
    }


def _infer_direction(ev_df: pd.DataFrame, sm_df: Optional[pd.DataFrame], cfg: dict) -> Optional[str]:
    """Direction should ideally come from summary metadata; if summary is
    missing, fall back to inspecting which handover start events are
    present in the events log."""
    if sm_df is not None and len(sm_df) > 0 and "direction" in sm_df.columns:
        d = sm_df.iloc[0]["direction"]
        if isinstance(d, str) and d.strip():
            return d.strip()
    # Fallback: look for tell-tale events
    scanning_state = cfg["states"]["vps_scanning_fsm_state"]
    outdoor_state = cfg["states"]["outdoor_target_state"]
    rel_ev = cfg["events"]["reliability_state"]
    has_gps_to_vps = ((ev_df["event"] == rel_ev) & (ev_df["to_state"] == scanning_state)).any()
    has_vps_to_gps = ((ev_df["event"] == rel_ev) & (ev_df["to_state"] == outdoor_state) &
                       (ev_df["elapsed_s"] > 0)).any()
    if has_gps_to_vps and not has_vps_to_gps:
        return "GPS_TO_VPS"
    if has_vps_to_gps and not has_gps_to_vps:
        return "VPS_TO_GPS"
    return None


def _reconstruct_position_jump(sm_df: pd.DataFrame, handover_end_s: Optional[float]) -> Dict[str, Any]:
    """Attempt to reconstruct a real position/heading jump around the
    acceptance moment using coordinate columns, per spec section 3.14.
    Returns NaN + NOT_EVALUABLE note if insufficient data.
    """
    result = {
        "position_jump_m_reconstructed": np.nan,
        "heading_jump_deg_reconstructed": np.nan,
        "position_jump_note": NOT_EVALUABLE,
    }
    if handover_end_s is None or sm_df is None or len(sm_df) == 0:
        return result

    coord_cols_options = [
        ("campus_x", "campus_y"), ("map_x", "map_y"),
    ]
    window = 2.0  # seconds before/after acceptance
    before = sm_df[(sm_df["elapsed_s"] >= handover_end_s - window) & (sm_df["elapsed_s"] < handover_end_s)]
    after = sm_df[(sm_df["elapsed_s"] > handover_end_s) & (sm_df["elapsed_s"] <= handover_end_s + window)]

    if before.empty or after.empty:
        result["position_jump_note"] = (
            f"{NOT_EVALUABLE}: no samples within +/-{window}s of acceptance (elapsed_s={handover_end_s})."
        )
        return result

    for xcol, ycol in coord_cols_options:
        if xcol in sm_df.columns and ycol in sm_df.columns:
            bx, by = before.iloc[-1][xcol], before.iloc[-1][ycol]
            ax, ay = after.iloc[0][xcol], after.iloc[0][ycol]
            if pd.notna(bx) and pd.notna(by) and pd.notna(ax) and pd.notna(ay):
                # A zero-valued coordinate frame (both readings exactly 0,0) is a
                # strong sign the field is an unpopulated stub, not real geometry.
                if (bx, by) == (0, 0) and (ax, ay) == (0, 0):
                    continue
                dist = math.hypot(ax - bx, ay - by)
                result["position_jump_m_reconstructed"] = round(dist, 4)
                result["position_jump_note"] = f"reconstructed from {xcol}/{ycol} at +/-{window}s window"
                break

    if "heading_deg" in sm_df.columns:
        bh = before.iloc[-1]["heading_deg"] if not before.empty else np.nan
        ah = after.iloc[0]["heading_deg"] if not after.empty else np.nan
        if pd.notna(bh) and pd.notna(ah):
            diff = abs(ah - bh) % 360
            diff = min(diff, 360 - diff)
            result["heading_jump_deg_reconstructed"] = round(float(diff), 3)

    if pd.isna(result["position_jump_m_reconstructed"]):
        result["position_jump_note"] = (
            f"{NOT_EVALUABLE}: coordinate columns present but unpopulated (stub/zero) "
            f"around acceptance moment (elapsed_s={handover_end_s})."
        )

    return result


def compute_trial_metrics(trial: TrialFiles, cfg: dict) -> Dict[str, Any]:
    """Compute the complete metrics row for a single trial."""
    validation = validate_trial(trial, cfg)
    row: Dict[str, Any] = {
        "session_id": trial.session_id,
        "validation_status": validation.status,
        "validation_notes": validation.notes_str(),
    }

    if validation.status == "INVALID":
        row["events_file"] = str(trial.events.path) if trial.events else None
        row["samples_file"] = str(trial.samples.path) if trial.samples else None
        row["summary_file"] = str(trial.summary.path) if trial.summary else None
        return row

    ev_df: pd.DataFrame = trial.events.df.copy()
    sm_df: pd.DataFrame = trial.samples.df.copy()
    su_df: Optional[pd.DataFrame] = trial.summary.df.copy() if (trial.summary and trial.summary.df is not None) else None

    ev_df = ev_df.sort_values("elapsed_s").reset_index(drop=True)
    sm_df = sm_df.sort_values("elapsed_s").reset_index(drop=True)

    # ---------------- A. Basic metadata ----------------
    row["harmony_version"] = _safe_first(ev_df, "harmony_version", _safe_first(su_df, "harmony_version"))
    row["harmony_profile"] = _safe_first(ev_df, "harmony_profile", _safe_first(su_df, "harmony_profile"))
    row["direction"] = _infer_direction(ev_df, su_df, cfg)
    row["destination"] = _safe_first(su_df, "destination", _safe_first(ev_df, "destination"))
    row["duration_s"] = float(_safe_first(su_df, "duration_s", ev_df["elapsed_s"].max()))
    row["start_utc"] = _safe_first(su_df, "start_utc", _safe_first(ev_df, "utc_iso"))
    row["end_utc"] = _safe_first(su_df, "end_utc")
    row["completed"] = _safe_first(su_df, "completed")
    row["end_reason"] = _safe_first(su_df, "end_reason")
    row["n_samples"] = int(len(sm_df))
    row["n_events"] = int(len(ev_df))
    for flag in ("quality_gate", "temporal_dwell", "map_id_check", "recovery_fsm", "adaptive_guidance"):
        if flag in ev_df.columns:
            row[flag] = _safe_first(ev_df, flag)

    trial_end_s = float(ev_df["elapsed_s"].max())

    # ---------------- B. Reconstructed timeline ----------------
    timeline: TrialTimeline = build_timeline(ev_df, cfg, row["direction"], trial_end_s)

    if not timeline.handover_attempts:
        row["handover_success_reconstructed"] = np.nan
        row["handover_latency_s"] = np.nan
        row["handover_note"] = f"{NOT_EVALUABLE}: direction unknown or unsupported ({row['direction']})."
        primary_handover = None
    else:
        primary_handover = timeline.handover_attempts[0]
        row["handover_success_reconstructed"] = (
            np.nan if primary_handover.success is None else int(primary_handover.success)
        )
        row["handover_latency_s"] = (
            primary_handover.latency_s if primary_handover.latency_s is not None else np.nan
        )
        row["handover_start_s"] = primary_handover.start_s
        row["handover_end_s"] = primary_handover.end_s
        row["handover_note"] = primary_handover.reason if not primary_handover.evaluable else ""

    # ---- Cross-check against summary.handover_success ----
    summary_success = _safe_first(su_df, "handover_success")
    row["handover_success_summary"] = summary_success
    if not pd.isna(row.get("handover_success_reconstructed", np.nan)) and not pd.isna(summary_success):
        mismatch = int(row["handover_success_reconstructed"]) != int(summary_success)
        row["handover_success_mismatch"] = bool(mismatch)
        if mismatch:
            row["validation_notes"] = (row["validation_notes"] + " | " if row["validation_notes"] else "") + (
                f"[METRIC_MISMATCH] reconstructed handover_success="
                f"{row['handover_success_reconstructed']} != summary handover_success={summary_success}; "
                f"reconstructed value kept as primary result."
            )
            if validation.status == "VALID":
                row["validation_status"] = "VALID_WITH_WARNINGS"
    else:
        row["handover_success_mismatch"] = np.nan

    # ---------------- Source switching ----------------
    switches = timeline.source_switches
    row["source_transitions_total"] = len(switches)
    pair_counts: Dict[str, int] = {}
    for sw in switches:
        key = f"{sw.from_source}->{sw.to_source}"
        pair_counts[key] = pair_counts.get(key, 0) + 1
    for key in ("Gps->Pdr", "Pdr->Gps", "Pdr->Vps", "Vps->Pdr", "Gps->Vps", "Vps->Gps"):
        row[f"switch_count_{key.replace('->', '_to_')}"] = pair_counts.get(key, 0)
    for key, cnt in pair_counts.items():
        col = f"switch_count_{key.replace('->', '_to_')}"
        if col not in row:
            row[col] = cnt

    row["false_handover_count"] = len(timeline.false_handovers)
    row["evaluable_switches"] = len(switches)
    row["false_handover_rate_percent"] = (
        round(100.0 * len(timeline.false_handovers) / len(switches), 2) if switches else np.nan
    )
    row["oscillation_episode_count"] = timeline.oscillation_episodes
    row["unnecessary_source_switches"] = timeline.oscillation_switch_count

    # Cross-check source_toggle_count
    summary_toggle = _safe_first(su_df, "source_toggle_count")
    row["source_toggle_count_summary"] = summary_toggle
    row["source_toggle_count_reconstructed"] = len(switches)
    if not pd.isna(summary_toggle):
        row["source_toggle_count_mismatch"] = bool(int(summary_toggle) != len(switches))
    else:
        row["source_toggle_count_mismatch"] = np.nan

    # ---------------- VPS attempts ----------------
    attempts = timeline.vps_attempts
    row["vps_attempts_total"] = len(attempts)
    row["vps_attempts_success"] = sum(1 for a in attempts if a.outcome == "success")
    row["vps_attempts_timeout"] = sum(1 for a in attempts if a.outcome == "timeout_fallback")
    row["vps_attempts_unresolved"] = sum(1 for a in attempts if a.outcome == "unresolved")
    scan_durations = [a.scan_duration_s for a in attempts if a.outcome == "success" and a.scan_duration_s is not None]
    row.update(_stats_block(pd.Series(scan_durations), "vps_localization_time_s"))
    if attempts:
        last_attempt = attempts[-1]
        row["vps_localization_time_last_s"] = last_attempt.scan_duration_s if last_attempt.outcome == "success" else np.nan
        row["vps_localization_outcome_last"] = last_attempt.outcome
    else:
        row["vps_localization_time_last_s"] = np.nan
        row["vps_localization_outcome_last"] = NOT_EVALUABLE

    # ---------------- Transition / PDR duration ----------------
    pdr_rows = sm_df[sm_df["state"] == cfg["states"]["entering_pdr_state"]] if "state" in sm_df.columns else pd.DataFrame()
    if not pdr_rows.empty:
        row["pdr_transition_duration_s"] = round(float(pdr_rows["elapsed_s"].max() - pdr_rows["elapsed_s"].min()), 3)
    else:
        row["pdr_transition_duration_s"] = np.nan

    # ---------------- Reliability at acceptance & jumps (only on SUCCESSFUL handovers) ----------------
    if primary_handover is not None and primary_handover.end_s is not None and bool(primary_handover.success):
        at_accept = sm_df[sm_df["elapsed_s"] <= primary_handover.end_s]
        if not at_accept.empty:
            last = at_accept.iloc[-1]
            for col, out in (
                ("gps_reliability", "gps_reliability_at_acceptance"),
                ("vps_reliability", "vps_reliability_at_acceptance"),
                ("gps_stable_s", "gps_stable_s_at_acceptance"),
                ("vps_stable_s", "vps_stable_s_at_acceptance"),
            ):
                row[out] = float(last[col]) if col in sm_df.columns and pd.notna(last.get(col)) else np.nan
        jump = _reconstruct_position_jump(sm_df, primary_handover.end_s)
        row.update(jump)
    else:
        for out in ("gps_reliability_at_acceptance", "vps_reliability_at_acceptance",
                    "gps_stable_s_at_acceptance", "vps_stable_s_at_acceptance",
                    "position_jump_m_reconstructed", "heading_jump_deg_reconstructed"):
            row[out] = np.nan
        row["position_jump_note"] = f"{NOT_EVALUABLE}: no evaluable handover acceptance moment."

    # Cross-check position/heading jump vs summary (if summary even has it)
    summary_pos_jump = _safe_first(sm_df, "position_jump_m")  # logger's own per-sample column, if any
    row["position_jump_m_logger_column"] = summary_pos_jump
    row["position_jump_m_logger_note"] = (
        "Logger's own position_jump_m/heading_jump_deg column is treated as UNRELIABLE "
        "(observed constant 0 in sample data / internal settle-loop delta) and is NOT used "
        "as the primary result; see position_jump_m_reconstructed instead."
    )

    # ---------------- GPS statistics ----------------
    if "gps_accuracy_m" in sm_df.columns:
        row.update(_stats_block(sm_df["gps_accuracy_m"], "gps_accuracy_m"))
    if "gps_reliability" in sm_df.columns:
        s = pd.to_numeric(sm_df["gps_reliability"], errors="coerce").dropna()
        row["gps_reliability_mean"] = round(float(s.mean()), 4) if len(s) else np.nan
        row["gps_reliability_median"] = round(float(s.median()), 4) if len(s) else np.nan
    if "gps_age_s" in sm_df.columns:
        s = pd.to_numeric(sm_df["gps_age_s"], errors="coerce").dropna()
        row["gps_age_s_mean"] = round(float(s.mean()), 4) if len(s) else np.nan
        row["gps_age_s_max"] = round(float(s.max()), 4) if len(s) else np.nan

    # ---------------- PDR statistics ----------------
    row["pdr_steps_total"] = int(_safe_first(su_df, "pdr_steps", sm_df["pdr_steps"].max() if "pdr_steps" in sm_df else np.nan)) \
        if not pd.isna(_safe_first(su_df, "pdr_steps", sm_df["pdr_steps"].max() if "pdr_steps" in sm_df else np.nan)) else np.nan
    if "pdr_confidence" in sm_df.columns:
        s = pd.to_numeric(sm_df["pdr_confidence"], errors="coerce")
        s = s[s > 0].dropna()
        row["pdr_confidence_mean"] = round(float(s.mean()), 4) if len(s) else np.nan
        row["pdr_confidence_median"] = round(float(s.median()), 4) if len(s) else np.nan

    # ---------------- VPS statistics ----------------
    if "vps_reliability" in sm_df.columns:
        s = pd.to_numeric(sm_df["vps_reliability"], errors="coerce")
        vps_active = sm_df[sm_df.get("vps_valid", 0) == 1] if "vps_valid" in sm_df.columns else sm_df.iloc[0:0]
        s_active = pd.to_numeric(vps_active["vps_reliability"], errors="coerce").dropna() if len(vps_active) else pd.Series(dtype=float)
        row["vps_reliability_mean_when_valid"] = round(float(s_active.mean()), 4) if len(s_active) else np.nan
    if "vps_confidence_available" in sm_df.columns:
        avail = pd.to_numeric(sm_df["vps_confidence_available"], errors="coerce").fillna(0)
        row["vps_confidence_available_rate_percent"] = round(100.0 * avail.mean(), 2) if len(avail) else np.nan
    if "vps_map_id_available" in sm_df.columns:
        mapavail = pd.to_numeric(sm_df["vps_map_id_available"], errors="coerce").fillna(0)
        row["vps_map_id_available_rate_percent"] = round(100.0 * mapavail.mean(), 2) if len(mapavail) else np.nan
    if "vps_map_matches" in sm_df.columns:
        matches = pd.to_numeric(sm_df["vps_map_matches"], errors="coerce").fillna(0)
        row["vps_map_match_rate_percent"] = round(100.0 * (matches > 0).mean(), 2) if len(matches) else np.nan

    # ---------------- HSR/FHR cross-check fields (trial-level components) ----------------
    row["handover_attempts_total_summary"] = _safe_first(su_df, "handover_attempts_total")
    row["handover_attempts_evaluable_summary"] = _safe_first(su_df, "handover_attempts_evaluable")
    row["successful_handovers_summary"] = _safe_first(su_df, "successful_handovers")
    row["false_handovers_summary"] = _safe_first(su_df, "false_handovers")
    row["hsr_percent_summary"] = _safe_first(su_df, "hsr_percent")
    row["fhr_percent_summary"] = _safe_first(su_df, "fhr_percent")

    return row
