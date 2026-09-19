"""
events.py
---------
Reconstructs the semantic timeline of a trial (source switches, VPS
attempts, handover windows, oscillation episodes) directly from the raw
`events_*.csv` rows.

This is the single place where the *meaning* of the logger's event names
is encoded, driven entirely by `config.yaml` — no event/state name is
hard-coded outside of this module's use of the config dict.

IMPORTANT — column semantics quirk of the real logger (see README):
`from_state` / `to_state` / `source` encode DIFFERENT things depending on
the value of `event`:
  * event == outdoor_state / indoor_state  -> to_state is the FSM state
  * event == source_switched                -> from_state/to_state are
                                                the OLD/NEW active source
  * event == reliability_state              -> from_state/to_state are
                                                the OLD/NEW "zone" state
  * event == vps_state                      -> to_state is the VPS
                                                sub-state (StartingVps /
                                                Scanning / IndoorLocalized)
This module is aware of that and dispatches accordingly.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


# ------------------------------------------------------------------
# Data classes describing reconstructed timeline objects
# ------------------------------------------------------------------

@dataclass
class SourceSwitch:
    elapsed_s: float
    from_source: str
    to_source: str


@dataclass
class VpsAttempt:
    attempt_index: int
    start_s: float                 # vps_state -> StartingVps
    scanning_s: Optional[float]    # vps_state -> Scanning
    end_s: Optional[float]         # localized OR fallback OR trial end
    outcome: str                   # "success" | "timeout_fallback" | "unresolved"
    scan_duration_s: Optional[float]   # end_s - (scanning_s or start_s)


@dataclass
class HandoverAttempt:
    direction: str                  # "GPS_TO_VPS" | "VPS_TO_GPS"
    start_s: Optional[float]
    end_s: Optional[float]
    success: Optional[bool]         # None = NOT_EVALUABLE
    latency_s: Optional[float]      # None if not evaluable
    evaluable: bool
    reason: str = ""                # explanation when success/latency is None


@dataclass
class TrialTimeline:
    source_switches: List[SourceSwitch] = field(default_factory=list)
    vps_attempts: List[VpsAttempt] = field(default_factory=list)
    handover_attempts: List[HandoverAttempt] = field(default_factory=list)
    false_handovers: List[SourceSwitch] = field(default_factory=list)
    oscillation_episodes: int = 0
    oscillation_switch_count: int = 0


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _note_has_fallback_pattern(note: object, patterns: List[str]) -> bool:
    if not isinstance(note, str):
        return False
    low = note.lower()
    return any(p.lower() in low for p in patterns)


def reconstruct_source_switches(events_df: pd.DataFrame, cfg: dict) -> List[SourceSwitch]:
    """Reconstruct every source_switched event as a (from, to) pair."""
    ev_name = cfg["events"]["source_switched"]
    rows = events_df[events_df["event"] == ev_name]
    switches: List[SourceSwitch] = []
    for _, row in rows.iterrows():
        frm = row.get("from_state")
        to = row.get("to_state")
        if pd.isna(frm) or pd.isna(to):
            continue
        switches.append(SourceSwitch(elapsed_s=float(row["elapsed_s"]), from_source=str(frm), to_source=str(to)))
    switches.sort(key=lambda s: s.elapsed_s)
    return switches


def reconstruct_vps_attempts(events_df: pd.DataFrame, cfg: dict, trial_end_s: float) -> List[VpsAttempt]:
    """Reconstruct each VPS localization attempt from vps_state transitions
    plus fallback events, WITHOUT trusting `vps_attempt` counter columns
    for the outcome (only used as an index hint)."""
    vps_ev = cfg["events"]["vps_state"]
    starting_state = cfg["states"]["vps_starting_state"]
    scanning_state = cfg["states"]["vps_scanning_state"]
    localized_state = cfg["states"]["vps_localized_state"]
    fallback_events = set(cfg["events"]["vps_fallback_events"])
    fallback_note_patterns = cfg.get("fallback_note_patterns", [])

    vps_rows = events_df[events_df["event"] == vps_ev].sort_values("elapsed_s")
    fallback_rows = events_df[events_df["event"].isin(fallback_events)].sort_values("elapsed_s")

    attempts: List[VpsAttempt] = []
    idx = 0
    pending_start: Optional[float] = None
    pending_scanning: Optional[float] = None

    starts = vps_rows[vps_rows["to_state"] == starting_state]
    localized = vps_rows[vps_rows["to_state"] == localized_state]
    scanning = vps_rows[vps_rows["to_state"] == scanning_state]

    for _, start_row in starts.iterrows():
        idx += 1
        start_s = float(start_row["elapsed_s"])

        # scanning_s: first Scanning event after this start, before the next start
        later_starts = starts[starts["elapsed_s"] > start_s]["elapsed_s"]
        next_start_s = float(later_starts.min()) if not later_starts.empty else (trial_end_s + 0.001)
        scan_candidates = scanning[(scanning["elapsed_s"] >= start_s) & (scanning["elapsed_s"] < next_start_s)]
        scanning_s = float(scan_candidates["elapsed_s"].min()) if not scan_candidates.empty else None

        # localized event within this attempt's window
        loc_candidates = localized[(localized["elapsed_s"] >= start_s) & (localized["elapsed_s"] < next_start_s)]
        end_s: Optional[float] = None
        outcome = "unresolved"

        if not loc_candidates.empty:
            end_s = float(loc_candidates["elapsed_s"].min())
            # Was this "localized" a genuine success, or a fallback-driven one?
            # Check for a fallback event at (approximately) the same time.
            near_fallback = fallback_rows[
                (fallback_rows["elapsed_s"] >= start_s) & (fallback_rows["elapsed_s"] <= end_s + 0.5)
            ]
            note_at_end = events_df[
                (events_df["elapsed_s"].between(end_s - 0.01, end_s + 0.01))
            ]["note"].dropna().tolist() if "note" in events_df.columns else []
            fallback_by_note = any(_note_has_fallback_pattern(n, fallback_note_patterns) for n in note_at_end)

            if not near_fallback.empty or fallback_by_note:
                outcome = "timeout_fallback"
            else:
                outcome = "success"
        else:
            # No localized state reached in this window at all.
            fb_in_window = fallback_rows[(fallback_rows["elapsed_s"] >= start_s) & (fallback_rows["elapsed_s"] < next_start_s)]
            if not fb_in_window.empty:
                end_s = float(fb_in_window["elapsed_s"].min())
                outcome = "timeout_fallback"
            else:
                outcome = "unresolved"  # attempt never resolved before trial end / next attempt

        scan_duration_s = None
        if end_s is not None:
            base = scanning_s if scanning_s is not None else start_s
            scan_duration_s = round(end_s - base, 3)

        attempts.append(VpsAttempt(
            attempt_index=idx,
            start_s=start_s,
            scanning_s=scanning_s,
            end_s=end_s,
            outcome=outcome,
            scan_duration_s=scan_duration_s,
        ))

    return attempts


def detect_handover(events_df: pd.DataFrame, cfg: dict, direction: str,
                     vps_attempts: List[VpsAttempt]) -> HandoverAttempt:
    """Detect the single (spec assumes one primary handover per direction
    trial) handover attempt for the given direction and evaluate its
    success/latency using RECONSTRUCTED state, never `final_state` alone.
    """
    dcfg = cfg["directions"].get(direction)
    if dcfg is None:
        return HandoverAttempt(direction=direction, start_s=None, end_s=None,
                                success=None, latency_s=None, evaluable=False,
                                reason=f"No config entry for direction={direction}")

    if direction == "GPS_TO_VPS":
        # Start: reliability_state -> VpsScanning (entering true handover volume)
        start_ev = cfg["events"]["reliability_state"]
        start_to = cfg["states"]["vps_scanning_fsm_state"]
        starts = events_df[(events_df["event"] == start_ev) & (events_df["to_state"] == start_to)]
        if starts.empty:
            return HandoverAttempt(direction=direction, start_s=None, end_s=None,
                                    success=None, latency_s=None, evaluable=False,
                                    reason="No reliability_state->VpsScanning event found; "
                                           "handover was never attempted or logger did not "
                                           "capture the start trigger.")
        start_s = float(starts["elapsed_s"].min())

        if not vps_attempts:
            return HandoverAttempt(direction=direction, start_s=start_s, end_s=None,
                                    success=False, latency_s=None, evaluable=True,
                                    reason="Handover volume entered but no VPS attempt "
                                           "(vps_state) was ever logged.")

        # Use the VPS attempt(s) that occur at/after start_s
        relevant = [a for a in vps_attempts if a.start_s >= start_s - 0.5]
        if not relevant:
            relevant = vps_attempts

        success_attempt = next((a for a in relevant if a.outcome == "success"), None)
        if success_attempt is not None:
            latency = round(success_attempt.end_s - start_s, 3) if success_attempt.end_s is not None else None
            return HandoverAttempt(direction=direction, start_s=start_s, end_s=success_attempt.end_s,
                                    success=True, latency_s=latency, evaluable=True)

        # No success attempt: fail if we saw a timeout/fallback, else NOT_EVALUABLE (still running)
        fallback_attempt = next((a for a in relevant if a.outcome == "timeout_fallback"), None)
        if fallback_attempt is not None:
            return HandoverAttempt(direction=direction, start_s=start_s, end_s=fallback_attempt.end_s,
                                    success=False, latency_s=None, evaluable=True,
                                    reason="VPS timed out and fell back to PDR-approximate pose "
                                           "(handover_completed_approximate); NOT counted as VPS success "
                                           "even though FSM reached IndoorVps.")

        return HandoverAttempt(direction=direction, start_s=start_s, end_s=None,
                                success=None, latency_s=None, evaluable=False,
                                reason="VPS attempt started but neither succeeded nor timed out "
                                       "before the trial log ends (trial likely truncated/aborted).")

    if direction == "VPS_TO_GPS":
        # Start: reliability_state -> OutdoorGps (leaving indoor volume) OR
        #        source_switched Vps->Gps as a fallback start marker.
        start_ev = cfg["events"]["reliability_state"]
        start_to = cfg["states"]["outdoor_target_state"]
        starts = events_df[(events_df["event"] == start_ev) & (events_df["to_state"] == start_to)]
        if starts.empty:
            return HandoverAttempt(direction=direction, start_s=None, end_s=None,
                                    success=None, latency_s=None, evaluable=False,
                                    reason="No reliability_state->OutdoorGps event found.")
        start_s = float(starts["elapsed_s"].min())

        sw_ev = cfg["events"]["source_switched"]
        target = cfg["directions"]["VPS_TO_GPS"]["target_source"].capitalize()
        # source_switched rows where to_state == "Gps" at/after start_s
        gps_accepts = events_df[
            (events_df["event"] == sw_ev) &
            (events_df["to_state"].astype(str).str.lower() == "gps") &
            (events_df["elapsed_s"] >= start_s)
        ]
        if gps_accepts.empty:
            return HandoverAttempt(direction=direction, start_s=start_s, end_s=None,
                                    success=None, latency_s=None, evaluable=False,
                                    reason="Left indoor volume but no source_switched->Gps "
                                           "event found before trial log ends.")
        end_s = float(gps_accepts["elapsed_s"].min())
        latency = round(end_s - start_s, 3)
        return HandoverAttempt(direction=direction, start_s=start_s, end_s=end_s,
                                success=True, latency_s=latency, evaluable=True)

    return HandoverAttempt(direction=direction, start_s=None, end_s=None,
                            success=None, latency_s=None, evaluable=False,
                            reason=f"Unsupported direction: {direction}")


def detect_false_handovers(switches: List[SourceSwitch], cfg: dict) -> List[SourceSwitch]:
    """A source switch A->B counted as FALSE/PREMATURE if it is followed by
    a B->A revert within `revert_window_s` seconds (the source flips right
    back), per spec section 3.3 — an explicit, documented rule rather than
    an invented one.
    """
    window = float(cfg["oscillation"]["revert_window_s"])
    false_ones: List[SourceSwitch] = []
    for i, sw in enumerate(switches):
        for later in switches[i + 1:]:
            if later.elapsed_s - sw.elapsed_s > window:
                break
            if later.from_source == sw.to_source and later.to_source == sw.from_source:
                false_ones.append(sw)
                break
    return false_ones


def detect_oscillation(switches: List[SourceSwitch], cfg: dict) -> tuple[int, int]:
    """Count oscillation episodes: a rolling window with >= min_switches
    source_switched events. Returns (episode_count, switches_involved).
    Episodes are counted disjointly (non-overlapping) to avoid inflating
    the count from a single burst of instability.
    """
    window = float(cfg["oscillation"]["window_s"])
    min_switches = int(cfg["oscillation"]["min_switches"])
    if len(switches) < min_switches:
        return 0, 0

    times = [s.elapsed_s for s in switches]
    episodes = 0
    involved = 0
    i = 0
    n = len(times)
    while i < n:
        j = i
        while j + 1 < n and times[j + 1] - times[i] <= window:
            j += 1
        count_in_window = j - i + 1
        if count_in_window >= min_switches:
            episodes += 1
            involved += count_in_window
            i = j + 1  # jump past this episode (disjoint)
        else:
            i += 1
    return episodes, involved


def build_timeline(events_df: pd.DataFrame, cfg: dict, direction: Optional[str],
                    trial_end_s: float) -> TrialTimeline:
    """Top-level entry point: build the full reconstructed timeline for one trial."""
    switches = reconstruct_source_switches(events_df, cfg)
    vps_attempts = reconstruct_vps_attempts(events_df, cfg, trial_end_s)

    handover_attempts: List[HandoverAttempt] = []
    if direction in ("GPS_TO_VPS", "VPS_TO_GPS"):
        handover_attempts.append(detect_handover(events_df, cfg, direction, vps_attempts))
    elif direction is None:
        # Unknown direction: attempt both, keep whichever is evaluable.
        for d in ("GPS_TO_VPS", "VPS_TO_GPS"):
            handover_attempts.append(detect_handover(events_df, cfg, d, vps_attempts))

    false_handovers = detect_false_handovers(switches, cfg)
    episodes, involved = detect_oscillation(switches, cfg)

    return TrialTimeline(
        source_switches=switches,
        vps_attempts=vps_attempts,
        handover_attempts=handover_attempts,
        false_handovers=false_handovers,
        oscillation_episodes=episodes,
        oscillation_switch_count=involved,
    )
