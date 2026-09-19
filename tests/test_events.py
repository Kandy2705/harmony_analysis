from __future__ import annotations

from harmony_analysis.events import (
    build_timeline,
    detect_false_handovers,
    detect_handover,
    detect_oscillation,
    reconstruct_source_switches,
    reconstruct_vps_attempts,
    SourceSwitch,
)
from .conftest import make_events_df


def test_reconstruct_source_switches(cfg):
    df = make_events_df([
        {"session_id": "S", "elapsed_s": 1.0, "event": "source_switched", "from_state": "Gps", "to_state": "Pdr"},
        {"session_id": "S", "elapsed_s": 5.0, "event": "source_switched", "from_state": "Pdr", "to_state": "Gps"},
        {"session_id": "S", "elapsed_s": 3.0, "event": "outdoor_state", "from_state": None, "to_state": "Calculating"},
    ])
    switches = reconstruct_source_switches(df, cfg)
    assert [s.elapsed_s for s in switches] == [1.0, 5.0]
    assert switches[0].from_source == "Gps" and switches[0].to_source == "Pdr"


def test_vps_attempt_success(cfg):
    df = make_events_df([
        {"session_id": "S", "elapsed_s": 10.0, "event": "vps_state", "to_state": "StartingVps"},
        {"session_id": "S", "elapsed_s": 12.0, "event": "vps_state", "to_state": "Scanning"},
        {"session_id": "S", "elapsed_s": 20.0, "event": "vps_state", "to_state": "IndoorLocalized",
         "note": "vps accepted"},
    ])
    attempts = reconstruct_vps_attempts(df, cfg, trial_end_s=30.0)
    assert len(attempts) == 1
    a = attempts[0]
    assert a.outcome == "success"
    assert a.end_s == 20.0
    assert a.scan_duration_s == 8.0  # 20 - 12 (scanning start)


def test_vps_attempt_timeout_fallback(cfg):
    df = make_events_df([
        {"session_id": "S", "elapsed_s": 10.0, "event": "vps_state", "to_state": "StartingVps"},
        {"session_id": "S", "elapsed_s": 12.0, "event": "vps_state", "to_state": "Scanning"},
        {"session_id": "S", "elapsed_s": 40.0, "event": "vps_state", "to_state": "IndoorLocalized"},
        {"session_id": "S", "elapsed_s": 40.0, "event": "handover_completed_approximate",
         "from_state": "VpsScanning", "to_state": "IndoorVps",
         "note": "30s VPS timeout; VPS was not accepted"},
    ])
    attempts = reconstruct_vps_attempts(df, cfg, trial_end_s=50.0)
    assert len(attempts) == 1
    assert attempts[0].outcome == "timeout_fallback"


def test_detect_handover_gps_to_vps_success(cfg):
    df = make_events_df([
        {"session_id": "S", "elapsed_s": 5.0, "event": "reliability_state",
         "from_state": "EnteringWithPdr", "to_state": "VpsScanning"},
        {"session_id": "S", "elapsed_s": 6.0, "event": "vps_state", "to_state": "StartingVps"},
        {"session_id": "S", "elapsed_s": 8.0, "event": "vps_state", "to_state": "Scanning"},
        {"session_id": "S", "elapsed_s": 20.0, "event": "vps_state", "to_state": "IndoorLocalized"},
    ])
    attempts = reconstruct_vps_attempts(df, cfg, trial_end_s=30.0)
    result = detect_handover(df, cfg, "GPS_TO_VPS", attempts)
    assert result.evaluable is True
    assert result.success is True
    assert result.latency_s == 15.0  # 20 - 5


def test_detect_handover_gps_to_vps_timeout_is_failure_even_with_indoorvps_state(cfg):
    """This is the exact trap described in the spec: FSM reaches IndoorVps via
    a fallback event, but it must be scored as a handover FAILURE."""
    df = make_events_df([
        {"session_id": "S", "elapsed_s": 5.0, "event": "reliability_state",
         "from_state": "EnteringWithPdr", "to_state": "VpsScanning"},
        {"session_id": "S", "elapsed_s": 6.0, "event": "vps_state", "to_state": "StartingVps"},
        {"session_id": "S", "elapsed_s": 8.0, "event": "vps_state", "to_state": "Scanning"},
        {"session_id": "S", "elapsed_s": 36.0, "event": "vps_state", "to_state": "IndoorLocalized"},
        {"session_id": "S", "elapsed_s": 36.0, "event": "handover_completed_approximate",
         "from_state": "VpsScanning", "to_state": "IndoorVps",
         "note": "30s VPS timeout; VPS was not accepted"},
    ])
    attempts = reconstruct_vps_attempts(df, cfg, trial_end_s=50.0)
    result = detect_handover(df, cfg, "GPS_TO_VPS", attempts)
    assert result.evaluable is True
    assert result.success is False  # NOT True, despite to_state == IndoorVps


def test_detect_handover_not_evaluable_when_never_started(cfg):
    df = make_events_df([
        {"session_id": "S", "elapsed_s": 1.0, "event": "trial_started"},
    ])
    result = detect_handover(df, cfg, "GPS_TO_VPS", [])
    assert result.evaluable is False
    assert result.success is None


def test_false_handover_detected_on_quick_revert(cfg):
    switches = [
        SourceSwitch(elapsed_s=10.0, from_source="Gps", to_source="Pdr"),
        SourceSwitch(elapsed_s=15.0, from_source="Pdr", to_source="Gps"),  # reverts within 8s window? no, 5s <= 8
    ]
    false_ones = detect_false_handovers(switches, cfg)
    assert len(false_ones) == 1
    assert false_ones[0].elapsed_s == 10.0


def test_false_handover_not_flagged_outside_window(cfg):
    switches = [
        SourceSwitch(elapsed_s=10.0, from_source="Gps", to_source="Pdr"),
        SourceSwitch(elapsed_s=100.0, from_source="Pdr", to_source="Gps"),  # far outside window
    ]
    false_ones = detect_false_handovers(switches, cfg)
    assert len(false_ones) == 0


def test_oscillation_detected_for_rapid_switch_burst(cfg):
    switches = [
        SourceSwitch(elapsed_s=0.0, from_source="Gps", to_source="Pdr"),
        SourceSwitch(elapsed_s=3.0, from_source="Pdr", to_source="Gps"),
        SourceSwitch(elapsed_s=6.0, from_source="Gps", to_source="Pdr"),
    ]
    episodes, involved = detect_oscillation(switches, cfg)
    assert episodes == 1
    assert involved == 3


def test_oscillation_not_triggered_for_isolated_switches(cfg):
    switches = [
        SourceSwitch(elapsed_s=0.0, from_source="Gps", to_source="Pdr"),
        SourceSwitch(elapsed_s=100.0, from_source="Pdr", to_source="Gps"),
    ]
    episodes, involved = detect_oscillation(switches, cfg)
    assert episodes == 0
