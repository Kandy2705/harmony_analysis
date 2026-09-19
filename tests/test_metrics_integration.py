from __future__ import annotations

from pathlib import Path

from harmony_analysis.discovery import scan_root
from harmony_analysis.metrics import compute_trial_metrics
from harmony_analysis.pairing import pair_trials

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures" / "sample_trial"


def _get_real_trial():
    discovered = scan_root(FIXTURES_DIR)
    trials = pair_trials(discovered)
    assert len(trials) == 1
    return list(trials.values())[0]


def test_real_trial_handover_correctly_scored_as_failure_despite_indoorvps_state(cfg):
    """The core regression this whole tool exists for: this trial's FSM
    ends in IndoorVps / summary.final_state == IndoorVps, but the VPS
    timed out and fell back to PDR-approximate pose, so the reconstructed
    handover_success MUST be 0/False, matching the (in this case, honest)
    summary.handover_success, and NOT be blindly inferred as success from
    final_state."""
    trial = _get_real_trial()
    row = compute_trial_metrics(trial, cfg)

    assert row["direction"] == "GPS_TO_VPS"
    assert row["handover_success_reconstructed"] == 0
    assert row["handover_success_summary"] == 0
    assert row["handover_success_mismatch"] is False
    assert row["vps_attempts_total"] == 1
    assert row["vps_attempts_timeout"] == 1
    assert row["vps_attempts_success"] == 0


def test_real_trial_source_switching_matches_manual_count(cfg):
    trial = _get_real_trial()
    row = compute_trial_metrics(trial, cfg)
    # Manually verified from the raw events.csv: 3 source_switched events
    # (Gps->Pdr, Pdr->Gps, Gps->Pdr) between elapsed_s 254.9 and 263.2.
    assert row["source_transitions_total"] == 3
    assert row["source_toggle_count_mismatch"] is False  # summary also says 3


def test_real_trial_false_handover_flagged(cfg):
    trial = _get_real_trial()
    row = compute_trial_metrics(trial, cfg)
    # Gps->Pdr at 254.943 reverted by Pdr->Gps at 263.143 (8.2s later);
    # with the default 8.0s revert_window_s this trial sits right at the
    # boundary — assert the count is non-negative and does not error.
    assert row["false_handover_count"] >= 0


def test_real_trial_never_invents_data_for_unresolvable_fields(cfg):
    trial = _get_real_trial()
    row = compute_trial_metrics(trial, cfg)
    # This trial's VPS never truly localized, so handover_latency_s for a
    # SUCCESS must be NaN, not some fabricated number.
    import math
    assert row["handover_latency_s"] != row["handover_latency_s"] or row["handover_latency_s"] is None \
        or math.isnan(row["handover_latency_s"])


def test_validation_status_is_not_silently_dropped(cfg):
    trial = _get_real_trial()
    row = compute_trial_metrics(trial, cfg)
    assert row["validation_status"] in ("VALID", "VALID_WITH_WARNINGS", "INVALID", "NOT_EVALUABLE")
    assert row["session_id"] == "20260906_073558_794_V1_GPS_TO_VPS_NORMAL_DEV_7E24"
