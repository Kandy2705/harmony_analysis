from __future__ import annotations

import numpy as np
import pandas as pd

from harmony_analysis.aggregate import _mean_ci, _proportion_ci, aggregate_trials


def test_proportion_ci_bounds_are_sane():
    lo, hi = _proportion_ci(8, 10, 0.95)
    assert 0 <= lo <= 80 <= hi <= 100


def test_proportion_ci_zero_n_returns_nan():
    lo, hi = _proportion_ci(0, 0, 0.95)
    assert np.isnan(lo) and np.isnan(hi)


def test_mean_ci_single_value_has_nan_bounds():
    mean, lo, hi = _mean_ci(pd.Series([5.0]), 0.95)
    assert mean == 5.0
    assert np.isnan(lo) and np.isnan(hi)


def test_mean_ci_widens_with_more_variance():
    tight = pd.Series([5.0, 5.1, 4.9, 5.05, 4.95])
    wide = pd.Series([1.0, 9.0, 2.0, 8.0, 3.0])
    _, lo_t, hi_t = _mean_ci(tight, 0.95)
    _, lo_w, hi_w = _mean_ci(wide, 0.95)
    assert (hi_w - lo_w) > (hi_t - lo_t)


def test_aggregate_trials_groups_by_version_and_direction(cfg):
    per_trial = pd.DataFrame([
        {"session_id": "S1", "harmony_version": "V1", "direction": "GPS_TO_VPS",
         "validation_status": "VALID", "handover_success_reconstructed": 1,
         "false_handover_count": 0, "evaluable_switches": 2,
         "handover_latency_s": 10.0, "pdr_transition_duration_s": 5.0,
         "vps_localization_time_s_mean": 8.0, "gps_accuracy_m_mean": 3.0,
         "position_jump_m_reconstructed": np.nan, "heading_jump_deg_reconstructed": np.nan,
         "source_transitions_total": 2, "unnecessary_source_switches": 0},
        {"session_id": "S2", "harmony_version": "V1", "direction": "GPS_TO_VPS",
         "validation_status": "VALID", "handover_success_reconstructed": 0,
         "false_handover_count": 1, "evaluable_switches": 3,
         "handover_latency_s": np.nan, "pdr_transition_duration_s": 6.0,
         "vps_localization_time_s_mean": 30.0, "gps_accuracy_m_mean": 4.0,
         "position_jump_m_reconstructed": np.nan, "heading_jump_deg_reconstructed": np.nan,
         "source_transitions_total": 3, "unnecessary_source_switches": 3},
        {"session_id": "S3", "harmony_version": "V2", "direction": "GPS_TO_VPS",
         "validation_status": "INVALID", "handover_success_reconstructed": np.nan,
         "false_handover_count": np.nan, "evaluable_switches": np.nan,
         "handover_latency_s": np.nan, "pdr_transition_duration_s": np.nan,
         "vps_localization_time_s_mean": np.nan, "gps_accuracy_m_mean": np.nan,
         "position_jump_m_reconstructed": np.nan, "heading_jump_deg_reconstructed": np.nan,
         "source_transitions_total": np.nan, "unnecessary_source_switches": np.nan},
    ])
    agg = aggregate_trials(per_trial, cfg, exclude_invalid=True)
    assert len(agg) == 1  # V2 excluded (INVALID, and it's its own group anyway)
    row = agg.iloc[0]
    assert row["harmony_version"] == "V1"
    assert row["N_trials"] == 2
    assert row["HSR_numerator"] == 1
    assert row["HSR_denominator"] == 2
    assert row["HSR_percent"] == 50.0
    assert row["FHR_numerator"] == 1
    assert row["FHR_denominator"] == 5
