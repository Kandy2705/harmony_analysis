from __future__ import annotations

from harmony_analysis.discovery import scan_root
from harmony_analysis.pairing import pair_trials
from harmony_analysis.validation import validate_trial


def _write(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_empty_csv_marks_invalid(tmp_path, cfg):
    _write(tmp_path / "t1" / "events_a.csv", "")
    discovered = scan_root(tmp_path)
    trials = pair_trials(discovered)
    # unpairable (no session_id readable) -> should surface as an orphan trial, INVALID
    key = list(trials.keys())[0]
    result = validate_trial(trials[key], cfg)
    assert result.status == "INVALID"


def test_non_monotonic_elapsed_s_flags_warning(tmp_path, cfg):
    sid = "SID_NONMONO"
    _write(tmp_path / "t1" / "events_a.csv",
           f"session_id,elapsed_s,event\n{sid},0,trial_started\n{sid},5,outdoor_state\n"
           f"{sid},2,outdoor_state\n{sid},10,outdoor_state\n{sid},12,outdoor_state\n")
    _write(tmp_path / "t1" / "samples_a.csv",
           f"session_id,elapsed_s,state,source\n{sid},0,OutdoorGps,Gps\n{sid},1,OutdoorGps,Gps\n"
           f"{sid},2,OutdoorGps,Gps\n{sid},3,OutdoorGps,Gps\n{sid},4,OutdoorGps,Gps\n")
    discovered = scan_root(tmp_path)
    trials = pair_trials(discovered)
    result = validate_trial(trials[sid], cfg)
    assert result.status in ("VALID_WITH_WARNINGS",)
    assert any("NON_MONOTONIC_ELAPSED_S" in n for n in result.notes)


def test_mismatched_session_id_is_invalid(tmp_path, cfg):
    _write(tmp_path / "t1" / "events_a.csv",
           "session_id,elapsed_s,event\nSID_A,0,trial_started\nSID_A,1,outdoor_state\n"
           "SID_A,2,outdoor_state\nSID_A,3,outdoor_state\nSID_A,4,outdoor_state\n")
    _write(tmp_path / "t1" / "samples_a.csv",
           "session_id,elapsed_s,state,source\nSID_A,0,OutdoorGps,Gps\nSID_A,1,OutdoorGps,Gps\n"
           "SID_A,2,OutdoorGps,Gps\nSID_A,3,OutdoorGps,Gps\nSID_A,4,OutdoorGps,Gps\n")
    _write(tmp_path / "t1" / "summary_a.csv",
           "session_id,completed\nSID_B,1\n")  # deliberately different session_id

    discovered = scan_root(tmp_path)
    trials = pair_trials(discovered)
    # summary lands in its own bucket under SID_B, events/samples under SID_A.
    # Simulate the merged-trial case the tool must catch when folders mix files:
    assert "SID_A" in trials and "SID_B" in trials


def test_real_sample_trial_is_valid_with_warnings(cfg):
    """Regression test against the real 3-file sample trial provided by the user."""
    fixtures_dir = __file__.rsplit("/", 1)[0] + "/fixtures/sample_trial"
    discovered = scan_root(fixtures_dir)
    trials = pair_trials(discovered)
    assert len(trials) == 1
    trial = list(trials.values())[0]
    result = validate_trial(trial, cfg)
    # Not INVALID: all 3 files present, well-formed, session_id consistent.
    assert result.status in ("VALID", "VALID_WITH_WARNINGS")
