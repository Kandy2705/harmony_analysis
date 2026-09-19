from __future__ import annotations

from pathlib import Path

from harmony_analysis.discovery import scan_root
from harmony_analysis.pairing import pair_trials


def _write(path: Path, content: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_pairing_matches_by_session_id_not_folder_name(tmp_path):
    # Deliberately put events/samples/summary in DIFFERENT folder names
    # to prove pairing does not depend on folder naming conventions.
    sid = "SID_001"
    _write(tmp_path / "weird_folder_a" / "events_foo.csv",
           f"session_id,elapsed_s,event\n{sid},0,trial_started\n")
    _write(tmp_path / "another_weird_folder" / "samples_bar.csv",
           f"session_id,elapsed_s,state,source\n{sid},0,OutdoorGps,Gps\n")
    _write(tmp_path / "yet_another" / "summary_baz.csv",
           f"session_id,completed\n{sid},1\n")

    discovered = scan_root(tmp_path)
    trials = pair_trials(discovered)

    assert sid in trials
    trial = trials[sid]
    assert trial.events is not None and trial.events.ok
    assert trial.samples is not None and trial.samples.ok
    assert trial.summary is not None and trial.summary.ok


def test_pairing_flags_duplicate_files(tmp_path):
    sid = "SID_DUP"
    _write(tmp_path / "t1" / "events_a.csv", f"session_id,elapsed_s,event\n{sid},0,trial_started\n")
    _write(tmp_path / "t1" / "events_b.csv", f"session_id,elapsed_s,event\n{sid},1,trial_started\n")

    discovered = scan_root(tmp_path)
    trials = pair_trials(discovered)

    assert sid in trials
    trial = trials[sid]
    assert len(trial.duplicate_events) == 1
    codes = [i.code for i in trial.pairing_issues]
    assert "DUPLICATE_FILE" in codes


def test_pairing_reports_missing_files_via_validation(tmp_path, cfg):
    from harmony_analysis.validation import validate_trial

    sid = "SID_MISSING_SAMPLES"
    _write(tmp_path / "t1" / "events_a.csv",
           f"session_id,elapsed_s,event\n{sid},0,trial_started\n{sid},1,app_resumed\n"
           f"{sid},2,outdoor_state\n{sid},3,outdoor_state\n{sid},4,outdoor_state\n")

    discovered = scan_root(tmp_path)
    trials = pair_trials(discovered)
    trial = trials[sid]
    result = validate_trial(trial, cfg)

    assert result.status == "INVALID"
    assert any("MISSING_SAMPLES_FILE" in n for n in result.notes)


def test_scan_root_recursive_finds_nested_files(tmp_path):
    _write(tmp_path / "V1" / "SCENARIO_A" / "trial_01" / "events_x.csv",
           "session_id,elapsed_s,event\nS1,0,trial_started\n")
    _write(tmp_path / "V2" / "SCENARIO_B" / "trial_99" / "events_y.csv",
           "session_id,elapsed_s,event\nS2,0,trial_started\n")

    discovered = scan_root(tmp_path, recursive=True)
    assert len(discovered.events_files) == 2
