from __future__ import annotations

import shutil
from pathlib import Path

from harmony_analysis.pipeline import run_analysis

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures" / "sample_trial"


def test_run_analysis_end_to_end_on_real_sample(tmp_path):
    # Recreate a realistic nested folder structure to also exercise recursive scan.
    root = tmp_path / "HARMONY_Experiments" / "V1" / "GPS_TO_VPS_NORMAL" / "trial_01"
    root.mkdir(parents=True)
    for f in FIXTURES_DIR.glob("*.csv"):
        shutil.copy(f, root / f.name)

    out_dir = tmp_path / "analysis_output"
    result = run_analysis(root=tmp_path / "HARMONY_Experiments", output_dir=out_dir)

    assert result.n_trials_found == 1
    assert (out_dir / "per_trial_metrics.csv").exists()
    assert (out_dir / "aggregate_results.csv").exists()
    assert (out_dir / "paper_metrics.csv").exists()
    assert (out_dir / "validation_report.csv").exists()
    assert (out_dir / "event_diagnostics.csv").exists()
    assert (out_dir / "report_summary.xlsx").exists()
    assert (out_dir / "plots").is_dir()
    assert any((out_dir / "plots").iterdir())


def test_run_analysis_handles_empty_root_gracefully(tmp_path):
    empty_root = tmp_path / "empty"
    empty_root.mkdir()
    out_dir = tmp_path / "out"
    result = run_analysis(root=empty_root, output_dir=out_dir)
    assert result.n_trials_found == 0
    assert (out_dir / "per_trial_metrics.csv").exists()
