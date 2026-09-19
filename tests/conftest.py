from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import yaml

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.yaml"


@pytest.fixture(scope="session")
def cfg():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def make_events_df(rows):
    """rows: list of dicts; missing columns are filled with NaN."""
    base_cols = ["utc_iso", "session_id", "harmony_version", "harmony_profile",
                 "elapsed_s", "event", "from_state", "to_state", "source",
                 "destination", "note", "campus_x", "campus_y", "heading_deg"]
    df = pd.DataFrame(rows)
    for c in base_cols:
        if c not in df.columns:
            df[c] = pd.NA
    return df


def make_samples_df(rows):
    base_cols = ["session_id", "elapsed_s", "state", "source", "active_source",
                 "gps_accuracy_m", "gps_reliability", "vps_reliability", "vps_valid",
                 "pdr_confidence", "campus_x", "campus_y", "heading_deg"]
    df = pd.DataFrame(rows)
    for c in base_cols:
        if c not in df.columns:
            df[c] = pd.NA
    return df
