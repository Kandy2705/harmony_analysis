"""
aggregate.py
------------
Aggregates per_trial_metrics rows into aggregate_results.csv (grouped by
harmony_version x direction [x scenario]) and paper_metrics.csv (the
clean 4-KPI-focused table for the paper).
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats


def _mean_ci(series: pd.Series, confidence: float) -> tuple[float, float, float]:
    """Return (mean, ci_low, ci_high) using a t-distribution CI."""
    s = pd.to_numeric(series, errors="coerce").dropna()
    n = len(s)
    if n == 0:
        return np.nan, np.nan, np.nan
    mean = float(s.mean())
    if n < 2:
        return mean, np.nan, np.nan
    sem = s.std(ddof=1) / np.sqrt(n)
    if sem == 0:
        return mean, mean, mean
    tcrit = scipy_stats.t.ppf(0.5 + confidence / 2.0, df=n - 1)
    return mean, mean - tcrit * sem, mean + tcrit * sem


def _proportion_ci(successes: int, n: int, confidence: float) -> tuple[float, float]:
    """Wilson score interval for a proportion; robust for small n / extreme p."""
    if n == 0:
        return np.nan, np.nan
    z = scipy_stats.norm.ppf(0.5 + confidence / 2.0)
    p = successes / n
    denom = 1 + z ** 2 / n
    centre = p + z ** 2 / (2 * n)
    adj = z * np.sqrt((p * (1 - p) + z ** 2 / (4 * n)) / n)
    lo = (centre - adj) / denom
    hi = (centre + adj) / denom
    return max(0.0, lo) * 100, min(1.0, hi) * 100


CONTINUOUS_METRICS = [
    "handover_latency_s",
    "pdr_transition_duration_s",
    "vps_localization_time_s_mean",
    "gps_accuracy_m_mean",
    "position_jump_m_reconstructed",
    "heading_jump_deg_reconstructed",
    "source_transitions_total",
    "unnecessary_source_switches",
]

RATE_METRICS = [
    # (numerator_col_builder, label)
]


def _continuous_block(df: pd.DataFrame, col: str, confidence: float) -> Dict[str, Any]:
    s = pd.to_numeric(df[col], errors="coerce").dropna() if col in df.columns else pd.Series(dtype=float)
    n = len(s)
    if n == 0:
        return {f"{col}_N": 0, f"{col}_mean": np.nan, f"{col}_std": np.nan, f"{col}_median": np.nan,
                f"{col}_Q1": np.nan, f"{col}_Q3": np.nan, f"{col}_min": np.nan, f"{col}_max": np.nan,
                f"{col}_ci_low": np.nan, f"{col}_ci_high": np.nan}
    mean, lo, hi = _mean_ci(s, confidence)
    return {
        f"{col}_N": n,
        f"{col}_mean": round(mean, 4),
        f"{col}_std": round(float(s.std(ddof=1)), 4) if n > 1 else 0.0,
        f"{col}_median": round(float(s.median()), 4),
        f"{col}_Q1": round(float(s.quantile(0.25)), 4),
        f"{col}_Q3": round(float(s.quantile(0.75)), 4),
        f"{col}_min": round(float(s.min()), 4),
        f"{col}_max": round(float(s.max()), 4),
        f"{col}_ci_low": round(lo, 4) if not np.isnan(lo) else np.nan,
        f"{col}_ci_high": round(hi, 4) if not np.isnan(hi) else np.nan,
    }


def _rate_block(numerator: int, denominator: int, label: str, confidence: float) -> Dict[str, Any]:
    pct = round(100.0 * numerator / denominator, 2) if denominator > 0 else np.nan
    lo, hi = _proportion_ci(numerator, denominator, confidence) if denominator > 0 else (np.nan, np.nan)
    return {
        f"{label}_numerator": numerator,
        f"{label}_denominator": denominator,
        f"{label}_percent": pct,
        f"{label}_ci_low": round(lo, 2) if not np.isnan(lo) else np.nan,
        f"{label}_ci_high": round(hi, 2) if not np.isnan(hi) else np.nan,
    }


def aggregate_trials(per_trial: pd.DataFrame, cfg: dict, exclude_invalid: bool = True) -> pd.DataFrame:
    """Group per-trial rows by harmony_version x direction [x scenario] and
    compute descriptive stats + 95% CI for continuous metrics, and rate +
    Wilson CI for HSR/FHR.
    """
    df = per_trial.copy()
    if exclude_invalid:
        df = df[df["validation_status"] != "INVALID"]

    group_cols = [c for c in cfg["aggregate"]["group_by"] if c in df.columns]
    if "scenario" in df.columns and "scenario" not in group_cols:
        group_cols.append("scenario")
    if not group_cols:
        df["_all"] = "ALL"
        group_cols = ["_all"]

    confidence = float(cfg["aggregate"]["confidence_level"])

    records: List[Dict[str, Any]] = []
    for keys, gdf in df.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        rec: Dict[str, Any] = dict(zip(group_cols, keys))
        rec["N_trials"] = int(len(gdf))

        # HSR — only over evaluable handovers (reconstructed success is not NaN)
        evaluable = gdf["handover_success_reconstructed"].dropna()
        successes = int(evaluable.sum()) if len(evaluable) else 0
        rec.update(_rate_block(successes, len(evaluable), "HSR", confidence))

        # FHR — false handovers / evaluable source switches, summed across trials
        fh = pd.to_numeric(gdf.get("false_handover_count", pd.Series(dtype=float)), errors="coerce").fillna(0)
        denom = pd.to_numeric(gdf.get("evaluable_switches", pd.Series(dtype=float)), errors="coerce").fillna(0)
        rec.update(_rate_block(int(fh.sum()), int(denom.sum()), "FHR", confidence))

        for col in CONTINUOUS_METRICS:
            rec.update(_continuous_block(gdf, col, confidence))

        records.append(rec)

    return pd.DataFrame.from_records(records)


def build_paper_metrics(aggregate_df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Clean, minimal table for direct use in the paper (spec section 6)."""
    rows = []
    for _, r in aggregate_df.iterrows():
        rows.append({
            "Version": r.get("harmony_version"),
            "Direction": r.get("direction"),
            "Scenario": r.get("scenario", ""),
            "N": r.get("N_trials"),
            "HSR_%": r.get("HSR_percent"),
            "HSR_95CI": f"[{r.get('HSR_ci_low')}, {r.get('HSR_ci_high')}]" if pd.notna(r.get("HSR_ci_low")) else "",
            "FHR_%": r.get("FHR_percent"),
            "FHR_95CI": f"[{r.get('FHR_ci_low')}, {r.get('FHR_ci_high')}]" if pd.notna(r.get("FHR_ci_low")) else "",
            "Handover_latency_s_mean": r.get("handover_latency_s_mean"),
            "Handover_latency_s_median": r.get("handover_latency_s_median"),
            "Source_switching_mean": r.get("source_transitions_total_mean"),
            "Unnecessary_switching_mean": r.get("unnecessary_source_switches_mean"),
            "VPS_localization_time_s_mean": r.get("vps_localization_time_s_mean_mean"),
            "PDR_transition_duration_s_mean": r.get("pdr_transition_duration_s_mean"),
            "Position_jump_m_mean": r.get("position_jump_m_reconstructed_mean"),
        })
    return pd.DataFrame(rows)
