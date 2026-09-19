"""
plotting.py
-----------
Publication-friendly plots (spec section 7). Colors come from a
matplotlib colormap (never hard-coded per-bar hex values); every axis is
labeled with version / direction / N / unit as applicable.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _palette(n: int, cfg: dict):
    cmap_name = cfg.get("plotting", {}).get("color_palette", "tab10")
    try:
        cmap = plt.get_cmap(cmap_name)
    except ValueError:
        cmap = plt.get_cmap("tab10")
    return [cmap(i % cmap.N) for i in range(n)]


def _bar_with_ci(ax, labels, values, ci_low, ci_high, ns, ylabel, title, colors):
    x = np.arange(len(labels))
    values = np.nan_to_num(np.array(values, dtype=float), nan=0.0)
    yerr_low = np.nan_to_num(values - np.array(ci_low, dtype=float), nan=0.0)
    yerr_high = np.nan_to_num(np.array(ci_high, dtype=float) - values, nan=0.0)
    yerr_low = np.clip(yerr_low, 0, None)
    yerr_high = np.clip(yerr_high, 0, None)
    ax.bar(x, values, color=colors, yerr=[yerr_low, yerr_high], capsize=4)
    ax.set_xticks(x)
    rot = 45 if len(labels) > 7 else 0
    ax.set_xticklabels([f"{l}\n(N={n})" for l, n in zip(labels, ns)], rotation=rot, ha="right" if rot else "center")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3)


def _labels_from_group(agg: pd.DataFrame) -> list[str]:
    parts = []
    for _, r in agg.iterrows():
        version = r.get("harmony_version", "?")
        direction = r.get("direction", "")
        scenario = r.get("scenario", "")
        label = f"{version}"
        if isinstance(direction, str) and direction:
            label += f"\n{direction}"
        if isinstance(scenario, str) and scenario:
            label += f"\n{scenario}"
        parts.append(label)
    return parts


def plot_hsr(agg: pd.DataFrame, out_dir: Path, cfg: dict) -> None:
    if agg.empty or "HSR_percent" not in agg.columns:
        return
    fig, ax = plt.subplots(figsize=(max(6, 1.2 * len(agg)), 5))
    labels = _labels_from_group(agg)
    colors = _palette(len(agg), cfg)
    _bar_with_ci(ax, labels, agg["HSR_percent"], agg["HSR_ci_low"], agg["HSR_ci_high"],
                 agg["N_trials"], "Handover Success Rate (%)",
                 "Handover Success Rate by Version / Direction", colors)
    ax.set_ylim(0, 105)
    fig.tight_layout()
    fig.savefig(out_dir / "handover_success_rate.png", dpi=cfg.get("plotting", {}).get("dpi", 150))
    plt.close(fig)


def plot_fhr(agg: pd.DataFrame, out_dir: Path, cfg: dict) -> None:
    if agg.empty or "FHR_percent" not in agg.columns:
        return
    fig, ax = plt.subplots(figsize=(max(6, 1.2 * len(agg)), 5))
    labels = _labels_from_group(agg)
    colors = _palette(len(agg), cfg)
    _bar_with_ci(ax, labels, agg["FHR_percent"], agg["FHR_ci_low"], agg["FHR_ci_high"],
                 agg["N_trials"], "False/Premature Handover Rate (%)",
                 "False Handover Rate by Version / Direction", colors)
    fig.tight_layout()
    fig.savefig(out_dir / "false_handover_rate.png", dpi=cfg.get("plotting", {}).get("dpi", 150))
    plt.close(fig)


def plot_latency(per_trial: pd.DataFrame, out_dir: Path, cfg: dict) -> None:
    df = per_trial.dropna(subset=["handover_latency_s"]) if "handover_latency_s" in per_trial.columns else pd.DataFrame()
    if df.empty:
        return
    groups = sorted(df["harmony_version"].astype(str).unique())
    data = [df[df["harmony_version"].astype(str) == g]["handover_latency_s"].values for g in groups]
    fig, ax = plt.subplots(figsize=(max(6, 1.2 * len(groups)), 5))
    colors = _palette(len(groups), cfg)
    tick_lbls = [f"{g}\n(N={len(d)})" for g, d in zip(groups, data)]
    try:
        bp = ax.boxplot(data, tick_labels=tick_lbls, patch_artist=True)
    except TypeError:
        bp = ax.boxplot(data, labels=tick_lbls, patch_artist=True)
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
    ax.set_ylabel("Handover latency (s)")
    ax.set_title("Handover Latency by Version")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "handover_latency.png", dpi=cfg.get("plotting", {}).get("dpi", 150))
    plt.close(fig)


def plot_source_switching(agg: pd.DataFrame, out_dir: Path, cfg: dict) -> None:
    col = "source_transitions_total_mean"
    if agg.empty or col not in agg.columns:
        return
    fig, ax = plt.subplots(figsize=(max(6, 1.2 * len(agg)), 5))
    labels = _labels_from_group(agg)
    colors = _palette(len(agg), cfg)
    ax.bar(labels, agg[col].fillna(0), color=colors)
    if len(labels) > 7:
        ax.tick_params(axis="x", labelrotation=45)
    ax.set_ylabel("Mean source transitions per trial")
    ax.set_title("Source Switching by Version / Direction")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "source_switching.png", dpi=cfg.get("plotting", {}).get("dpi", 150))
    plt.close(fig)


def plot_vps_localization_time(per_trial: pd.DataFrame, out_dir: Path, cfg: dict) -> None:
    col = "vps_localization_time_s_mean"
    if col not in per_trial.columns:
        return
    df = per_trial.dropna(subset=[col])
    if df.empty:
        return
    groups = sorted(df["harmony_version"].astype(str).unique())
    data = [df[df["harmony_version"].astype(str) == g][col].values for g in groups]
    fig, ax = plt.subplots(figsize=(max(6, 1.2 * len(groups)), 5))
    colors = _palette(len(groups), cfg)
    tick_lbls = [f"{g}\n(N={len(d)})" for g, d in zip(groups, data)]
    try:
        bp = ax.boxplot(data, tick_labels=tick_lbls, patch_artist=True)
    except TypeError:
        bp = ax.boxplot(data, labels=tick_lbls, patch_artist=True)
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
    ax.set_ylabel("VPS localization time (s)")
    ax.set_title("VPS Localization Time by Version")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "vps_localization_time.png", dpi=cfg.get("plotting", {}).get("dpi", 150))
    plt.close(fig)


def plot_gps_accuracy_distribution(per_trial: pd.DataFrame, out_dir: Path, cfg: dict) -> None:
    col = "gps_accuracy_m_mean"
    if col not in per_trial.columns:
        return
    s = per_trial[col].dropna()
    if s.empty:
        return
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.hist(s, bins=min(15, max(3, len(s))), color=_palette(1, cfg)[0], edgecolor="white")
    ax.set_xlabel("Mean GPS accuracy per trial (m)")
    ax.set_ylabel("Trial count")
    ax.set_title(f"GPS Accuracy Distribution (N={len(s)})")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "gps_accuracy_distribution.png", dpi=cfg.get("plotting", {}).get("dpi", 150))
    plt.close(fig)


def generate_all_plots(per_trial: pd.DataFrame, aggregate: pd.DataFrame, out_dir: Path, cfg: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    style = cfg.get("plotting", {}).get("style")
    if style and style in plt.style.available:
        plt.style.use(style)
    plot_hsr(aggregate, out_dir, cfg)
    plot_fhr(aggregate, out_dir, cfg)
    plot_latency(per_trial, out_dir, cfg)
    plot_source_switching(aggregate, out_dir, cfg)
    plot_vps_localization_time(per_trial, out_dir, cfg)
    plot_gps_accuracy_distribution(per_trial, out_dir, cfg)
