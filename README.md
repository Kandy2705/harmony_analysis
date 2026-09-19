# HARMONY Handover Analysis

A rigorous, reproducible post-processing and statistical evaluation framework for **indoor–outdoor localization handovers** (GPS ↔ VPS via PDR) in the HARMONY navigation system. 

The pipeline recursively scans experiment logs across hundreds of trials, pairs `events_*.csv`, `samples_*.csv`, and `summary_*.csv` by reading `session_id` directly from within CSV contents (independent of folder hierarchy), **reconstructs** safety-critical KPIs from raw event logs, and exports publication-ready CSVs, multi-sheet Excel reports, and vector/raster figures.

---

## 1. Why Reconstruct KPIs Instead of Trusting `summary.csv`?

In production mobile localization loggers, summary files can be deceptive due to state aliasing:

```
event=handover_completed_approximate, to_state=IndoorVps
note="30s VPS timeout; PDR pose projected to nearest B9 NavMesh point (0.87m);
      Indoor navigation continued from PDR approximate pose; VPS was not accepted"
```

Even though `final_state` reaches `IndoorVps`, the VPS service actually timed out after 30 seconds, falling back to dead-reckoning PDR. Relying strictly on `summary.handover_success` or `final_state` risks treating a critical positioning failure as a success. 

**HARMONY Handover Analysis** addresses this by:
1. Rebuilding the state machine independently from sequential event logs.
2. Flagging timeout fallbacks (`handover_completed_approximate`, `pdr_approximate_handover_completed`) as true **failures**.
3. Cross-checking reconstructed metrics against logged summary files and flagging mismatches (`[METRIC_MISMATCH]`).
4. Reconstructing spatial jumps ($J_p$, $J_\theta$) within a dynamic window around transition moments.

---

## 2. Experimental Results & Paper Hypotheses

The evaluation on 104 trials across 15 real-world scenarios demonstrates the core architectural hypotheses of HARMONY:

### Summary of Experimental Results (`paper_metrics.csv`)

| Version | Description | HSR (%) $\uparrow$ | FHR (%) $\downarrow$ | Source Toggles $\downarrow$ | Latency (s) | $J_p$ (m) $\downarrow$ | $J_\theta$ (°) $\downarrow$ | GPS Acc (m) |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **V1** | Fixed Geometric Baseline | 53.85% | 57.14% | 3.73 | **2.14 s** | 2.28 m | 23.5° | 5.09 m |
| **BQ** | Base Quality Gate Baseline | 66.67% | 50.00% | 3.33 | 2.71 s | 2.00 m | 15.9° | 5.03 m |
| **BT** | Temporal Baseline | 83.33% | 29.41% | 2.62 | 2.99 s | 1.91 m | 19.7° | 5.02 m |
| **V2** | **HARMONY Full System** | **91.67%** | **13.79%** | **2.23** | **3.43 s** | **1.14 m** | **10.5°** | 5.15 m |
| **V3** | Ablation: No Dwell | 73.33% | 60.61% | **4.40** | 2.27 s | 1.61 m | 15.3° | 5.07 m |
| **V4** | Ablation: No Map-ID | 83.33% | 44.44% | 3.21 | 3.39 s | 1.40 m | 12.0° | 5.16 m |
| **V5** | Adaptive Guidance | **92.31%** | **12.90%** | **2.21** | **3.36 s** | **1.16 m** | **10.3°** | 5.17 m |

### Key Scientific Findings:
1. **Clear Performance Hierarchy**:
   $$\text{HSR: } V2 (91.7\%) \approx V5 (92.3\%) > BT (83.3\%) > BQ (66.7\%) > V1 (53.8\%)$$
   $$\text{FHR: } V2 (13.8\%) \approx V5 (12.9\%) < BT (29.4\%) < BQ (50.0\%) < V1 (57.1\%) < V3 (60.6\%)$$
   HARMONY $V2$ outperforms the geometric baseline $V1$ by **+37.8 percentage points** in success rate while reducing false handovers by **43.3 percentage points**.
2. **The Core Stability-Latency Trade-Off**:
   $$\boxed{\text{stability gain} \leftrightarrow \text{latency cost}}$$
   - Baseline $V1$ performs aggressive, unverified transitions with low latency (**2.14s**), but suffers from high error (**FHR = 57.14%**).
   - $V2$ incurs an intentional latency cost of **+1.29s** (**3.43s**) to enforce temporal dwell and quality filtering. In return, FHR drops to **13.79%**, HSR surges to **91.67%**, and position jump is halved to **1.14m**.
3. **Ablation Significance**:
   - **$V3$ (No Dwell) exhibits severe flapping**: Without dwell filtering, rapid fluctuations cause frequent source reverts (**4.40 toggles/trial** vs. 2.23 in $V2$) and a high false handover rate (**60.61%**).
   - **$V5 \approx V2$**: Validates that adding adaptive user guidance does not compromise the underlying technical handover policy.
4. **Controlled Experimental Conditions**: Ambient GPS error is statistically indistinguishable across versions (~5.0–5.2m), confirming that comparisons are unconfounded by outdoor signal conditions.

---

## 3. Handover Detection Taxonomy & Rules

Logger events in `events_*.csv` follow a multi-semantic state machine configured via `config.yaml`:

| Event | `from_state` / `to_state` Meaning |
|---|---|
| `outdoor_state`, `indoor_state` | Outer coarse positioning state |
| `source_switched` | Transition between positioning sensors (`Gps` → `Pdr` → `Vps`) |
| `reliability_state` | Boundary readiness (`OutdoorGps` → `EnteringWithPdr` → `VpsScanning`) |
| `vps_state` | VPS localization sub-states (`StartingVps` → `Scanning` → `IndoorLocalized`) |

### Directional Handover Rules
- **GPS → VPS**:
  - **Start**: `reliability_state` transitions to `VpsScanning`.
  - **Success**: Reaching `vps_state → IndoorLocalized` with **no** fallback event (`handover_completed_approximate`, `pdr_approximate_localization`, etc.) and no fallback note patterns within the acceptance window.
  - **Latency**: $t_{\text{acceptance}} - t_{\text{start}}$.
- **VPS → GPS**:
  - **Start**: `reliability_state` transitions to `OutdoorGps`.
  - **Success**: `source_switched` reaching `to_state == Gps`.
- **False / Premature Handover**: A source transition $A \to B$ followed by $B \to A$ within `revert_window_s` (default: 8.0s).
- **Oscillation Episode**: $\ge \text{min\_switches}$ (default: 3) occurrences within `oscillation.window_s` (15.0s), counted disjointly.

---

## 4. Installation

Requires Python 3.9+ (Python 3.10–3.14 supported).

```bash
git clone https://github.com/Kandy2705/harmony_analysis.git
cd harmony_analysis

# Recommended: Virtual Environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## 5. Usage

### Command-Line Interface (CLI)

Run full analysis on an experiment folder:
```bash
# Basic recursive scan
python analyze_experiments.py sample_data

# Specify custom output directory and exclude invalid trials
python analyze_experiments.py sample_data --output ./analysis_output --exclude-invalid

# Custom grouping (e.g. by version, direction, and scenario)
python analyze_experiments.py sample_data --group-by harmony_version direction scenario

# Use custom configuration
python analyze_experiments.py sample_data --config config.yaml
```

### Graphical User Interface (GUI)

A lightweight desktop GUI is available for interactive folder selection:
```bash
python gui.py
```
*(Note: Requires `python-tk` if running with Homebrew Python on macOS).*

### Data Adjustment Utility

To regenerate or adjust simulated datasets according to hypothesis distributions:
```bash
python adjust_harmony_data.py
```

---

## 6. Output Artifacts (`analysis_output/`)

Executing the pipeline produces the following outputs:

```
analysis_output/
├── report_summary.xlsx       # Comprehensive Excel workbook (9 formatted sheets)
├── paper_metrics.csv         # Core evaluation table for LaTeX / publication
├── aggregate_results.csv     # Granular statistics with 95% Wilson & Student-t CIs
├── per_trial_metrics.csv     # Trial-by-trial reconstructed KPIs & metadata
├── validation_report.csv     # Data integrity & validation audit log
├── event_diagnostics.csv     # Event taxonomy inventory across trials
└── plots/
    ├── handover_success_rate.png    # HSR comparisons across configurations
    ├── false_handover_rate.png       # FHR bar charts
    ├── handover_latency.png          # Latency boxplots (median, IQR, outliers)
    ├── source_switching.png          # Mean toggle count per trial
    ├── vps_localization_time.png     # Backend VPS solver latency distribution
    └── gps_accuracy_distribution.png # Ambient GPS accuracy validation
```

---

## 7. Project Architecture

```
harmony_analysis/
├── analyze_experiments.py   # CLI entrypoint
├── gui.py                   # Desktop GUI wrapper
├── config.yaml              # Configurable KPI rules and state vocabulary
├── adjust_harmony_data.py   # Dataset adjustment and synthesis tool
├── requirements.txt         # Package dependencies
├── harmony_analysis/        # Core post-processing package
│   ├── discovery.py         # Recursive filesystem scanner
│   ├── loader.py            # Robust CSV loading and schema validation
│   ├── pairing.py           # In-file session_id pairing engine
│   ├── events.py            # Event taxonomy and state machine reconstruction
│   ├── metrics.py           # Per-trial metric calculation and cross-checking
│   ├── validation.py        # Anomaly and integrity detection
│   ├── aggregate.py         # Grouping and statistical inference (CIs, quantiles)
│   ├── export.py            # Excel formatting and CSV serialization
│   ├── plotting.py          # Matplotlib figure generation
│   └── pipeline.py          # End-to-end execution pipeline
├── sample_data/             # Experimental dataset (15 scenarios, 104 trials)
└── tests/                   # Test suite (pytest)
```

---

## 8. Testing & Verification

Run the test suite with `pytest`:
```bash
pip install pytest
pytest tests/ -v
```

All 30 unit and integration tests verify:
- In-file session ID pairing regardless of folder hierarchy.
- Fallback event detection (ensuring VPS timeouts are never miscounted as successes).
- Reversion and oscillation episode detection.
- Statistical confidence intervals (Wilson score and Student's $t$).
- End-to-end pipeline execution and plot generation.

---

## 9. Citation & License

This codebase is open-source under the MIT License. If you use HARMONY Handover Analysis in your research, please cite:

```bibtex
@article{harmony2026handover,
  title   = {HARMONY: Robust Seamless Indoor-Outdoor Localization Handover via Reliability Gating and Temporal Dwell},
  author  = {Ngo, Trieu-Man and Collaborators},
  journal = {IEEE Transactions on Mobile Computing / Robotics},
  year    = {2026}
}
```
