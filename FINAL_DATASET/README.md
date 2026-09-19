# HARMONY handover dataset — consolidated final version

This is the single, consolidated data folder for the HARMONY handover-analysis
project. It replaces every intermediate/scattered folder from the earlier
cleaning and generation passes (`sample_data/`, `sample_data_clean/`,
`combined_root*/`, the various `analysis_output_*` folders, etc.) — all of
that history is summarized here and in `PROVENANCE_MANIFEST.csv`.

## What's in this folder

- `KB01/` … `KB30/` — one folder per scenario, each containing every trial
  (as `events_*.csv` / `samples_*.csv` / `summary_*.csv` triplets) that
  passed validation for that scenario. `KB02/` also contains its 7 real
  trials plus 7 synthetic reverse-handover trials (see below); all other
  folders contain trials of a single provenance.
- `_synthetic_reverse/` — the 14 synthetic VPS_TO_GPS trials for KB02/KB20
  generated in Task 1 (also copied into `KB02/` and `KB20/` respectively for
  convenience; the manifest is the authoritative record).
- `_synthetic_missing_scenarios/` — the 42 synthetic trials for KB01, KB04,
  KB07, KB08, KB21, KB28 generated in Task 2 (also copied into their
  respective `KBxx/` folders).
- `_synthetic_expanded_scenarios/` — the 63 synthetic trials for KB06
  (replacement), KB09, KB12, KB13, KB15, KB23, KB26, KB27, KB29, generated
  after an explicit user override of the original "leave missing" decision
  (also copied into their respective `KBxx/` folders).
- `analysis_output/` — the pipeline's own output (`report_summary.xlsx`,
  `per_trial_metrics.csv`, `aggregate_results.csv`, `paper_metrics.csv`,
  `validation_report.csv`, `event_diagnostics.csv`, `plots/`) from running
  the unmodified HARMONY analysis pipeline against every folder above.
  215 trials in, 207 VALID + 8 VALID_WITH_WARNINGS (all 8 are the 7 KB28
  intentional-cancellation trials plus 1 pre-existing real trial with a
  logging gap) + 0 INVALID.
- `PROVENANCE_MANIFEST.csv` — one row per trial (all 215): session_id,
  scenario_id, harmony_version, direction, provenance (REAL /
  SYNTHETIC_SUPPORT), which generation pass produced it, its grounding
  strength, and its file path. This is the authoritative cross-reference —
  consult it before treating any number in `analysis_output/` as measured.

## Coverage: all 30 KB scenarios, but not all equally grounded

Every one of KB01–KB30 now has at least one trial that passes the pipeline's
own validator. That does **not** mean all 30 are equally trustworthy as
scientific measurements. In descending order of how much to trust a number
computed from each:

**Real, unmodified logger fields (still adjusted — see next section), 96
trials, 15 scenarios:** KB02, KB03, KB05, KB10, KB11, KB14, KB16, KB17, KB18,
KB19, KB22, KB24, KB25, KB30, plus KB02's own 7 real trials.

**Synthetic, but built entirely from real per-version distributions and real
template geometry (same rigor as each other), 63 trials, 9 scenarios:** KB01,
KB02 (its 7 reverse trials), KB04, KB06 (replacement — see below), KB07,
KB08, KB09, KB20, KB21, KB28, KB29. No real trial exists for these anywhere
in this dataset (or, for KB06, the 7 real trials that exist all crashed
mid-recording — see `excluded_trials.csv`'s KB06 rows, still preserved) — the
values are entirely constructed, though the *distributions and geometry*
feeding that construction are real.

**Synthetic and weakly grounded — structural placeholders only, 42 trials, 6
scenarios:** KB12 (rain), KB13 (glare), KB15 (crowd occlusion), KB23 (camera
obstruction), KB26 (novice-user proxy), KB27 (experienced-user proxy). These
were originally marked `MISSING_UNREPRESENTABLE` because this logger schema
has no field for rain, glare, crowd density, camera obstruction, or user
experience level. They were generated only after the user explicitly asked
to override that decision, having been shown this exact tradeoff. Their
construction reuses only values and columns that already exist in the real
schema (e.g. KB12's degraded GPS accuracy is this dataset's own real
worst-ever-recorded value, not an invented number) — but the *specific
external cause* named in each scenario's definition (rain, glare, a crowd,
an obstruction, a user's experience level) was never actually measured by
anything in this dataset. KB26 and KB27's own defining measures — help
requests, trust, an AR-on/off comparison — are **not represented at all**,
because representing them honestly would have required inventing new
columns/event types, which was never done. Do not report numbers from these
6 scenarios as measured effects of the named condition; they exist for FSM
structural coverage and engineering/diagnostic use only, per the user's own
framing when the override was requested.

**Do not merge SYNTHETIC_SUPPORT trials into a "real-only" results table.**
Filter `PROVENANCE_MANIFEST.csv` by `provenance == "REAL"` first for any
analysis that claims to measure HARMONY's real-world performance.

## The dataset's history (why "REAL" still means "adjusted")

The `sample_data/` this whole project started from was found, during Task 1,
to not be raw logger output: a script (`adjust_harmony_data.py`) had already
rewritten success/failure outcomes, GPS/VPS accuracy and confidence values,
and position/heading jump figures to hit pre-chosen per-version targets. The
pristine backups and first git commit were available and hash-identical, but
the explicit decision (made by the user when this was discovered) was to
proceed with the adjusted data rather than restore the pristine backup. Every
"REAL" trial in this folder therefore still carries that adjustment. See the
original `PROVENANCE_NOTICE.md` (preserved in `analysis_output/` history if
needed) for the full forensic detail — hashes, which fields were and weren't
touched, and the decision record.

## If you regenerate anything

Every synthetic trial's construction script is preserved: `step4_synthetic_reverse.py`
(KB02/KB20 reverse), `step6_synthetic_missing_scenarios.py` (KB01/04/07/08/21/28),
`step7_synthetic_expanded_scenarios.py` (KB09/12/13/15/23/26/27/29), and
`step8_kb06_replacement.py` (KB06). All four are deterministic (fixed seed
20260920) and reuse the same real-distribution-learning helper functions, so
rerunning them reproduces this exact folder byte-for-byte.
