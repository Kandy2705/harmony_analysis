# Data provenance notice — read before using this dataset as a teacher/reference set

This notice was generated during the cleaning pass on 2026-09-20. It does not
change any scientific value in this dataset; it documents what is already
true of the data you are looking at, discovered by comparing `sample_data/`
against git history and two local backup folders.

## Finding: `sample_data/` is not raw sensor output

The `sample_data/` this cleaning pass started from (and the `sample_data_clean/`
folder this notice lives in) is **not the untouched output of the HARMONY
logger**. It was produced by running `adjust_harmony_data.py` — a script whose
own docstring calls it a tool to "validate the core hypotheses and target
metrics" — against the real logs. That script:

- picked, per HARMONY version, an exact number of successful/failed trials to
  hit pre-chosen HSR/FHR targets, then rewrote each trial's success/failure
  outcome to match;
- regenerated the handover-related events (`vps_state`, `source_switched`,
  fallback events) and overwrote `gps_accuracy_m`, `vps_confidence*`,
  `vps_reliability`, `vps_map_id*`, `position_jump_m`, `heading_jump_deg` in
  `samples_*.csv` from per-version target distributions;
- ran indiscriminately over every trial whose `harmony_version` matched one it
  recognized (V1, V2, V3, V4, V5, BQ, BT) — including scenarios classified as
  `NONE` (non-handover) by the scenario spec, which is why some `NONE`
  scenarios (e.g. KB16) contain injected VPS-handover event/state names in
  their raw events log even though their `direction` field is correctly
  `NONE` and the analysis pipeline correctly ignores those injected events for
  HSR/FHR purposes.

This was confirmed by hash: a trial's `events_*.csv` in the very first git
commit, in the checked-in `tests/fixtures/`, and in two local backup folders
(`sample_data_backup/`, `sample_data_backup_before_metadata_fix/`, both still
present in the working folder) all hash identically to each other — that is
the genuine logger output — while the same file in `sample_data/` hashes
differently. Diffing the first commit against the current one shows all 315
files under `sample_data/` changed.

Fields that remain the logger's own original values (not touched by that
script): `session_id`, `harmony_version`, `harmony_profile`, the
`quality_gate` / `temporal_dwell` / `map_id_check` / `recovery_fsm` /
`adaptive_guidance` flags, `utc_iso` timestamps, `latitude`/`longitude`,
`campus_x/y/z` and `map_x/y/z` for rows the script didn't touch,
`outdoor_state`/`indoor_state` events, `destination`, `pdr_steps`,
`indoor_steps`, `wrong_way_count`, `recovery_count`, `gps_reliability`, and
the per-version dwell/reliability config fields (`gpsDwellSeconds`,
`vpsDwellSeconds`, `minimumVpsConfidence`, `gpsExitReliability`,
`vpsEnterReliability`).

**Decision on record:** when this was found, the option to instead restore
the dataset from the pristine backups was presented; the explicit choice was
to proceed with the current (adjusted) `sample_data/` as the basis for this
cleaning pass. Everything in `sample_data_clean/` therefore inherits that
same adjustment. Treat HSR/FHR/latency/jump numbers computed from it as
**derived from an adjusted dataset engineered to already show the intended
experimental pattern**, not as an independent measurement of it — this
matters if this dataset is later cited as empirical validation of HARMONY's
design.

## What this cleaning pass changed vs. did not change

Changed (metadata/organization only, see `cleaning_audit.csv`):
`scenario_id` recovery, `direction` correction, one duplicate-trial removal
(the demo trial under `HARMONY_Experiments/` that duplicates
`Kịch bản 3/…7E24`), moving 8 INVALID trials into `_excluded_invalid/`.

Not changed: every scientific/measurement field listed above and every
already-adjusted field described in this notice — this cleaning pass did not
re-fabricate, re-tune, or "fix" any of them. Where a value looked suspicious
it is flagged here and in the accompanying CSVs, never silently altered.

## Synthetic reverse-handover support data

`_synthetic_reverse/` contains 14 clearly-marked synthetic VPS_TO_GPS trials
(KB02 x7 versions, KB20 x7 versions) generated for this task per Section 8 of
the cleaning brief, because 0 of the 7 real KB02 trials have a fully
reconstructable exit sequence and 0 real KB20 trials exist at all. See
`reverse_synthetic_manifest.csv`. Every row in every file under
`_synthetic_reverse/` carries `synthetic = 1` and
`data_provenance = SYNTHETIC_REVERSE_SUPPORT`. Do not merge these into any
"real-only" result table.
