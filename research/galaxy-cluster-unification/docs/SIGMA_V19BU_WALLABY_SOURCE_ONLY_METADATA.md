# Sigma V19BU WALLABY source-only metadata

## Decision

The independent galaxy holdout now has a real, hash-bound source-side
candidate universe rather than only a survey citation. The official CASDA TAP
service returned 711 WALLABY DR1 source-finding rows representing the 592
unique H I detections reported by the survey. The downloaded CSV is 210,929
bytes with SHA-256
`58e59cdfc48cf8192ff2038b949285adfa3ab0b1038603b31afcf81df8fbac4d`.

This is a **source-metadata checkpoint**, not a selected galaxy sample and not
a gravity result. No kinematic target value was queried or opened.

## Enforced source/target split

WALLABY publishes two distinct TAP tables:

- `AS102.wallaby_pilot_dr1_sourcefind_cat_v01` contains the source-finding
  candidate universe;
- `AS102.wallaby_pilot_dr1_kinmodel_cat_v01` contains the kinematic targets.

The executed source query projects only identity and provenance, sky position,
integrated H I flux and uncertainty, noise/reliability/quality metadata,
moment-zero angular geometry, distance and H I mass. The exact query and its
hash are frozen in the V19BU config.

The target table's **schema names** were inspected to establish an enforceable
deny-list. No target row was read. The committed CSV contains none of the
systemic-velocity, fitted-inclination, kinematic-position-angle, radial-grid,
rotation-curve, surface-density-model or uncertainty columns. It also contains
no velocity field, cube, residual or halo result.

The raw spectral cubes remain sealed because they encode both gas source
structure and line-of-sight kinematics. They can be opened only after the
action and universal constants are frozen, through the preregistered forward
observation-space control.

## What the source metadata proves

The 711 rows contain 119 repeated team-release entries; no silent
deduplication was performed. There are 592 unique source names and at most two
rows per name. The source-side catalog spans the Hydra, Norma and NGC 4636
group fields. All 711 rows have an H I mass value, and 641 have catalog
reliability at least 0.9.

The archive `kflag` metadata has 109 unique names at value 2, matching the
published size of the kinematically modeled release. This is an availability
cross-check only. Neither `kflag` nor any other field selects the eventual 32
or more WALLABY holdouts.

The result therefore proves that the primary independent pool is large enough
to support source-blind stratification. It does not prove that any individual
galaxy has complete stellar photometry, molecular gas, distance/inclination
covariance, adequate beam coverage or an admissible target.

## Next safe work

Before opening kinematics, join independent optical photometry and environment
metadata and freeze the eight required galaxy strata. Repeated team-release
rows must be resolved by a provenance rule that cannot see target values.
Only after one action and at most five constants are frozen may the separate
kinematic table and selected cubes be opened once for scoring.

## Reproduction

```powershell
python scripts/check_sigma_v19bu_wallaby_source_only_metadata.py
python -m pytest tests/test_sigma_v19bu_wallaby_source_only_metadata.py -q
```

The machine-readable result is
`results/sigma_v19bu_wallaby_source_only_metadata/report.json`.

## Primary public sources

- WALLABY DR1 data and TAP instructions:
  <https://wallaby-survey.org/data/data-pilot-survey-dr1/>
- CSIRO source-finding collection: <https://doi.org/10.25919/09yg-d529>
- Source-catalog paper: <https://arxiv.org/abs/2211.07094>
- Kinematic-catalog paper: <https://arxiv.org/abs/2211.07333>
