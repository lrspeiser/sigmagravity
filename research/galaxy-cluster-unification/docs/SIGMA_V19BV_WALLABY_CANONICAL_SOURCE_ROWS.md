# Sigma V19BV WALLABY canonical source rows

## Decision

V19BV provides a deterministic one-row-per-name view of the 592-source V19BU
WALLABY candidate universe, but it also shows that source-release choice is a
real systematic rather than a clerical detail.

All 119 repeated names are Hydra TR1/TR2 pairs. A frozen source-only default
policy ranks rows by published quality flag, successful kinematic-product
availability, source reliability, integrated source signal-to-noise, source
mask size, release number and archive identifiers. It never reads the
kinematic product itself.

The output contains 592 unique names and passes every integrity and target-
separation gate. It is a canonical processing view, **not** the final galaxy
holdout.

## Robustness result

Five prespecified priority orders were applied to each release pair. Only 27
of the 119 duplicate names choose the same row under all five policies. The
other 92 choose different TR1/TR2 rows under at least one reasonable ordering.

That result changes how the eventual test must be run:

- the default canonical row can drive ordinary preprocessing;
- all 92 policy-sensitive alternatives remain in the immutable V19BU input;
- their alternate source measurements must be propagated as a baryonic-source
  systematic or excluded under a target-blind eligibility rule; and
- an apparent Sigma/MOND/halo difference that disappears under the alternative
  release is not robust evidence for gravity.

The original audit briefly used `catalogue_id` as though it uniquely identified
a release row. It does not. The final implementation uses the archive primary
`id`, which exposed the 92 ambiguities before this checkpoint was committed.

## What selected the default rows

Across all 592 names, 473 were already unique. Among the 119 duplicates, the
first differing default criterion was:

| Criterion | Names |
|---|---:|
| Higher source reliability | 81 |
| Higher integrated source S/N | 25 |
| Higher kinematic-product availability flag | 12 |
| Lower source quality flag | 1 |

The final canonical view contains 109 names with the published successful-
model flag, matching the public kinematic release size. This is target
availability, not a velocity or gravity score.

## Target seal

No systemic velocity, fitted inclination, kinematic position angle, radial
grid, rotation speed, velocity-field pixel, model residual or halo result was
read. No final holdout was selected, and no action, universal constant or
Solar-System calculation changed. The raw alternatives and the derived CSV
are both hash-bound.

## Reproduction

```powershell
python scripts/build_sigma_v19bv_wallaby_canonical_source_rows.py
python -m pytest tests/test_sigma_v19bv_wallaby_canonical_source_rows.py -q
```

The machine-readable report is
`results/sigma_v19bv_wallaby_canonical_source_rows/report.json`.

## Primary source

The flag meanings and the recommendation to omit most overlapping Hydra TR1
entries from aggregate summaries are documented in the WALLABY DR1 source
paper: <https://arxiv.org/abs/2211.07094>.
