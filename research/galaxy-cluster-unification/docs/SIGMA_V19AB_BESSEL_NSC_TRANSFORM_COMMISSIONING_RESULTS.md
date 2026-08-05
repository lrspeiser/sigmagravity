# Sigma V19AB Bessel-to-NSC transformation commissioning results

## Decision

V19AB **failed** its frozen commissioning gate.  The empirical transformation
is not authorized for the 57 ambiguous Bullet member cones.  No ambiguous
candidate was scored, no counterpart was selected, and no mass, current,
lensing, halo, or gravity quantity was opened.

The failure is narrow but physically meaningful: full magnitudes distinguish
the provisional validation pairs, while colors alone do not do so reliably.
The project must not treat aperture-sensitive total brightness as sufficient
identity evidence.

## Frozen validation results

| Diagnostic | Frozen requirement | Result | Pass? |
|---|---:|---:|---:|
| Median absolute error, `g-B` | at most 0.45 mag | 0.1250 mag | yes |
| Median absolute error, `r-R` | at most 0.45 mag | 0.0538 mag | yes |
| Median absolute error, `i-I` | at most 0.45 mag | 0.0821 mag | yes |
| Median absolute error, `z-I` | at most 0.45 mag | 0.0885 mag | yes |
| Full-offset rank-one retrieval | at least 4/5 | 5/5 | yes |
| Full-offset mean reciprocal rank | at least 0.80 | 1.00 | yes |
| Color-only rank-one retrieval | at least 3/5 | 2/5 | **no** |
| Color-only mean reciprocal rank | at least 0.65 | 0.60 | **no** |

The color-only true-pair ranks were `[1, 1, 3, 3, 3]` for validation members
`26, 57, 66, 71, 21`.  The full-offset ranks were all one.  Member 66 also has
a large held-out `g-B` residual of about `-0.612` mag, while the other median
offset errors remain small.

## What this teaches us

The ten-row robust regression learned a coherent average Bessel-to-NSC mapping,
and the five held-out singleton pairs were easy to retrieve when absolute
brightness was included.  But NSC and the original FORS1 paper do not
necessarily measure the same fraction of a galaxy: different apertures,
seeing, segmentation, and deblending can shift total magnitudes while leaving
colors less affected.  A match that succeeds only after using total brightness
is therefore too fragile for a baryonic mass map.

This does not show that the provisional singleton identities are wrong, nor
does it test the long-wave gravity hypothesis.  It shows that the present
empirical evidence is not sufficiently invariant to select the ambiguous
objects without a stronger forward model.

## Next route

The next source-side route should be frozen separately and use one of:

1. a forward SED calculation through measured Bessel, NSC/DECam, and HST/ACS
   filter curves, marginalizing aperture normalization and photometric errors;
2. forced photometry on common images with the same apertures and segmentation;
   or
3. higher-precision source coordinates or an independently matched catalog.

The first option is the least expensive immediate test.  It must predict
colors without using absolute aperture normalization and must pass the same
five validation anchors before any ambiguous member is opened.

## Reproducibility

- Frozen protocol: `configs/sigma_v19ab_bessel_nsc_transform_commissioning.json`
- Runner: `scripts/run_sigma_v19ab_bessel_nsc_transform_commissioning.py`
- Commissioning sample: `data/derived/sigma_v19ab_bessel_nsc_transform_commissioning/commissioning_sample.csv`
- Validation predictions: `data/derived/sigma_v19ab_bessel_nsc_transform_commissioning/validation_predictions.csv`
- Validation retrieval matrix: `data/derived/sigma_v19ab_bessel_nsc_transform_commissioning/validation_retrieval.csv`
- Machine-readable report: `results/sigma_v19ab_bessel_nsc_transform_commissioning/report.json`
