# Sigma V19AV signed-flux stack results

## Decision

V19AV **failed closed** because only 200/568 candidates (35.21%) reached
stacked signal-to-noise of at least three in every `griz` band, below the
frozen 90% requirement.

The signed stack itself passed every anchor-consistency gate and every one of
the 57 members has at least one complete candidate. The failure is therefore a
candidate-population depth/completeness failure, not a failure to combine
multi-epoch fluxes.

## Frozen result

| Gate | Requirement | Result | Pass? |
|---|---:|---:|---:|
| Anchor stack rows | 75 | 75 | yes |
| Candidate stack rows | 2,840 | 2,840 | yes |
| Validation `g-r` median error | at most 0.25 mag | 0.0922 mag | yes |
| Validation `r-i` median error | at most 0.25 mag | 0.0457 mag | yes |
| Validation `i-z` median error | at most 0.25 mag | 0.0564 mag | yes |
| Validation rank-one retrieval | at least 3/5 | 3/5 | yes |
| Validation mean reciprocal rank | at least 0.65 | 0.7333 | yes |
| Complete `griz` candidates | at least 90% | **35.21%** | **no** |
| Members with at least one complete candidate | 57/57 | 57/57 | yes |
| Candidate association score | forbidden | none | yes |

The validation true-pair ranks remain `1, 1, 3, 3, 1`; stacking did not alter
the previously validated behavior.

## Interpretation

The Huber stack solves the exact problem it was designed to solve: it includes
all signed exposures, preserves the anchor color relation, and produces a
stable multi-epoch observable. It does not make faint or spurious broad-cone
candidates into high-significance four-band detections.

This distinction matters. V19AA deliberately retained every catalog source in
six-arcsecond cones because the published Bullet right ascensions were rounded
to whole time-seconds. Many of those 568 sources are unrelated foreground or
background objects, and some are too faint for robust four-band DECam aperture
photometry. Requiring 90% completeness was intentionally stringent and has
now rejected the current image depth.

The project must not respond by lowering the S/N threshold or the 90% gate.
V19AU and V19AV are two different failed candidate-completeness tests. The
current single-exposure DECam family should pause here.

## Existing deeper-catalog check

The already-acquired HSC route cannot replace it for the Bullet Cluster. Among
these 568 candidates, 529 have some HSC identity but only 32 have all three
requested HST/ACS-like HSC bands. The earlier full HSC audit found only 46 of
793 Bullet cone candidates with complete F435W/F606W/F814W.

The next data route should therefore be a homogeneous, deeper, calibrated
coadd catalog or coadd images—most plausibly DES DR2/DELVE in this footprint—
with its coverage and public schema frozen before any candidate-level query.
If no such source gives materially better complete-band coverage, the honest
output is a broad association posterior with many null/low-information states,
not a forced complete stellar mass map.

## Claim boundary

V19AV does not score or choose any candidate and does not infer mass, current,
lensing, halo or gravity. Its 10/5 anchor check is consistency evidence on an
already-open validation set, not a new holdout.

Reproducibility:

- `configs/sigma_v19av_signed_flux_stack.json`
- `scripts/run_sigma_v19av_signed_flux_stack.py`
- `results/sigma_v19av_signed_flux_stack/report.json`
- `data/derived/sigma_v19av_signed_flux_stack/`
