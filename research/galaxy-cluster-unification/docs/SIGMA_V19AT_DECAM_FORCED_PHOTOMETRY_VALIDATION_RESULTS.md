# Sigma V19AT DECam forced-photometry validation results

## Decision

V19AT **passed every frozen validation gate** using the single
development-selected four-arcsecond area-scaled measurement. No alternate
validation aperture or deblending rule was inspected.

This validates a color-consistency measurement for the five provisional
singleton pairs. It is not yet absolute photometry, an ambiguous-member
association, a stellar-mass map, or a Sigma Gravity test.

## Frozen validation result

| Gate | Requirement | Result | Pass? |
|---|---:|---:|---:|
| Complete validation `griz` | 5/5 | **5/5** | yes |
| Crowded member 57 complete `griz` | required | **complete** | yes |
| Median absolute `g-r` error | at most 0.25 mag | **0.0979 mag** | yes |
| Median absolute `r-i` error | at most 0.25 mag | **0.0460 mag** | yes |
| Median absolute `i-z` error | at most 0.25 mag | **0.0491 mag** | yes |
| True-pair rank-one retrieval | at least 3/5 | **3/5** | yes |
| Mean reciprocal rank | at least 0.65 | **0.7333** | yes |
| Measurement memberships retained | 362/362 | **362/362** | yes |

True-pair ranks were `1, 1, 3, 3, 1` for members `26, 57, 66, 21, 71`,
respectively.

## The crowded-object recovery

Member 57 was the decisive case. The existing NSC rows had nonzero catalog
flags for 13/14 `g`, all 25 `r`, all 14 `i`, and 7/11 `z` measurements. V19AD
therefore had no accepted `r` or `i` four-arcsecond color at all.

The frozen image-level rule produced positive measurements for every exposure:

| Filter | Valid image measurements | Robust repeated-exposure scatter |
|---|---:|---:|
| `g` | 14/14 | 0.0376 mag |
| `r` | 25/25 | 0.0331 mag |
| `i` | 14/14 | 0.0433 mag |
| `z` | 11/11 | 0.0287 mag |

Its three validation-color residuals were -0.0632, +0.0699 and +0.0006 mag,
and its true partner ranked first. That is strong evidence that a fixed
image-level mask plus area correction can recover internally consistent colors
where the catalog-quality rule failed.

## Remaining weakness

The pass is not uniformly strong. Member 66 has residuals of -0.4786 mag in
`g-r` and -0.1755 mag in `r-i`, and ranks third. This is the same object that
already showed an unusual held-out `g-B` residual in V19AB. Member 21 also
ranks third despite moderate individual color residuals.

Thus the result clears the preregistered gate but does not support deterministic
identity selection from color alone. An ambiguous-candidate stage must retain
full likelihoods and a null/ambiguous state rather than forcing the nearest
color match.

## Calibration and scientific boundary

The validation transform was fitted only on the ten development objects and
was not recalibrated on these five results. It absorbs the stable filter offsets
caused by the Community Pipeline's rough Gaia-G-based characterization at this
declination. Those image magnitudes are still not final native-band AB total
fluxes.

Before stellar-mass inference, the project needs a separately frozen
field-star/color calibration or a homogeneous externally calibrated image
product. V19AT authorizes only a new ambiguous-candidate image likelihood—not
a mass-current map.

## Reproducibility

- Frozen protocol: `configs/sigma_v19at_decam_forced_photometry_validation.json`
- Runner: `scripts/run_sigma_v19at_decam_forced_photometry_validation.py`
- Machine report: `results/sigma_v19at_decam_forced_photometry_validation/report.json`
- Measurements and aggregates: `data/derived/sigma_v19at_decam_forced_photometry_validation/`
