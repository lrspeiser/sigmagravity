# Sigma V19AJ FORS1 compact-core astrometry results

## Decision

**Failed unchanged: one infrared association short of the frozen minimum.** The compact-core hypothesis substantially improved the astrometry and resolved the V19AH red-band residual failure, but I retained 29 of 30 fixed Gaia associations rather than the required 30. Therefore no three-filter WCS, shared-star or geometric-center gate was evaluated as passing.

The frozen report SHA-256 is `8061d7ca76b7708aa928215f16bc15d652db2a4fd991c0a4a37cd9609c55502d`.

## Results

| Filter | Accepted | Fitted median | Fitted p95 | LOO median | LOO p95 | LOO p95 |
|---|---:|---:|---:|---:|---:|---:|
| B | 37/37 | 0.422 px | 2.084 px | 0.439 px | 2.407 px | 0.480 arcsec |
| I | 29/30 | not fitted | not fitted | not fitted | not fitted | not fitted |
| R | 31/32 | 0.577 px | 2.080 px | 0.624 px | 2.546 px | 0.508 arcsec |

All reported B and R fitted and leave-one-out residual gates passed. R's fitted median improved from 1.369 pixels in V19AH to 0.577 pixels, a 57.8% reduction. B retained every association. The compact core therefore addressed the broad-aperture drift identified in V19AI.

## Exact remaining failure

A diagnostic rerun of the frozen centroid function found only two rejections across all filters:

- I source `5484997146246822272`: centroid shift 2.862 pixels.
- R source `5484997047463884672`: centroid shift 2.050 pixels.

Both exceeded the unchanged 2-pixel ceiling; neither reached the later moment checks. The red filter could lose one of its 32 associations and still pass the minimum. The infrared filter had exactly 30 inputs and therefore could lose none.

The gate was not relaxed. Accepting a filter solely because it missed the threshold by one would make the evidence rule depend on the observed outcome.

## Interpretation and next test

The data now support a narrower conclusion: broad center-of-light apertures are inappropriate for these crowded FORS1 foreground stars, but a compact aperture is astrometrically effective for nearly every association. A final materially different local estimator is justified: interpolate the curvature immediately around the already-frozen integer peak, so distant asymmetric or neighboring flux cannot move the solution. Its estimator and all gates must be frozen before evaluating its subpixel offsets.

No member/candidate coordinate or cutout, science photometry, deblend, mass/current model, lensing/halo payload, gravity equation or holdout was opened or changed. This is an observational preparation result, not evidence for or against long-wavelength Sigma Gravity.
