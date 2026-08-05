# Sigma V19AK FORS1 local-quadratic astrometry results

## Decision

**Failed unchanged; end the local-centroid-estimator branch.** The local quadratic method retained 35/37 B, 29/30 I and 27/32 R associations. I and R therefore failed the frozen minimum of 30 before a WCS could be scored. The report SHA-256 is `0a4420953e558f133668d6b12fadb9b6cce7f5c8d6317fc2ffc7476526f6f27e`.

## Scored B result

The 35 accepted B associations passed every residual gate:

- fitted median 0.442 pixels and p95 2.240 pixels;
- exact leave-one-out median 0.469 pixels and p95 2.832 pixels;
- leave-one-out p95 0.565 arcsec.

This confirms that local quadratic interpolation can be accurate when the central pixels form a clean concave peak. It does not solve the cross-filter sample problem.

## Rejection anatomy

| Filter | Accepted | Centroid-shift rejects | Nonconcave-peak rejects |
|---|---:|---:|---:|
| B | 35/37 | 1 | 1 |
| I | 29/30 | 1 | 0 |
| R | 27/32 | 0 | 5 |

The I rejection was the same source that failed V19AJ: `5484997146246822272`, with a quadratic shift of 2.818 pixels. Five R stencils were not concave, including the V19AJ rejected source `5484997047463884672`. This means the immediate central pixels in R are often too noisy, asymmetric or blended for an unconstrained single-peak quadratic model.

## What was learned

Across the frozen variations:

1. Integer peaks gave adequate B and I astrometry but an R median of 1.369 pixels.
2. A broad 5-pixel center of light improved B but drifted toward surrounding flux, leaving 27 I and 27 R sources.
3. A compact 3-pixel center of light produced the best balance: 37 B, 29 I and 31 R, and drove R's fitted median to 0.577 pixels.
4. A 3-by-3 quadratic summit avoided broad flux but was not a valid concave surface for five R sources.

The limitation is therefore not simply subpixel quantization. It is the combination of crowding/asymmetry and band-dependent central-pixel structure. Repeating aperture sizes would tune the estimator to these outcomes without creating an independent solution.

## Next permissible strategy

Use a field-level astrometric/PSF method that estimates a spatially varying point-spread function from many stars and fits blends jointly, or acquire an external astrometrically calibrated image. That is materially different from another isolated centroid formula. It must receive its own frozen validation sample and gates.

No member/candidate payload, science photometry, baryonic mass/current model, lensing/halo payload, gravity equation or holdout was opened or changed. This failure concerns source preparation only and neither supports nor falsifies long-wavelength Sigma Gravity.
