# Sigma V19AI FORS1 subpixel astrometry results

## Outcome

**Failed the unchanged accepted-star-count gate.**

The fixed 5-pixel-aperture subpixel centroid sharply improved B-band astrometry, but I and R retained only 27 foreground stars each. The frozen minimum was 30 per filter, so V19AI stopped before fitting I or R and did not authorize member work.

## B-band result

B accepted 36 of its 37 frozen V19AH Gaia associations. Relative to the integer-peak V19AH result:

| Metric | V19AH integer peak | V19AI subpixel | V19AI gate |
|---|---:|---:|---:|
| Fitted median residual | 0.732 px | 0.361 px | <= 1.0 px |
| Fitted p95 residual | 2.513 px | 1.535 px | <= 3.0 px |
| Leave-one-out median | not scored | 0.375 px | <= 1.0 px |
| Leave-one-out p95 | not scored | 1.735 px / 0.346 arcsec | <= 3.0 px / 0.75 arcsec |

The accepted B stars had median centroid shift 0.475 pixel, median moment FWHM 3.795 pixels and median ellipticity 0.075. This confirms the basic diagnosis: integer peak quantization was materially limiting the earlier WCS.

## Why I and R stopped

A post-failure audit using the already frozen centroid function found that every rejection was caused by the unchanged 2-pixel maximum centroid shift:

- B: 1 rejected of 37;
- I: 3 rejected of 30, leaving 27;
- R: 5 rejected of 32, leaving 27.

No rejection was due to nonfinite pixels, edge truncation, FWHM or ellipticity. Most rejected shifts were only slightly beyond the threshold (2.02-2.61 pixels), with one I source at 3.87 pixels. The broad 5-pixel center-of-light aperture is therefore being pulled toward neighboring or asymmetric flux in this crowded field.

This does not justify raising the shift limit or lowering the 30-star count. A materially different next test should localize only the stellar core with a smaller fixed aperture and annulus, retain the same associations and all independent residual gates, and still forbid source rematching. Leave-one-out residuals will determine whether the resulting cores carry valid astrometric information.

## Scientific boundary

V19AI used only the frozen foreground-star associations and their local stamps. It performed no source detection or rematching and opened no member coordinate, member cutout, deblending model, photometric calibration, mass/current input, lensing target, halo map or gravity result. It changes no long-wavelength Sigma term or constant.
