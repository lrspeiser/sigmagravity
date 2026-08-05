# Sigma V19AH FORS1 Gaia astrometry results

## Outcome

**Failed the unchanged all-filter astrometry gate.**

B and I independently pass every frozen Gaia gate. R fails one gate: its median Gaia residual is 1.369 pixels, above the preregistered 1.0-pixel ceiling. The result is retained as a data-preparation failure; no threshold was relaxed and no member, lens or gravity input was opened.

## Preserved engineering stop

The first execution stopped because Astropy rejected two numerically equivalent ICRS coordinate arrays carrying different frame metadata. It yielded no Gaia match, WCS, source count or residual. That report is preserved as `engineering_failure_frame_metadata.json` with SHA-256 `71d676e8130801e46537e8e3f8987bf71cda8f312eaedd2d35afff9aab755369`.

The only correction removed the `obstime` frame attribute from coordinates whose RA and Dec numbers had already been propagated to 1998. The Gaia rows, propagated coordinates, source detector, matcher, RANSAC, WCS fitter, thresholds and gates were unchanged. The corrected runner was re-frozen at SHA-256 `f2f44107cd8c86f4cd0ccfafec931ec39562ff024d7c7e630af0c3e1eed40f0d` before the successful match calls.

## Filter results

| Metric | B | I | R | Gate |
|---|---:|---:|---:|---:|
| Detected full-field peaks | 1080 | 897 | 1014 | diagnostic |
| Gaia inliers | 37 | 30 | 32 | >= 30 |
| Median residual (pixel) | 0.732 | 0.916 | **1.369** | <= 1.0 |
| 95th-percentile residual (pixel) | 2.513 | 2.794 | 2.669 | <= 3.0 |
| 95th-percentile residual (arcsec) | 0.501 | 0.558 | 0.533 | <= 0.75 |
| Similarity scale error | 0.239% | 0.204% | 0.211% | <= 2% |
| Similarity rotation | 0.039 deg | 0.063 deg | 0.064 deg | <= 2 deg |
| Orientation | identity | identity | identity | identity |

All filters use 3,259 foreground stars surviving the catalog-only Gaia quality selection. Twenty-seven exact Gaia source IDs are inliers in all three filters, above the frozen minimum of 20.

## Cross-filter geometry

The independently fitted sky coordinates at the geometric image center differ by:

- B-I: 0.140 arcsec;
- B-R: 0.044 arcsec;
- I-R: 0.114 arcsec.

All are far below the 1-arcsec cross-filter gate. Thus the failure is not a gross orientation, center or scale problem. It is the R-band median point-location accuracy.

## What the failure teaches us

The current shared foreground-star solver records the integer pixel returned by a local-maximum detector. It does not refine each matched star to a subpixel centroid before fitting the WCS. The R result is consistent with this quantization becoming the limiting error: its p95 residual, scale, rotation, orientation, inlier count and cross-filter center all pass, while the median is only 0.369 pixel above the gate.

That interpretation is a testable algorithmic hypothesis, not permission to edit V19AH. A new protocol may add a frozen, local-background-subtracted subpixel centroid for the already identified foreground-star peaks, rerun the identical catalog selection and gates, and require that the refinement improve independent residuals without losing inliers. It must not lower the 1-pixel ceiling.

## Scientific boundary

V19AH opened no cluster-member coordinate, candidate coordinate, member-centered cutout, photometry, deblending model, stellar population model, mass map, current map, Chandra source match, shock result, lensing target, halo map or gravity prediction. It changes no long-wavelength Sigma formula or parameter.

Consequently:

- V19AH is not a failure of the long-wavelength gravity premise;
- its WCS products are diagnostic and must not yet authorize member photometry;
- the next astrometry protocol must pass all three bands before full-field masks and Gaia-star PSFs are frozen.
