# Sigma V19AI FORS1 subpixel astrometry plan

## Hypothesis

V19AH passed every B and I astrometric gate and all R gates except the median-residual ceiling: R was 1.369 pixels versus 1.0. The shared detector reports integer local-maximum coordinates. V19AI isolates the hypothesis that integer peak quantization, rather than a wrong Gaia association, orientation, scale or field center, caused the near-failure.

V19AH remains failed. V19AI neither relaxes its ceiling nor reruns source detection or matching.

## Frozen sample

The input associations are exactly the 37 B, 30 I and 32 R V19AH RANSAC inliers. No source may be added, rematched or substituted. V19AI may reject an existing association only for a frozen local-centroid quality failure.

## Frozen centroid

For each foreground star, use a 21-by-21 stamp centered on its V19AH integer peak. Estimate the local background as the median from radius 7 to 10 pixels. Iterate a positive-flux center of light three times inside a 5-pixel aperture. Reject edge-truncated or nonfinite apertures, nonpositive net weight, shifts greater than 2 pixels, moment FWHM outside 1.5-12 pixels, or ellipticity above 0.7.

This is not science photometry. The net weight and moments are used only to locate and qualify foreground-star centroids.

## WCS and independent residual gate

Fit a TAN WCS from all accepted refined centroids and their unchanged V19AH epoch-propagated Gaia coordinates. In addition to the fitted residuals, score exact leave-one-star-out residuals: each foreground star is predicted by a WCS fitted without that star.

Every filter must retain at least 30 centroids, fitted median <= 1 pixel, fitted p95 <= 3 pixels, LOO median <= 1 pixel, LOO p95 <= 3 pixels and <= 0.75 arcsec. At least 20 exact Gaia IDs must survive in all three filters, and geometric-center coordinates must agree within 1 arcsec. R must improve on its V19AH fitted median of 1.368694 pixels.

## Boundary

V19AI contains no detector/source discovery, rematching, member coordinate, candidate coordinate, member cutout, deblender, photometric calibration, mass/current model, lensing target, halo map or gravity result. A pass authorizes only the next frozen full-field mask and PSF stage.
