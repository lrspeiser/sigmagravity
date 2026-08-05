# Sigma V19AH FORS1 Gaia astrometry plan

## Purpose

V19AG established that the original FORS1 B/R/I frames pass global detector-calibration gates. Their inherited WCS is deliberately marked approximate because the active mosaic removes pre/overscan columns. V19AH fits a new celestial WCS from foreground Gaia DR3 stars before any cluster-member coordinate or member-centered cutout is opened.

This is an astrometric calibration, not a deblending, mass, lensing or gravity fit.

## Frozen inputs

The only image inputs are the three hash-locked V19AG calibrated full frames. The foreground reference is the already frozen Bullet-field Gaia DR3 cone acquired for V19G Chandra registration. V19AH reads only its Gaia table; it does not open a Chandra source catalog, source match, registered event image or shock result.

Gaia sources must have G <= 19, RUWE < 1.4, at least eight visibility periods, no duplicate flag, a 5- or 6-parameter astrometric solution, and finite proper motion. Positions are propagated from the Gaia reference epoch to each 1998 exposure epoch. The `pmra` cosine convention is applied explicitly.

## Frozen solve

For detection only, the small nonfinite flat-mask fraction is replaced by the image's finite global median; the saved science pixels are not changed. Full-field peaks must exceed eight robust background standard deviations and be separated by at least 2 arcsec. The central 90 arcsec is excluded identically in every band so cluster galaxies cannot drive the WCS.

The initial center and scale come from each source FITS header, not the cluster light. Eight discrete image orientations are tested. A translation histogram supplies tentative pairs and a fixed-seed RANSAC similarity transform rejects galaxy clumps, blends, cosmic rays and saturated artifacts. The inliers produce a final TAN WCS.

## Frozen gates

Each of B, R and I must independently have:

- at least 30 Gaia inliers;
- median residual <= 1 pixel;
- 95th-percentile residual <= 3 pixels and <= 0.75 arcsec;
- similarity-scale correction <= 2%;
- the expected identity orientation;
- absolute similarity rotation <= 2 degrees.

At least 20 Gaia source IDs must be inliers in all three filters, and the sky coordinates assigned to the geometric image center by any two filters must agree within 1 arcsec.

## Leakage boundary

V19AH contains no cluster-member coordinate, candidate coordinate, member cutout, source photometry, deblender, photometric calibration, stellar-population model, mass/current inference, lensing target, halo map or gravity formula. A pass authorizes a separately frozen full-field bad/saturated/vignetted mask and Gaia-star PSF audit. A failure is not a Sigma Gravity failure.
