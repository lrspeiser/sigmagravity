# Sigma V19AJ FORS1 compact-core astrometry plan

## Reason for this test

V19AI replaced integer source peaks with a 5-pixel-radius positive-flux centroid. B improved strongly, but I retained only 27 of 30 stars and R retained only 27 of 32, below the unchanged minimum of 30. All nine rejected associations failed for one reason only: the centroid moved more than 2 pixels. None failed the finite-data, FWHM or ellipticity checks. The rejected shifts were 2.018-3.873 pixels, consistent with a broad aperture being pulled toward neighboring or asymmetric light in this crowded field.

V19AJ tests that measurement explanation directly. It does not relax the shift, star-count, residual, center-consistency or shape gates.

## Frozen change

Use the exact V19AH Gaia associations and integer starting peaks. Reduce the stamp from 21 by 21 to 13 by 13 pixels, the positive-flux aperture radius from 5 to 3 pixels, and the background annulus from 7-10 to 4.5-6 pixels. Iterate the same center-of-light operation three times. All other centroid rejection gates remain identical to V19AI.

This compact core should be less sensitive to adjacent sources while still measuring the central stellar point-spread function. The test can fail either by still losing stars or by retaining them but producing poor independent astrometric predictions.

## Unchanged evidence gates

Every filter must retain at least 30 fixed associations, fitted median <= 1 pixel, fitted p95 <= 3 pixels, exact leave-one-out median <= 1 pixel, leave-one-out p95 <= 3 pixels and <= 0.75 arcsec. At least 20 exact Gaia IDs must survive in all filters, geometric centers must agree within 1 arcsec, and R must improve on its V19AH fitted median of 1.368694 pixels.

## Boundary

No detection, rematching, new star, member coordinate, candidate coordinate, deblending, science photometry, mass/current inference, lensing payload, halo payload or gravity result is permitted. This stage prepares a trustworthy baryonic observation; it does not test the long-wavelength Sigma field itself.
