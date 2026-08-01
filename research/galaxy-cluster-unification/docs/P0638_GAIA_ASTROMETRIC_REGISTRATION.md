# P0638 Gaia astrometric registration

## Why this stage is necessary

All 13 H I moment-0 products have celestial WCS, but only four of the V-band
images do. Independently centering the gas and stellar images would erase real
component offsets; centering on the brightest optical clump would be unstable
for irregular galaxies. Either shortcut could manufacture or suppress the
geometric signal being tested.

P0638 instead registers each optical image from foreground Gaia DR3 stars. The
galaxy pixels and all velocity products are excluded from the fit.

## Frozen method

The initial plate scale and field center come from the independent P0637
photometry. Point sources above eight background standard deviations are
matched against Gaia stars brighter than G=19. Eight possible image
orientations are searched, a two-dimensional translation mode supplies robust
tentative pairs, and RANSAC rejects galaxy clumps, cosmic rays, blends, and
high-proper-motion outliers. The inlier sky/pixel pairs define a final TAN WCS.

Every image must have at least 30 Gaia inliers, median residual below one pixel,
95th-percentile residual below three pixels and 2.5 arcsec, and plate-scale
correction below 2%. The four images with archived WCS provide an external
cross-check: the cataloged galaxy center must agree within 2.5 pixels.

## Scientific boundary

This is an astrometric calibration, not a galaxy fit. It uses no velocity
field, rotation curve, kinematic center, kinematic orientation, gravitational
formula, or target residual. The resulting WCS lets the next stage retain
measured gas-star offsets while constructing physical baryonic maps.
