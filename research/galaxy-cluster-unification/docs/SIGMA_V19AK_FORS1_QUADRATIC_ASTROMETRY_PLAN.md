# Sigma V19AK FORS1 local-quadratic astrometry plan

## Hypothesis

V19AJ showed that a smaller light aperture recovered all B associations and enough R associations, with every scored astrometric residual gate passing. It missed the infrared count gate by one because one compact center-of-light result still moved 2.862 pixels. The remaining question is whether even the compact aperture follows asymmetric surrounding flux rather than the central point-source peak.

## Frozen estimator

Use the unchanged V19AH integer peaks and Gaia associations. Fit a general two-dimensional quadratic to only the 3-by-3 pixels centered on each peak. Accept its stationary point only if the fitted surface is concave and its offset from the frozen peak is no more than the unchanged 2-pixel ceiling. Do not clip an offset or choose another peak.

After locating the quadratic summit, measure background and second moments in the same 3-pixel core and 4.5-6-pixel annulus used by V19AJ. Apply the unchanged 1.5-12-pixel FWHM and 0.7 ellipticity gates. Thus the narrow estimator sets position while the broader core still checks whether the association resembles a usable star.

## Evidence gates

The science images, Gaia associations, minimum 30 stars per filter, minimum 20 shared Gaia IDs, all fitted and exact leave-one-out residual ceilings, center-consistency ceiling and R improvement condition are byte-for-byte identical to V19AI. No filter may be rescued by lowering its required count.

## Decision boundary

A full pass authorizes a later, separately frozen full-field mask/PSF stage. A failure ends this centroid-estimator branch and requires a new astrometric data or modeling strategy, not another aperture or relaxed gate.

No member/candidate payload, science photometry, baryonic mass/current model, lensing/halo payload, gravity parameter or holdout may be opened here.
