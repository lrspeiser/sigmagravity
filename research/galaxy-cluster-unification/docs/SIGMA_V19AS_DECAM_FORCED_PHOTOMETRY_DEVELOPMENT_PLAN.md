# Sigma V19AS DECam forced-photometry development plan

## Question

Can one fixed image-level rule recover stable multiband photometry for the ten
predefined development galaxies, including crowded objects, without learning
anything from the five validation galaxies or from lensing and gravity
outcomes?

V19AS is deliberately narrower than a mass-map reconstruction. It tests the
measurement obstacle exposed by V19AD: catalog aperture colors disappeared in
crowded cases because all measurements in some bands carried deblending flags.

## Frozen comparison

Every one of the 670 development measurement memberships is processed. A
28-arcsecond image cutout is background-corrected with a clipped affine plane.
The detector-header FWHM fixes the smoothing scale. Positive-source islands are
split with a watershed whose target marker is the already-frozen sky
coordinate.

Two circular apertures are diagnostic: four and eight arcseconds in diameter.
Three exact flux rules are compared:

1. `raw`: ordinary aperture sum;
2. `area_scaled`: mask other watershed segments and correct for missing area;
3. `rotate180`: fill contaminated pixels from the point reflected through the
   target center, with area correction only when the reflected pixel is also
   unusable.

The point-reflection rule tests a concrete physical assumption—approximately
centrosymmetric early-type cluster galaxies—rather than allowing a free
profile per exposure. Its known risk is bias for asymmetric galaxies.

## Validation firewall

Only the ten development IDs can reach the measurement loop. The runner checks
the complete frozen split and aborts if a validation row enters that list. On
detectors that contain both splits, a four-arcsecond mask is applied at every
validation coordinate before background fitting or source detection.

The validation coordinates act only as exclusion masks. Their fluxes,
morphologies, residuals and colors are not measured.

## Selection rule

The recommended rule is chosen lexicographically by:

1. number of development objects with complete `griz`;
2. valid-measurement fraction;
3. repeated-exposure robust scatter;
4. leave-one-development-object-out Bessel-to-DECam color error;
5. four-arcsecond catalog agreement as a calibration diagnostic.

No validation, ambiguous-member, lensing, inferred-halo or gravity outcome can
enter the ranking. The winner must be frozen in a new protocol before the five
validation anchors are measured.

## Claim boundary

A pass would show that the image data support a reproducible deblended
measurement rule on the development set. It would not validate that rule,
identify ambiguous Bullet members, infer stellar masses, construct a current
map, or support the long-wavelength Sigma field.
