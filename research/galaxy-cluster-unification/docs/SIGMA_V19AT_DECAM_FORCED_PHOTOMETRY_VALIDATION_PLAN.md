# Sigma V19AT DECam forced-photometry validation plan

## Frozen question

Does the single rule recommended by V19AS transfer to the five sealed
singleton galaxies, including member 57, whose catalog `r` and `i`
four-arcsecond measurements were all rejected by the earlier deblending flag?

This is a once-only validation. No alternative aperture or variant may be
inspected after the result.

## Measurement frozen from development

- variant: watershed-mask plus area scaling;
- aperture diameter: 4 arcseconds;
- cutout: 28 arcseconds;
- background annulus: 8--13 arcseconds with a clipped affine plane;
- detection threshold: 2.5 times the smoothed background noise;
- exposure selection: none;
- validation memberships retained: 362 across 110 detector groups.

The implementation calls the exact hash-bound V19AS measurement functions.

## Color calibration boundary

The current DECam image headers provide repeatable Community Pipeline
characterization magnitudes, but not a final native-band AB calibration. At
this southern declination the pipeline documentation says the reference is
Gaia G regardless of DECam filter. V19AS accordingly found stable but large
filter offsets relative to the later NSC calibration.

V19AT does not repair those offsets with validation data. It fits three affine
color transformations (`g-r`, `r-i`, `i-z`) on the ten development objects
only, then applies those frozen transformations to validation.

## Gates

V19AT passes only if:

1. all five validation objects, and member 57 specifically, have complete
   `griz` image measurements;
2. every color has median absolute validation error at most 0.25 mag;
3. at least three of five Bessel-color predictions retrieve their true DECam
   partner at rank one;
4. mean reciprocal rank is at least 0.65; and
5. all 362 memberships remain in the result.

Failure closes this exact route. It cannot be rescued by looking at the raw or
rotate-180 validation outcomes, a different aperture, a changed threshold, or
a validation-derived zeropoint.

## Claim boundary

A pass would validate a color-consistency measurement for the five provisional
singleton pairs and authorize a new, separately frozen ambiguous-candidate
likelihood. It would not establish absolute photometry, identify a candidate,
infer stellar mass, construct a current map, or test Sigma Gravity.
