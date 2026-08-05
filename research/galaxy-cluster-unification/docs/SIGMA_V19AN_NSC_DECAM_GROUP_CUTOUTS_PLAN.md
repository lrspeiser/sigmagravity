# Sigma V19AN NSC DECam grouped-cutout plan

## Decision being tested

V19AM proved that every approved anchor measurement has one calibrated DECam
descriptor.  V19AN asks the next narrower question: can every required image
group be retrieved as a valid FITS image whose celestial WCS contains all of
its frozen anchors?

This is acquisition and structural validation only.  It cannot select a PSF,
judge a galaxy profile, compare a validation color or infer a stellar mass.

## Frozen grouping and footprint

The exact group key is `(exposure, detector extension)`.  All 1,032
measurement memberships collapse losslessly to 139 groups across 82 unique
exposures.

For each group:

1. Take the coordinate midpoint of the minimum and maximum frozen anchor RA
   and declination.
2. Add a fixed 0.01-degree (36-arcsecond) true-sky margin on every side.  The RA
   coordinate margin is divided by the cosine of the center declination.
3. Preserve the exact NSC image reference and detector extension while
   replacing only the SIA `POS` and `SIZE` fields.
4. Retrieve every group.  No seeing, flag, band, date, split or photometric
   value can rank or omit an image.

The resulting manifest was built before pixel access and frozen at SHA-256
`a4b82c98ecab29b4ab2174619ca50f88399fafddc427ab40cf997b42e231df22`.
Its largest coordinate request is 0.21762866 by 0.11861111 degrees, below the
frozen 0.25 by 0.15-degree ceilings.

## Gates

The stage passes only if all 139 groups:

- return HTTP 200;
- open as structurally valid FITS with a two-dimensional image;
- contain at least one finite pixel and a celestial WCS;
- contain every anchor assigned to that exact exposure/extension; and
- remain in the hashed download manifest.

The integrity check may read pixels only to establish array structure and the
finite-value fraction.  It may not evaluate morphology, residuals, flux,
background, PSF quality or validation performance.

## Claim boundary

A pass authorizes a separately frozen development-only photometry method.  It
does not show that the images are adequate for that method.  A retrieval or WCS
failure stops the corresponding group from scientific use but does not permit
replacement with a hand-selected exposure.
