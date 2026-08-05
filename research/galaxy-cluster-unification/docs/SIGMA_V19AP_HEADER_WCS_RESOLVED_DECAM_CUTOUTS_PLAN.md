# Sigma V19AP header-WCS-resolved DECam cutout plan

## Purpose

V19AP repairs the transport-index ambiguity that stopped V19AO. It preserves
all 139 already-frozen exposure-extension groups and all 1,032 measurement
memberships. It does not choose or reject an exposure based on pixels.

## Frozen resolution rule

For each of the 37 groups whose stale NSC descriptor requires Archive
fallback:

1. Keep the V19AO current Archive file identity and MD5 fixed.
2. Fetch the file's complete public FITS header list from the Archive header
   endpoint, without retrieving new image pixels.
3. Convert each two-dimensional celestial header into a WCS.
4. Require exactly one FITS HDU WCS to contain every frozen NSC anchor in the
   group, with the predeclared one-pixel tolerance.
5. Freeze the selected primary-plus-HDU retrieval URL.

The procedure does not assume that the correct FITS HDU is the `vohdu` value
plus one. Empirically, all 37 resolved groups have an offset of one, but that is
an outcome of the independent WCS-containment rule, not an input to it.

The other 102 groups retain their exact frozen NSC SIA cutout URLs.

## Metadata-only freeze result

- Groups: **139**
- Measurement memberships: **1,032**
- NSC SIA groups: **102**
- Archive header-WCS groups: **37**
- Unique current Archive files/header payloads: **22**
- Unique corrected Archive retrieval URLs: **37**
- Archive groups with exactly one all-anchor WCS: **37/37**
- New image pixels opened during V19AP planning: **0**

Frozen resolved-plan SHA-256:

`f267d200b0a95c03b16eaeedbfa2957034a4332c51fb84b5db8dda4135530193`

## Acquisition gates

The runner must stop on the first failure. Every group must return a valid FITS
image with celestial WCS, at least one finite pixel, and every frozen anchor
inside the returned footprint. Archive subsets must also return the exact CCD
number and extension name frozen from the full-header WCS. No group may be
dropped or substituted.

Passing V19AP will establish transport completeness and structural integrity
only. It will not establish PSF quality, deblending, flux, color, stellar mass,
mass current, lensing, or any gravity result.
