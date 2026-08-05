# Sigma V19AQ all-Archive DECam cutout plan

## Why V19AQ exists

V19AP proved that the Archive selected-HDU route can recover stale detectors,
but the mixed plan still depended on the NSC SIA cutout service. At group 50,
that service reproducibly returned a header-only FITS response with no image
body. V19AQ removes that transport dependency for every group.

The target list is unchanged: 139 exposure-extension groups containing all
1,032 frozen measurement memberships.

## Frozen file-identity rule

For each of the 82 exposures:

1. Query the Archive at a representative frozen coordinate using the exact
   frozen exposure name.
2. If one exact `.fits` or `.fits.fz` basename exists, require matching DECam,
   InstCal, image, object, and filter metadata and keep it.
3. If a stale `c4d` basename is absent, remove only its terminal processing
   token and choose the unique most recently updated valid calibrated image
   having that complete prefix.
4. If a renamed `tu...` basename is absent, derive the observation timestamp
   from its unique frozen SIA association ID, require the matching-filter `ooi`
   science product, and choose the unique most recently updated valid row.

Both the exact query and any fallback query are persisted. No pixel, PSF,
morphology, lensing, halo, or gravity value participates in identity selection.

## Frozen detector rule

For every group, read the selected file's complete public header list and
require exactly one two-dimensional celestial WCS to contain every frozen
anchor. Freeze a primary-plus-selected-HDU Archive URL from that independently
resolved FITS index. No arithmetic offset from Archive metadata is assumed.

## Metadata-only freeze result

- Groups: **139**
- Memberships: **1,032**
- Unique exposures: **82**
- Unique selected Archive files: **82**
- Unique selected detector URLs: **139**
- Exact-basename exposures: **14** (25 groups)
- Stale-`c4d` prefix fallbacks: **22** (37 groups)
- Frozen-association `tu...` fallbacks: **46** (77 groups)
- Unique all-anchor detector WCS: **139/139**
- New V19AQ image pixels opened before freeze: **0**

Frozen resolved-plan SHA-256:

`99661e740b98b3a728a0952ded3620ca1c688900994fec809f8cd1dccd1bd31e`

## Retrieval and integrity gates

Each frozen URL receives at most three attempts. HTTP errors and structurally
invalid payloads consume the same retry budget, and an invalid payload is never
persisted. A persisted detector must have a valid FITS structure, celestial
WCS, finite pixels, all frozen anchors, and the exact frozen CCD identity.

Archive subset products may omit embedded `CHECKSUM`/`DATASUM` keywords. Every
raw payload receives a SHA-256. If either FITS checksum keyword is present, it
must validate; absence is reported separately and is not misrepresented as a
successful embedded checksum.

A pass establishes image transport and structural integrity only. It does not
establish photometry, color, stellar mass, mass current, lensing, or any gravity
theory result.
