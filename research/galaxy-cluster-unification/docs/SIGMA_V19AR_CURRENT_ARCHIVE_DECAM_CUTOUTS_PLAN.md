# Sigma V19AR current-Archive DECam cutout plan

## Why V19AR exists

V19AQ froze exact legacy filenames when they existed. That policy stopped at a
2020 `v1` product whose selected detector reproducibly failed decompression.
V19AR replaces filename loyalty with one uniform processing-version policy for
all 82 original observations.

The target evidence remains unchanged: 139 groups and all 1,032 frozen
measurement memberships.

## Frozen current-product rule

For a `c4d` exposure, remove only its terminal processing-version token and
query the complete calibrated `ooi`/filter prefix. For a renamed `tu...`
exposure, derive the archive timestamp from its unique frozen SIA association
ID and require the matching-filter `ooi` product.

All candidate rows must be DECam InstCal object images and must share exactly
one `original_filename`, proving that they are processing versions of the same
original observation. Select the unique row with the greatest Archive
`file_updated` timestamp.

This metadata rule is applied to every exposure. It does not inspect whether a
target is bright, isolated, round, well-lensed, or favorable to any gravity
model.

## Frozen detector rule

Read the selected current file's complete public header list and require
exactly one two-dimensional celestial WCS to contain every frozen anchor in
the group. Freeze the primary-plus-selected-HDU retrieval URL from that FITS
index. No detector-number offset is assumed.

## Metadata-only freeze result

- Groups: **139**
- Memberships: **1,032**
- Original observations: **82**
- Unique current Archive files: **82**
- Unique selected-detector URLs: **139**
- `c4d` observation identities: **36 exposures / 62 groups**
- Association-derived `tu...` identities: **46 exposures / 77 groups**
- Legacy `v1` products replaced by current processing: **14 exposures / 25 groups**
- Unique all-anchor detector WCS: **139/139**
- New V19AR image pixels opened before freeze: **0**

Frozen resolved-plan SHA-256:

`2cf636171321f3865387e5c1026bde32c0d9b7bddc1971bbbff70f2b79f93871`

## Retrieval and integrity gates

Each URL has a fixed three-attempt budget. Network, FITS, decompression, WCS,
detector, finite-pixel, containment, and present-checksum failures all consume
that same budget; invalid payloads are never persisted.

Every exact raw payload receives a SHA-256. Embedded FITS `CHECKSUM` or
`DATASUM` keywords may be absent from Archive subset reconstructions, but any
that are present must validate.

Passing V19AR establishes complete image transport and structural integrity
only. It does not establish PSF quality, photometry, color, stellar mass, mass
current, lensing, or a Sigma-field prediction.
