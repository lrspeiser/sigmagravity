# Sigma V19AM NSC SIA anchor-coverage results

## Decision

**Passed all frozen metadata-coverage gates.**  Every one of the 1,032 retained
V19AC measurements for the fifteen already-open V19AB anchors has exactly one
NSC DR2 SIA descriptor for the same exposure and band.  Every matched product
is a calibration-level-2 DECam `InstCal` image with FITS access.

The manifest SHA-256 is
`12a7dcb915753a42edbe0a2fafdcc77be19b66b6a90fd5250ca4a1d722ee7c8f`.

## Coverage anatomy

| Quantity | Result |
|---|---:|
| Approved anchors | 15: 10 development, 5 validation |
| Measurement/descriptor pairs | 1,032 / 1,032 |
| Unique exposures | 82 |
| Unique exposure-plus-detector-extension groups | 139 |
| `g`, `r`, `i`, `z`, `Y` rows | 196, 348, 208, 152, 128 |
| Development / validation rows | 670 / 362 |
| Missing or multiply matched measurements | 0 |
| Exposures rejected or ranked | 0 |

The SIA `access_estsize` metadata sum to 321,748,166 bytes, or 306.8 MiB, if
every 0.01-degree object-centered cutout were retrieved separately.  The same
measurements occupy only 139 exposure/extension groups.  A subsequent frozen
retrieval stage can therefore request one bounding cutout per group, with a
fixed sky margin, while retaining every measurement and avoiding redundant
downloads.

## What this establishes

The failed FORS1 strategy is no longer the only route to image-level anchor
photometry.  Independently calibrated DECam products exist and can be joined
losslessly to the exact NSC measurement records.  The raw VOTable response for
each anchor, each query URL and the complete exact-match manifest are preserved
and hashed.

This is not yet evidence that the pixels are scientifically adequate.  V19AM
did not download a FITS image, inspect a galaxy, estimate a PSF, choose an
exposure, fit a profile or compare a validation color.  It also did not open an
ambiguous counterpart, lensing/halo payload or gravity result.

## Authorized next stage

Freeze one deterministic bounding-box rule for each of the 139
exposure/extension groups, including an invariant sky margin and maximum
request size.  Then retrieve every group, verify HTTP completeness, FITS
integrity, WCS containment and exposure/filter identity, and stop before any
photometric model is selected.  Pixel-quality and profile-photometry gates must
be specified in a later protocol so the five validation anchors cannot tune
the method.
