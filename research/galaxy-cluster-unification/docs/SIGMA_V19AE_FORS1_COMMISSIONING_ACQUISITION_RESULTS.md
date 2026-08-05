# Sigma V19AE FORS1 commissioning-frame acquisition results

## Decision

V19AE passed every frozen lossless-acquisition gate.  The original ESO FORS1
commissioning science frames and the predetermined manual bias/flat pool are
now local, hashed and provenance-tracked.  No FITS header or pixel array was
opened, and no photometry or association model is yet authorized.

## Acquired payload

| Role | Files |
|---|---:|
| Archive-category Bessel `B/R/I` science frames | 7 |
| Full-field 1x1 bias frames | 14 |
| Bessel `B/R/I` twilight flats | 25 |
| **Total** | **46** |

The exact downloaded volume is **278,666,181 bytes**.  The science set contains
two `B_BESS`, three `R_BESS`, and two `I_BESS` exposures.  Every dataset ID was
frozen before download and the live TAP response reproduced the exact set.

## Gates

- all three archive metadata queries returned HTTP 200;
- all 46 downloads returned HTTP 200;
- every response length matched its HTTP `Content-Length` when present;
- every metadata CSV, ADQL query, request form and downloaded payload is
  SHA-256 hashed;
- the local filenames use a portable underscore encoding for the colons in ESO
  identifiers, while the literal archive `dp_id` remains unchanged in all
  metadata and provenance; and
- no FITS payload, source, member, candidate, flux, counterpart, mass, current,
  lensing, halo or gravity result was opened or produced.

The first execution stopped before creating any FITS payload because Windows
forbids literal colons in filenames.  An implementation-only correction
encoded nonportable filename punctuation and changed no archive identifier,
query, calibration choice, gate or scientific authorization.

## Meaning

This clears the acquisition bottleneck but not the photometry bottleneck.  ESO
CalSelector cannot associate calibrations to these commissioning files, so the
manual bias/flat pool must still pass a preregistered header and image-quality
audit.  The files must not be treated as calibrated merely because they were
downloaded successfully.

The next protocol must be frozen before the compressed FITS payloads are
opened.  It must define detector/header compatibility, decompression, bias and
flat construction, bad-pixel/cosmic-ray treatment, astrometric registration,
PSF and galaxy profiles, multiband neighbor deblending, background covariance,
validation splits and failure thresholds.

## Reproducibility

- Frozen protocol: `configs/sigma_v19ae_fors1_commissioning_frames.json`
- Acquisition runner: `scripts/download_sigma_v19ae_fors1_commissioning_frames.py`
- Raw files and exact query records: `data/raw/sigma_v19ae_fors1_commissioning_frames/`
- Machine-readable report: `results/sigma_v19ae_fors1_commissioning_frames/provenance.json`
