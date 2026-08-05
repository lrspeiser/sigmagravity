# Sigma V19AE FORS1 commissioning-frame acquisition plan

## Decision

Acquire the original public ESO FORS1 Bessel `B/R/I` commissioning frames and
a frozen contemporaneous bias/twilight-flat pool without opening any FITS
payload.  This is the cleanest available route around the catalog deblending
failure because it returns to the same filters and images from which the
published Bullet member magnitudes were measured.

## Why this is a separate acquisition gate

The V19AD clean-aperture test failed because Bullet member 57 is blended in
the NSC catalog.  DELVE independently identifies the same object but assigns
SExtractor flag 3 in every band, and DECaPS has no coverage.  Relaxing the
flag after seeing that result would be retrospective.

The primary paper states that its photometry came from FORS1 commissioning
images taken in December 1998.  ESO TAP metadata identifies the frames under
programs `60.A-9203(A/B)`.  Seven archive-category science images cover the
field in the exact Bessel filters: two `B`, three `R`, and two `I` frames.
Their aggregate archive size is about 45 MB.

ESO CalSelector cannot build an association for a representative image in
either raw-to-master or raw-to-raw mode; it reports `UNKNOWN_CATEGORY` and
`complete=false`.  This is not surprising for commissioning data.  A metadata
inventory nevertheless finds a deterministic manual pool of 14 full-field
bias frames and 25 Bessel `B/R/I` twilight flats.  V19AE acquires that pool but
does not assume it is valid.

## Frozen boundary

Before freeze, only tabular archive metadata and calibration-association XML
were read.  No FITS file was downloaded or opened.  The protocol freezes all
46 dataset identifiers and fails if the archive query returns a different
set.

The acquisition runner may:

- execute the three exact archive queries;
- verify their exact schemas and dataset identifiers;
- download the exact compressed payloads atomically;
- verify HTTP content lengths when supplied; and
- hash every query, request, metadata response and file.

It may not open a FITS header or pixel array, prefer a science exposure using
image content, evaluate a member, choose a counterpart, fit a flux, infer a
mass/current, read a lensing/halo result, or alter gravity physics.

## What must be frozen next

After acquisition, a new protocol must specify—before any FITS payload is
opened—the header-compatibility rules, overscan/bias method, flat construction,
cosmic-ray handling, astrometric reference, PSF model, joint multiband source
profiles, background model, treatment of neighbor overlap, uncertainty
propagation, development/validation split and failure thresholds.

The attractive feature is that a successful image-level association no longer
needs a Bessel-to-NSC filter transformation.  The published member colors and
the candidate fluxes would be measured in the same passbands.  That advantage
is only real if the reduction and deblending rules are fixed before the member
pixels are examined.

## Reproduction after freeze

```powershell
python scripts/download_sigma_v19ae_fors1_commissioning_frames.py
python -m pytest -q tests/test_sigma_v19ae_fors1_commissioning_frames.py
```
