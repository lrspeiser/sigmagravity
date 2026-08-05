# Sigma V19AF FORS1 header-compatibility plan

## Purpose

V19AF is the narrow bridge between lossless acquisition and pixel calibration.
It determines whether all 46 frozen science, bias and flat files have one
compatible detector/readout configuration without interpreting any pixel.

This stage is necessary because ESO CalSelector does not recognize the 1998
commissioning frames.  Manually selected calibration frames must not be
combined merely because their dates and filter labels look plausible.

## Frozen operation

Each V19AE `.fits.Z` hash is reverified.  The file is decompressed inside a
temporary directory with GNU `gzip -dc`, the full decompressed byte stream is
hashed, and only complete 2880-byte primary-header blocks through the first
`END` card are parsed.  The decompressed file is then deleted automatically.

The runner records every primary card and derives a detector signature from
the image dimensions plus a predetermined list of ESO detector, window,
binning, output, gain and readout keywords.  A candidate keyword is ignored
only when absent from every header.  It is a failure if the keyword exists in
only some files, or if any active detector-signature value differs across the
science, bias and flat roles.

The runner also verifies that every file is a simple two-dimensional primary
image and is long enough to contain the padded data size implied by `BITPIX`
and `NAXIS1/2`.  This check uses file lengths, not pixel values.

## Claim boundary

A pass means only that the manual calibration pool is structurally compatible
at header level.  It says nothing yet about bias stability, flat uniformity,
bad pixels, saturation, cosmic rays, astrometry, sky subtraction, source
profiles or photometric accuracy.

The next global-pixel protocol must be written after this header inventory but
before any data array is requested.  No member coordinate or candidate cutout
can be used in that calibration-readiness stage.

## Reproduction after freeze

```powershell
python scripts/run_sigma_v19af_fors1_header_compatibility.py
python -m pytest -q tests/test_sigma_v19af_fors1_header_compatibility.py
```
