# Sigma V19AO resilient DECam cutouts: failed-closed result

V19AO did **not** pass. It accepted zero image groups and stopped at the first
predeclared structural gate.

## What failed

The first stale NSC group was mapped by the Archive `vohdu` response to
`hdu_idx=34`. The corresponding public selected-HDU URL returned a valid FITS
payload containing CCD 34 (`N3`), but none of the group's 14 frozen anchors
fell inside that detector's celestial WCS. The runner therefore stopped before
writing a completed manifest or interpreting any photometry.

## Root cause

A metadata-only read of the same Archive file's complete header list found one
and only one detector containing all 14 anchors: FITS HDU 35, CCD 35 (`N4`).
For this file, the `vohdu` result index cannot be passed unchanged to the
Archive `retrieve?...hdus=` selector. V19AO assumed those index semantics were
identical even though it did not assume a stale-to-current detector-number
offset.

This is a transport-protocol failure, not evidence against an exposure, a
galaxy, or a gravity formula. No target may be dropped because of it.

## Required repair

The successor protocol must use each current file's full Archive header list
and select a FITS HDU only when its WCS uniquely contains every frozen anchor
assigned to the group. This decision uses headers and coordinates only. It
must not use pixel quality, flux, PSF, morphology, lensing, halo, or gravity
results. All corrected URLs must be frozen before new pixel retrieval begins.

The machine-readable failure record is
`results/sigma_v19ao_resilient_decam_cutouts/report.json`.
