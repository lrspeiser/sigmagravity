# Sigma V19AR current-Archive DECam cutouts: passed result

V19AR passed every preregistered image-transport and structural-integrity gate.

## Result

- Groups retrieved: **139/139**
- Frozen measurement memberships retained: **1,032/1,032**
- Original observations/current Archive files: **82/82**
- Raw selected-detector payloads: **139**
- Total raw bytes: **733,527,360**
- Structurally readable FITS images: **139/139**
- Celestial WCS: **139/139**
- Frozen anchors contained: **all anchors in all groups**
- Frozen detector identity reproduced: **139/139**
- Minimum finite-pixel fraction: **1.0**
- Smallest anchor-to-edge margin: **22.444585 pixels**
- Unique raw SHA-256 values recorded: **139/139**
- Embedded `CHECKSUM`/`DATASUM` keywords present: **0/139**
- Groups dropped or selected by science outcome: **0**

All downloads completed on their first attempt under the frozen V19AR plan.

## What was learned from the transport sequence

V19AN failed because stale NSC SIA descriptors returned server errors. V19AO
then exposed different index semantics between the Archive spatial-HDU result
and the FITS HDU selector. V19AP fixed that mapping but a non-stale SIA request
returned a header-only body. V19AQ removed SIA transport but preserved a corrupt
legacy `v1` processing file. V19AR applied one metadata-only current-processing
rule to every original observation and completed the sample.

The important methodological point is that no target was dropped after a
transport failure. Each repair changed a universal archive/structure rule,
froze it before the next pixel access, and retained all 1,032 memberships.

## Claim boundary

This is not a photometric or gravity result. It establishes that the full
calibrated detector sample needed for the next mass-current reconstruction is
present, hash-bound, structurally readable, and correctly located on the sky.

V19AR has not yet selected a PSF model, deblended cluster members, measured
fluxes or colors, inferred stellar masses, constructed mass-current maps, or
tested the long-wave Sigma field against lensing.

Machine-readable outputs:

- `results/sigma_v19ar_current_archive_decam_cutouts/report.json`
- `data/derived/sigma_v19ar_current_archive_decam_cutouts/download_manifest.csv`
