# Sigma V19AC NSC per-measurement photometry results

## Result

V19AC passed every frozen acquisition gate.  All 226 requested NSC candidate
objects returned at least one measurement.  Ten HTTP-200 batches retained
8,199 exposure-level rows, their exact ADQL queries, and their exact encoded
request forms.  No aperture, exposure, source, or candidate was rejected.

| Quantity | Result |
|---|---:|
| Requested/returned unique objects | 226 / 226 |
| Measurement rows | 8,199 |
| DECam (`c4d`) rows | 8,006 |
| Mayall Mosaic-3 (`k4m`) rows | 59 |
| Bok 90Prime (`ksb`) rows | 134 |
| `g/r/i/z/Y` rows | 1,468 / 2,855 / 1,687 / 1,322 / 867 |

## Cluster-specific instrument finding

Joining the already-frozen V19AA cluster labels after acquisition shows:

| Cluster | Unique NSC objects | Rows | Instruments |
|---|---:|---:|---|
| Bullet | 167 | 8,006 | DECam only |
| Abell 2146 | 59 | 193 | Mosaic-3 and 90Prime |

This is important for the failed V19AB color test.  All Bullet colors can be
constructed with one camera's passbands; the apparent mismatch is not caused
by mixing DECam, Bok, and Mayall filters.  Abell 2146 must remain
instrument-separated in later photometric work.

Every Abell measurement has finite values for all four fixed apertures.  On
the Bullet side all 8,006 rows have finite `MAG_AUTO`, 7,999 have finite
4-arcsec-diameter aperture magnitudes, and 7,988 have finite 8-arcsec values.  The lower
level data are therefore sufficient to commission a same-aperture color
measurement.

## What this does and does not establish

V19AC only establishes data availability and provenance.  It does not show
that any candidate is the spectroscopic member and it does not rescue V19AB's
failed color-only gate.

The next protocol can freeze one aperture and one exposure-combination rule,
using the same ten development and five validation singleton IDs.  It must
again pass color-only retrieval before any of the 57 ambiguous Bullet cones is
scored.  A sensible first choice is the 4-arcsec-diameter aperture: it is several
times typical seeing, retains nearly complete coverage, and is less vulnerable
to neighboring light than the 8-inch aperture.  That choice must be committed
before validation is recomputed.

## Reproducibility

- Frozen protocol: `configs/sigma_v19ac_nsc_measurement_photometry.json`
- Runner: `scripts/download_sigma_v19ac_nsc_measurement_photometry.py`
- Raw batches: `data/raw/sigma_v19ac_nsc_measurement_photometry/`
- Combined measurements: `data/derived/sigma_v19ac_nsc_measurement_photometry/all_measurements.csv`
- Machine-readable report: `results/sigma_v19ac_nsc_measurement_photometry/provenance.json`
