# Sigma V19BT blind-cluster source readiness

## Decision

Six of the eight V19BH metadata-only candidates have a direct public path to
both near-infrared HST imaging and Chandra data without opening a future
lensing target:

| Source-imaging state | Relaxed side | Disturbed side | Total |
|---|---:|---:|---:|
| Direct HST F160W plus Chandra | 3 | 3 | 6 |
| Reserve | 1 | 1 | 2 |

The six span nominal published `M500` values from `0.79` to
`5.7e14 solar masses`, a factor of 7.22. Every listed Chandra observation has
more than 1,000 counts inside `R500`.

All 28 whitelisted HST science-image URLs returned HTTP 200/206 at freeze
time. Their declared aggregate content length is 4,782,516,480 bytes; the
files were not downloaded while V19W was using the response-production host.

This is a **source-imaging preflight**, not selection of the final six and not
holdout admission. None yet has a complete, uncertainty-propagated model of
gas, BCG, intracluster light and member galaxies.

## Why two systems remain reserves

- **SDSS J1002+2031** has Chandra and published HST imaging, but it is absent
  from the direct SGAS HLSP image/model table and the source publication says
  the available evidence does not yield a well-constrained strong-lens model.
- **SDSS J1226+2149** has excellent Chandra depth but only F606W was used for
  its BCG/ICL measurement rather than the comparable F160W available for the
  other systems. It is also one member of the projected J1226+2149/J1226+2152
  pre-merger pair, requiring a separate deprojection and component-assignment
  audit.

Neither reserve is replaced after looking at a gravity result. The final six
remain unselected.

## Strict data boundary

The acquisition whitelist contains only MAST paths under `/images/v1/` and
the Chandra observation identifiers published in the source-side sample.
Kappa, gamma, deflection, magnification, critical-curve, bulk-model and raw
multiple-image products remain forbidden.

A temporary SGAS manuscript container was removed after it was found to bundle
allowed source metadata with forbidden coordinate tables. No image-coordinate
value was ingested or used, and no gravity formula was scored. Future source
acquisition must use the explicit product whitelist instead of a mixed paper
bundle.

## What is still needed before any cluster is source-ready

1. Download and hash only the whitelisted HST images and listed Chandra
   observations after the active V19W production job exits.
2. Construct common WCS, PSF, mask, sky and astrometric uncertainty products.
3. Infer stellar-mass, BCG/ICL and member-probability ensembles without using
   a target-fitted lens model for membership.
4. Infer gas surface density and line-of-sight depth with the same pipeline for
   every system.
5. Only then inspect family/image counts and positional-error metadata, without
   opening coordinate values. Freeze the final sample and sealed target hashes
   after all source and metadata gates pass.

The primary source-side references are the
[Chandra Strong Lens Sample](https://arxiv.org/abs/2511.12707v1) and the
[MAST SGAS archive](https://archive.stsci.edu/hlsp/sgas).

## Reproduction

```powershell
python scripts/check_sigma_v19bt_blind_cluster_source_readiness.py
python -m pytest tests/test_sigma_v19bt_blind_cluster_source_readiness.py -q
```

The machine-readable checkpoint is
`results/sigma_v19bt_blind_cluster_source_readiness/report.json`.
