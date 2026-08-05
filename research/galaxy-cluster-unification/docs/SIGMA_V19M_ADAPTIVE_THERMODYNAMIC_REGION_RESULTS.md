# Sigma V19M adaptive thermodynamic region results

## Outcome

The independent V19H thermodynamic-region lane passed both its frozen machine
gate and its required post-hash visual audit:

| Cluster | Total contour bins | Admitted spectral regions | Admission fraction |
|---|---:|---:|---:|
| Bullet Cluster | 392 | 366 | 93.4% |
| Abell 2146 | 138 | 128 | 92.8% |

Every admitted region has broad-band signal-to-noise of at least 40, at least
1,300 net counts, and source fraction of at least 0.8.  The median admitted net
counts are 1,678 for Bullet and 1,666 for Abell 2146.  No scientific threshold
or cluster-specific setting was changed.

## Why this advances the causal measurement

V19J--V19L attempted to infer a shock curve by linking thresholded one-pixel
ridges.  That representation failed three different implementation gates.
V19M does not rescue it.  Instead it divides the complete two-dimensional
diffuse emission into connected, response-ready regions of comparable
statistical information.  Temperature and emission measure can later be fitted
inside every admitted region; thermodynamic discontinuities are then
comparisons between measured region posteriors rather than a brightness ridge
chosen by geometry.

The topology audit found that all 366 admitted Bullet regions and all 128
admitted Abell 2146 regions are single connected components.  Visual inspection
found feature-following central bins, broader low-surface-brightness outskirts,
the expected inherited point-source masks, and no contour-forest, straight-path,
checkerboard, disconnected or chip-edge artifact.

## What this does not yet prove

This is a region-identifiability result, not a temperature map or evidence for
new gravity.  No spectrum, ARF, RMF, temperature, density, Mach number, lensing
target or gravity parameter was constructed or opened.

The unexpectedly large admitted map--494 regions total--makes the next risk
computational rather than statistical.  The next frozen gate must enumerate
every nonempty observation/CCD/region response cell and estimate storage and
runtime before starting `specextract`.  It may parallelize independent cells,
but it may not merge regions, discard observations or raise the S/N target after
seeing these counts.

Authoritative artifacts are in
`results/sigma_v19m_adaptive_thermodynamic_regions/`, including the report,
content-addressed bin maps and region files, machine topology diagnostics, and
the visual audit.
