# Sigma v19G target-blind astrometry gate

## Outcome

All twenty cleaned Chandra observations in the Bullet Cluster/Abell 2146
causal-development pair passed the frozen Gaia DR3 astrometric gate.  The
registered source products now share an external ICRS frame without allowing
scale, rotation, or shear to absorb a future apparent-halo displacement or
shape.

This is a measurement result, not evidence for Sigma Gravity.  No registered
science image, shock morphology, replacement-cluster lensing coordinate,
inferred halo map, or gravity score was inspected.

## Frozen design

- The Gaia cones, reference observations, match radii, residual threshold, and
  transform family were fixed before either Gaia query or any cross-match.
- The reference observation for each cluster was selected only by greatest
  cleaned exposure: ObsID 5356 for the Bullet Cluster and ObsID 12247 for
  Abell 2146.
- Each reference was matched absolutely to Gaia DR3.  Every other observation
  was matched to its Gaia-corrected reference catalog.
- Every transform was translation only.  Rotation was fixed to zero, scale to
  one, and shear to zero.
- Every observation required at least three accepted pairs and a recomputed
  radial RMS no greater than 0.5 arcsec.
- No transform could be applied unless all twenty observations passed.

## Measured result

| Quantity | Result | Gate |
|---|---:|---:|
| Gaia sources, Bullet cone | 8,101 | nonempty |
| Gaia sources, Abell 2146 cone | 2,404 | nonempty |
| Chandra observations passed | 20/20 | 20/20 |
| Accepted source pairs | 239 | at least 60 total implied by per-observation gate |
| Fewest pairs in one observation | 3 | at least 3 |
| Best radial RMS | 0.1763 arcsec | no more than 0.5 arcsec |
| Worst radial RMS | 0.3499 arcsec | no more than 0.5 arcsec |
| Shape-changing degrees of freedom | 0 | 0 |

The limiting observation was Abell 2146 ObsID 10888 with exactly three pairs
and 0.1870-arcsec RMS.  The largest RMS was 0.3499 arcsec for Abell 2146 ObsID
12246.  Both passed without relaxing a threshold.  Corrected science and
matched blank-sky events were then generated for every observation.

## What this establishes

Future offsets among gas, galaxies, merger fronts, and a predicted metric
response can no longer be attributed to independently shifted Chandra
pointings at the half-arcsecond level.  It does not establish that any offset
is physical, that the merger clock is identifiable, or that baryons predict a
lensing halo.

The next mandatory boundary is to freeze one common source-map grid, one
automated edge likelihood, adaptive spectral and gas-uncertainty rules, and a
projection/clock ensemble before viewing a registered image.  Replacement
cluster lensing remains sealed until the causal source and its transfer rule
are fully frozen.

Machine-readable records:

- `configs/sigma_v19g_gaia_hierarchical_astrometry.json`
- `results/sigma_v19g_gaia_acquisition/provenance.json`
- `results/sigma_v19g_chandra_astrometry/report.json`
