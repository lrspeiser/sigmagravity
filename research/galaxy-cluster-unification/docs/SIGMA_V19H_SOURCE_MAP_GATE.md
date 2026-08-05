# Sigma v19H source-map gate

## Outcome

Both causal-development clusters passed the frozen common-map gate using all
twenty Gaia-registered Chandra observations.  The maps have enough coverage
and source counts to proceed to automated merger-front and thermodynamic
measurement without relaxing a threshold.

| Quantity | Bullet Cluster | Abell 2146 | Gate |
|---|---:|---:|---:|
| Observations included | 10/10 | 10/10 | all |
| Valid area inside 1,000 kpc | 100.000% | 99.9937% | at least 80% |
| Net 0.5-2.0 keV counts inside 1,000 kpc | 428,996.1 | 166,537.4 | at least 10,000 |
| Diffuse-centroid convergence | pass | pass | less than 0.1 output pixel |
| Frozen source-map files | 9 | 9 | complete |

The products include soft-band counts, exposure, scaled blank sky and
background variance; broad-band spectral-support equivalents; and the common
analysis mask.  Their frozen snapshots total 48,481,920 bytes.

This gate did not display a registered science image, search for an edge,
extract a spectrum, fit a temperature/density/Mach number, construct a
projection clock, open a replacement lensing target, or change a gravity
parameter.  It is data readiness, not evidence for the causal-assembly
hypothesis.

## Next decision

The frozen V19H algorithm must now find a statistically significant shock in
each cluster without hand-drawn sectors or published front coordinates, fit
the density and temperature jumps, identify at least two member phase
components, and propagate all quantities into the 4,096-draw projection/clock
ensemble.  Failure at any point is an identifiability failure and does not
authorize lowering a threshold.

The discrete member requirement subsequently failed in both clusters.  The
frozen BIC rule selected one phase component after 4,000/4,000 bootstraps
converged.  See `docs/SIGMA_V19H_MEMBER_PHASE_IDENTIFIABILITY_FAILURE.md`.
V19H therefore cannot pass its complete advance gate, although its map products
remain valid inputs to a separately frozen continuous-field successor.

Machine-readable records:

- `configs/sigma_v19h_causal_observable_protocol.json`
- `results/sigma_v19h_source_maps/report.json`
