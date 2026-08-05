# Sigma V19AZ probabilistic member-position/current ensemble results

## Decision

V19AZ **passed every frozen gate**.  It produced an exact one-to-one positional
posterior for the 57 non-anchor Bullet Cluster members with published Bessel
BRI, retained the 15 validated anchors, kept all six missing-BRI members
explicit, and serialized 8,192 deterministic realizations of the 72-member
BRI position and line-of-sight-current proxy.

This is a baryonic-input uncertainty result.  No lensing observation,
dark-matter map, halo fit, gravity residual, or Sigma parameter was read.

## Exact assignment result

The 640 candidate hypotheses contain 568 unique catalog sources.  Sixty
candidate IDs occur for more than one spectroscopic member, but the conflict
graph separates into only six coupled components and 39 single-member
components.  The largest coupled component has four members.

V19AZ evaluated all 450,396 Cartesian component states and retained all
386,969 legal one-to-one states.  Every component was normalized by exact
enumeration and log-sum-exp; no Markov-chain approximation was used.

| Diagnostic | Result | Frozen gate |
|---|---:|---:|
| Maximum local normalization error | 0 | <= `1e-12` |
| Maximum component normalization error | `3.33e-16` | <= `1e-12` |
| Maximum member-marginal error | `4.44e-16` | <= `1e-12` |
| Maximum candidate occupancy | 0.92483 | <= 1 |
| Maximum local-to-joint probability change | `4.13e-15` | diagnostic |
| Probability an independent draw has no collision | `0.9999999999999954` | diagnostic |

The important finding is that **one-to-one candidate competition is not the
source of the remaining uncertainty**.  Although 60 candidate IDs are shared,
the high-probability states almost never try to assign the same object twice.
Conditioning on exclusivity changes no state probability at a scientifically
meaningful level.  The uncertainty is instead dominated by whether a member
has any catalog counterpart at all and by the coarse published coordinate
cell when it does not.

## What the positional posterior says

Among the 57 probabilistic members:

- the expected realization contains 38.74 catalog-candidate positions and
  18.26 private-null positions;
- 41 members have a catalog candidate as their most probable state and 16 have
  the private null as their most probable state;
- 11 members have null probability at least 0.90, including nine at least
  0.99;
- the median null probability is 0.0786;
- the median radial positional standard deviation is 1.474 arcsec, with range
  0.642 to 3.694 arcsec.

A null state does not remove a spectroscopic galaxy.  It samples that galaxy's
position uniformly inside the original rounding rectangle.  That distinction
allows later field calculations to retain its published light and velocity
without pretending that a noisy optical catalog object is certainly the same
source.

## Ensemble and current proxy

The final compressed ensemble contains 589,824 rows: 8,192 draws times 72
finite-BRI members.  Every draw is one-to-one and contains all 72 members.  The
maximum difference between a sampled state frequency and its exact marginal
is 0.01404, below the frozen 0.03 gate.

The nominal median cluster velocity is `cz = 88,884.5 km/s`.  Catalogued
redshift errors are redrawn in each realization.  Relative Bessel-I luminosity,
normalized to `I=20`, ranges from 0.895 to 36.31 with median 4.365.  Multiplying
that relative luminosity by the cluster-rest-frame line-of-sight velocity
provides the stored current proxy.

This is not an absolute mass map.  It contains no stellar mass-to-light ratio,
K correction, extinction model, or stellar-population fit.  It is also not a
three-dimensional current: both transverse velocity components remain
explicitly `unmeasured_not_imputed`.

## Reproducibility audit

The frozen runner was executed twice after deterministic gzip serialization
was added.  All seven scientific outputs were byte-for-byte identical,
including the 26.0 MB compressed ensemble and the rendered map.  The hashes
are recorded in
`results/sigma_v19az_probabilistic_member_current_ensemble/reproducibility_audit.json`.
The report timestamp is intentionally not part of that comparison.

## Consequence for the gravity program

V19AZ supplies the right positional uncertainty object for testing the
directional long-wavelength, baryonic-current, and frame-dragging-inspired
terms.  Those tests should be run over the ensemble rather than on one selected
counterpart catalog.

It also sharpens the remaining limitations:

1. candidate-collision logic cannot rescue or materially change this map;
2. the 11 high-null members need better source positions if sub-arcsecond
   member structure becomes decisive;
3. six spectroscopic members still lack published BRI;
4. a universal stellar-population or mass-to-light calibration is required
   before calling the luminosity map a stellar-mass map; and
5. a directional current term must marginalize transverse velocities instead
   of interpreting the observed line-of-sight component as a full vector.

The next cluster-source step is to combine this member ensemble with the
independently constructed gas density, temperature, and bulk-structure maps
after V19W response production completes.  Only then should a source-state
long-wave equation be frozen and exposed to the raw lensing targets.
