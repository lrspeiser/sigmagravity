# P0682 spent-deflection research snapshot

Frozen: 2026-08-02

## Bottom line

The current absolute field equation still fails cluster image topology and is
not a replacement for dark matter or MOND. The next first-principles clue is
now sharper, however: in five non-boundary spent clusters, the field supplied
by a compact-halo comparator is overwhelmingly radial and aligned with the
baryonic deflection.

The median required radial halo/baryon ratio ranges from `6.12` to `11.64`.
Its geometric mean is `8.59`, with `0.127 dex` scatter (a factor `1.34`). All
five systems pass the frozen morphology rule; only three have sufficiently
reliable compact-halo targets for predictor selection. No baryonic predictor
therefore advances.

The result supports testing a simple cluster-regime radial amplification
before adding more angular machinery. It does not establish that `8.59` is a
constant of nature, and it does not solve image topology. P0683 has now tested
the first such field coefficient. Its one-setting, dimension-fixed QUMOND
channel exponent stays within 3.5% of fixed RAR on spent galaxies and closes
51% of the cluster radial gap, but misses both frozen cluster gates and does
not advance.

## Evidence table

| Stage | Outcome | Interpretation |
|---|---:|---|
| P0675 absolute raw topology | fail; `17.83 arcsec`, 7/7 missing multiplicity | The current tensor law is not a viable cluster lens. |
| P0678 RXJ2129 decomposition | halo/scalar RMS `3.317`; alignment `0.995` | Missing field is broad and radial on the registered 3D map. |
| P0679-P0681 derivative audit | refinement closed | Deflection conclusions survive; derivative-derived kappa/critical metrics remain provisional. |
| P0682 five-cluster morphology | 5/5 pass | Radial alignment repeats across spent systems. |
| P0682 constant amplitude | `8.59`, scatter factor `1.34` | A constant branch is provisionally simpler than an object predictor. |
| P0682 predictor selection | no survivor; 3 reliable targets | Sample and comparator reliability are insufficient. |
| P0683 potential-channel QUMOND | galaxies `1.035x` RAR; clusters `0.285 dex` | Clean regime onset, but wrong within-cluster ordering; no advancement. |

The P0678 and P0682 amplitudes use different baryonic maps and annuli. The
registered 3D RXJ2129 scalar RMS is roughly twice the Tian spherical-profile
value used in P0682. This map sensitivity is now an explicit next-stage test.

## Next scientific gates

1. Keep P0683's successful potential-depth onset and test a frozen,
   dimensionless path-dilution factor based on
   `eta=|Phi_b|/(r g_b)`, which is inversely ordered with the spent required
   cluster ratio. Treat the five-system correlation only as a formula
   generator.
2. Do not use observed image radii, target halo amplitudes, or per-cluster
   settings in the formula.
3. Require the resulting law to agree under the Tian spherical and registered 3D
   baryon reconstructions.
4. Run spent RXJ2129 topology. Require all families to recover multiplicity
   and parity, critical structure, heldout RMS below `3 arcsec`, and no fitted
   nuisance parameter at a boundary.
5. Repeat on three solver grids and fixed stellar/gas sensitivities.
6. Only a survivor may open P0633 galaxy kinematics and P0640 cluster lensing
   once, with one universal parameter vector, then run Solar tests.

## Researcher simulator and hosted API

The simulator will become a public test service after the formula schema,
solver fixtures, and validation contracts stabilize:

1. package real and seeded synthetic galaxies/clusters as versioned objects
   with units, provenance, licenses, and content hashes;
2. expose a safe, unit-aware formula language plus named-model catalog through
   FastAPI/OpenAPI and a Python SDK;
3. support calls to create a synthetic galaxy, load a named real system, submit
   a formula, run galaxy/cluster/Solar/topology suites, and retrieve immutable
   artifacts and comparator scores;
4. host the web interface and authenticated thin API gateway on Vercel;
5. dispatch heavy Poisson/AQUAL/QUMOND and lens-root jobs to versioned Cloud
   Run Jobs or Modal workers with object storage and strict resource limits;
6. reproduce local pass and failure fixture hashes in staging before public
   access; and
7. add arbitrary researcher code only later in a network-disabled sandbox.

The endpoint schemas, security boundary, deployment split, and launch criteria
are in
[`docs/PUBLIC_SIMULATOR_API_PLAN.md`](../../docs/PUBLIC_SIMULATOR_API_PLAN.md).

## Canonical evidence

- [`P0682 results`](../../docs/P0682_SPENT_MULTICLUSTER_DEFLECTION_ATLAS_RESULTS.md)
- [`P0683 results`](../../docs/P0683_POTENTIAL_CHANNEL_QUMOND_RESULTS.md)
- [`P0677 absolute-field snapshot`](../2026-08-02-p0677-absolute-field-audit/README.md)
- [`public simulator/API plan`](../../docs/PUBLIC_SIMULATOR_API_PLAN.md)
