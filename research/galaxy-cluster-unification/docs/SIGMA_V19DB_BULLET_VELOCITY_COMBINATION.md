# Sigma V19DB Bullet velocity-region combination

## Payload-blind freeze

V19DB is frozen after the committed V19DA source-only preflight and before any
new PHA channel, response array, line centroid, temperature, abundance,
redshift or velocity is opened. Its payload-blind execution-plan preflight
passes with:

| Item | Frozen value |
|---|---:|
| Bullet primary regions | 43 |
| Source-only V19M bins | 366 |
| Primary response cells | 3,483 |
| Primary observations | 9 VFAINT ObsIDs |
| Abell 2146 access | sealed |
| ObsID 554 in the primary branch | excluded |

The exact runner SHA-256 is
`aeb92be4afd2e399c5a474b5fa5c5fcc930ba3e9917e1e13e7b0ec3e8e455149`.
The plan was produced entirely from the committed region manifest and response
indexes. It did not open a source/background PHA or ARF/RMF scientific array.

## Frozen combination

Every region uses the V19CW-commissioned response topology:

1. combine source and background PHAs within each ObsID using ASCA background
   scaling and PHA exposure accounting;
2. combine the corresponding ARFs and RMFs within that ObsID with zero
   intermediate RMF threshold;
3. combine the nine observation-level products;
4. apply the frozen `1e-6` RMF sparsification once, at the final level;
5. retain the final source PHA ungrouped for later WStat analysis.

Every one of the 3,483 primary response cells must occur exactly once. Full-PHA
source counts must be conserved exactly in each region, and the final source
PHA must point to the frozen background, ARF and RMF in the same directory.

Before the other 42 regions are accepted, the lowest deterministic group ID is
also combined directly. Flat, power-law, thermal and Fe-line proxy spectra are
forward-folded through the direct and hierarchical responses. Every relative
L1 difference must be at most `1e-8`.

## Claim boundary

This stage verifies deterministic response-aware spectral assembly. It does
not fit or inspect a temperature, abundance, redshift or velocity. It cannot
admit signed gas current, select a Sigma source, change a gravity equation or
authorize Abell 2146 access. Only a complete combination pass authorizes the
already-frozen Bullet spectral reproduction.
