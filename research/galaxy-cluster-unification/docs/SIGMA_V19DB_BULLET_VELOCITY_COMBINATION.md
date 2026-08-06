# Sigma V19DB Bullet velocity-region combination

## Outcome

The frozen Bullet primary combination passes every registered gate. All 43
regions now have an ungrouped source PHA, associated blank-sky background, ARF
and RMF suitable for the already-frozen two-temperature reproduction.

| Audit | Result |
|---|---:|
| Regions combined | 43 / 43 |
| Source-only bins used exactly once | 366 / 366 |
| Primary response cells used exactly once | 3,483 / 3,483 |
| Full-PHA source counts, expected / combined | 674,283 / 674,283 |
| Regions with exact source-count conservation | 43 / 43 |
| Regions with exact PHA response links | 43 / 43 |
| Frozen response-aware products | 172 |
| Frozen product bytes | 103,066,560 |

The direct-versus-hierarchical pilot passed. Its largest relative L1
forward-fold difference was `5.645841956611248e-9`, below the frozen `1e-8`
gate. The Fe-line proxy difference was `3.966974953036414e-9`. Source counts,
the source exposure, and the ARF/RMF grids also passed their exact or frozen
tolerance gates.

The terminal report SHA-256 is
`25104d0a4f7840e3c25b7b2eda99ff6186c9ef3b668af8695289d422a984e4aa`.
This pass authorizes the Bullet spectral reproduction; it is not a velocity or
gravity result.

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

This stage verified deterministic response-aware spectral assembly. It did
not fit or inspect a temperature, abundance, redshift or velocity. It cannot
admit signed gas current, select a Sigma source, change a gravity equation or
authorize Abell 2146 access. The complete combination pass authorizes only the
already-frozen Bullet spectral reproduction.
