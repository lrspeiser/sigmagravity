# P0692 routing-continuum snapshot

Frozen before intermediate-fraction scores: 2026-08-02

## Bottom line

A 17-row, non-promotable atlas found one viable spent RX J2129 row at routing
fraction `f=0.30`. It recovers all `15/15` training and `7/7` heldout image
roots, scores `0.495/2.692 arcsec`, has no missing-multiplicity family, recovers
both parities and critical curves in all seven families, and is `1.0615x` the
object-specific compact-halo heldout RMS. Two families contain potentially
observable surplus images, exactly the frozen maximum.

This is a topology bifurcation, not a selected universal constant. Lower
fractions miss branches; higher fractions create too many branches and rapidly
worsen heldout positions. Because the row was exposed on spent data, it cannot
advance.

The next parameter-free generator is projected baryonic spectral anisotropy,
`e_2D=1-lambda_min/lambda_max`. RX J2129 gives `0.272023`, close to the viable
transition with exact circle/line limits and no fitted coefficient. It must be
frozen and tested on real 2D spent galaxy morphologies and RX J2129 before any
sealed target is opened.

## Public simulator path

Researchers will be able to call named real objects or seeded synthetic
galaxies/clusters, submit a safe unit-aware formula, and run either a frozen
test or a clearly labeled sweep. Vercel will host the UI and typed gateway;
asynchronous Cloud Run Jobs or Modal workers will run fields and lens roots.
Every response will include data, formula, seed, solver, parameter-accounting,
and comparator hashes.

See [`the public simulator/API plan`](../../docs/PUBLIC_SIMULATOR_API_PLAN.md).

## Canonical evidence

- [`P0692 continuum results`](../../docs/P0692_SPENT_LINEAR_ROUTING_CONTINUUM_RESULTS.md)
- [`P0691 geometry-gated result`](../../docs/P0691_MULTIPOLE_GATED_SOURCE_ROUTING_RESULTS.md)
- [`P0690 full-routing result`](../../docs/P0690_SOURCE_ROUTING_EMPIRICAL_SCREEN_RESULTS.md)
