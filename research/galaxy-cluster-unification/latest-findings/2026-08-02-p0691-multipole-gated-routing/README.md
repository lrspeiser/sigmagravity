# P0691 multipole-gated routing snapshot

Frozen before scores: 2026-08-02

## Bottom line

The baryonic map uniquely sets a routing fraction of `q_b=0.118863`, and the
resulting 3D field is numerically excellent: residual `3.20e-14`, negligible
curl, and `10.39 arcsec` median physical deflection. It nevertheless fails the
raw lens test with only `13/15` training and `5/7` heldout roots, three missing-
multiplicity families, and parity diversity in only five of seven families.

This rejects one global shape number as the routing controller. It does not
show that all geometric source routing is impossible. The next frozen step is
a diagnostic continuum atlas between the local and fully routed sources. No
fraction from that spent atlas may be promoted; it is only a way to decide
whether the entire linear family should be retired or whether a future local
geometric quantity has a specific behavior to predict.

## Public simulator path

The public version will let researchers select real catalog systems or create
seeded synthetic galaxies/clusters, submit safe unit-aware formulas, launch
asynchronous field and topology jobs, and compare immutable results with
Newtonian, MOND/RAR, GR-baryon, and declared halo baselines. Vercel is planned
for the interface and typed gateway, with Cloud Run Jobs or Modal for the
scientific workers.

See [`the public simulator/API plan`](../../docs/PUBLIC_SIMULATOR_API_PLAN.md).

## Canonical evidence

- [`P0691 results`](../../docs/P0691_MULTIPOLE_GATED_SOURCE_ROUTING_RESULTS.md)
- [`P0690 full-routing result`](../../docs/P0690_SOURCE_ROUTING_EMPIRICAL_SCREEN_RESULTS.md)
- [`P0689 mathematical audit`](../../docs/P0689_SOURCE_CONSERVING_BARYONIC_ROUTING_AUDIT.md)
