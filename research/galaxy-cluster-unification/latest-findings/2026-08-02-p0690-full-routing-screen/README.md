# P0690 full-routing snapshot

Frozen before scores: 2026-08-02

## Bottom line

Routing 100% of the positive generator is rejected. It produces useful
topological structure but far too much gravity: `0.871/1.089 dex` cluster
error, `23.81 arcsec` median physical deflection, and only `14/15` training and
`4/7` heldout exact roots.

The informative change is that all seven source families now recover both
parities and critical curves; four have exact multiplicity. Source placement
matters, but the routing strength must come from geometry rather than a fitted
fraction.

The registered baryonic map supplies that geometry through a normalized
quadrupole `q_b=0.11886`, with exact limits zero for a sphere and one for a
line. The next frozen candidate is

\[
S_{\rm mix}=(1-q_b)S_{\rm local}+q_bS_{\rm route}.
\]

It preserves the successful spherical local law and activates routing only in
proportion to measured baryonic nonsphericity.

## Hosted researcher model

The service plan remains Vercel for the UI and typed gateway, with Cloud Run
Jobs or Modal workers for real/synthetic field solves and lens-root searches.
Formula submissions will use a safe unit-aware expression language and return
immutable data, formula, solver, parameter-accounting, and comparator hashes.

See [`the public simulator/API plan`](../../docs/PUBLIC_SIMULATOR_API_PLAN.md).

## Canonical evidence

- [`P0690 results`](../../docs/P0690_SOURCE_ROUTING_EMPIRICAL_SCREEN_RESULTS.md)
- [`P0689 mathematical audit`](../../docs/P0689_SOURCE_CONSERVING_BARYONIC_ROUTING_AUDIT.md)
- [`P0685-P0686 local-field topology`](../../docs/P0685_P0686_LOCKED_PATH_QUMOND_RESULTS.md)
