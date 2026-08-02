# P0693 projected-spectral joint-screen snapshot

Frozen before scores: 2026-08-02

## Bottom line

The parameter-free projected baryonic controller succeeds on spent RX J2129
raw topology but fails the preregistered real-galaxy requirement.

RX J2129 calculates `e_2D=0.272023`, recovers every `15/15 + 7/7` observed
image root, scores `0.601/2.670 arcsec`, has no missing family, two allowed
surplus-image families, all parities and critical curves, and no near-bound
nuisance. Its heldout RMS is `1.053x` the object-specific compact halo.

DDO154 calculates `e_2D=0.083524` but scores `3.943 km/s`, `1.352x` algebraic
MOND and slightly worse than ordinary 3D QUMOND. The joint candidate therefore
fails and cannot advance.

The next frozen diagnostic maps the entire allowed DDO154 mixture continuum.
If no routing fraction closes the galaxy gap, the shared linear source pair is
retired and the next equation must change the galaxy endpoint itself while
preserving the cluster topology mechanism.

## Public simulator path

The hosted service will report each domain and gate separately, including
whether a result is frozen, diagnostic, spent, or sealed. Vercel remains the
UI and typed gateway; asynchronous Cloud Run Jobs or Modal workers run the
field and topology jobs.

See [`the public simulator/API plan`](../../docs/PUBLIC_SIMULATOR_API_PLAN.md).

## Canonical evidence

- [`P0693 joint result`](../../docs/P0693_PROJECTED_SPECTRAL_ROUTING_JOINT_RESULTS.md)
- [`P0692 continuum result`](../../docs/P0692_SPENT_LINEAR_ROUTING_CONTINUUM_RESULTS.md)
- [`P0635 real-map commissioning`](../../docs/P0635_REAL_2D_DDO154_COMMISSIONING.md)
