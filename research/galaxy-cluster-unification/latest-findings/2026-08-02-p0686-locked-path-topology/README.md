# P0686 locked path-QUMOND snapshot

Frozen before scores: 2026-08-02

## Bottom line

The promising radial generator survives a real 3D field solve but fails raw
multiple-image topology. This is useful falsification: the equation produces
enough total bending, but puts too much of the new response at large radius and
too little near the center.

The locked field has no RX J2129-specific gravity parameter and no fitted
photon amplitude. It solves to a `2.66e-14` normalized residual, produces
`9.96 arcsec` strong-lens RMS physical deflection, is `3.336x` the scalar-AQUAL
field, and has negligible numerical curl.

The raw test then recovers only `14/15` training and `6/7` spent-heldout exact
roots. Three of seven families are missing images, five of seven recover both
parities, and both external-shear nuisances hit their hard bounds. Positional
RMS is therefore infinite. Critical curves exist for all seven families, so
the result is stronger than earlier one-root/no-critical-curve failures but it
does not advance.

## New constraint learned

The local coordinate

\[
\eta_b={|\Phi_b|\over r g_b}
\]

diverges toward the center of an extended core and suppresses the extra
channels there. In this solve the median exponent rises from `1.28` inside
15 kpc to `2.71` at 150-225 kpc; physical deflection rises from `4.27` to
`20.11 arcsec`. Annular amplitude alone did not expose that hollow topology.

The next spent-data test replaces the local coordinate with one automatically
derived baryonic system coordinate evaluated where `r g_b(r)` peaks. It adds
no per-object fitted setting and avoids inserting an RX J2129-specific core
radius.

## Hosted researcher model

The deployment target remains a Vercel interface and typed API gateway backed
by isolated asynchronous field-solver workers. Researchers will be able to
load named real systems or reproducibly generate seeded synthetic galaxies and
clusters, submit safe unit-aware formulas, and receive immutable results with
MOND, Newtonian, and compact-halo comparator accounting.

See [`the public simulator/API plan`](../../docs/PUBLIC_SIMULATOR_API_PLAN.md).

## Canonical evidence

- [`P0685-P0686 results`](../../docs/P0685_P0686_LOCKED_PATH_QUMOND_RESULTS.md)
- [`P0684 radial generator`](../../docs/P0684_PATH_DILUTED_QUMOND_RESULTS.md)
- [`P0682 cluster deflection atlas`](../../docs/P0682_SPENT_MULTICLUSTER_DEFLECTION_ATLAS_RESULTS.md)
