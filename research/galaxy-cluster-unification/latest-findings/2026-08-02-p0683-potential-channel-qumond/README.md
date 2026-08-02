# P0683 potential-channel QUMOND snapshot

Frozen: 2026-08-02

## Bottom line

The project has isolated a repeated radial cluster requirement and tested the
first one-setting field coefficient designed to produce it. The equation is
promising enough to learn from, but it fails its frozen advancement gates and
has not earned a raw-topology or sealed-data run.

P0682 finds that the compact-halo comparator field is radially aligned with
the baryonic field in all five non-boundary spent clusters. Their median
radial halo/baryon ratios have geometric mean `8.59` and factor `1.34` scatter.

P0683 places a potential-dependent exponent inside one QUMOND source equation.
The dimension-fixed primary has one universal transition setting and no
per-object gravity parameter. It remains within `3.5%` of fixed RAR on 131
spent galaxies, passes the Solar force proxies, and reduces the all-five
cluster log error from `0.583` to `0.285 dex`. It closes `51.2%` of the gap,
short of the frozen `75%` rule, and misses the reliable-three cluster gate at
`0.309 dex`.

## What the failure teaches

Potential depth is a good galaxy/cluster separator but a bad sole
within-cluster controller. The selected equation underpredicts MACS0329,
MACS0429, and MACS1115, while overpredicting MACS1931 and RXJ2129. Increasing
one common amplitude cannot correct both groups.

A post-result audit identifies a more targeted baryonic coordinate:

\[
\eta={|\Phi_b|\over r g_b}.
\]

It counts, roughly, how many local radial-acceleration lengths are stored in
the potential. The required radial ratio is inversely ordered by `eta` across
the five non-boundary clusters (`Spearman rho=-0.90`, only five systems).
This is a formula generator, not evidence of a law.

The next test should keep the successful potential-depth onset but dilute the
extra channel exponent as `eta` grows. The clean primary candidate is a
dimension-fixed inverse-square-root path factor; alternative powers should be
diagnostic only. It must be frozen before scoring and cannot use lens-image
radii, target halo amplitude, or object labels.

## Concrete next gates

1. Run the frozen path-diluted QUMOND reconnaissance on the same 131 spent
   galaxies and six spent cluster radial targets.
2. Require `<=1.05x` fixed RAR galaxies, `<=0.20 dex` on both all-five and
   reliable-three clusters, `>=75%` gap closure, and the Solar force limits.
3. Only a pass may enter the registered 3D QUMOND solver.
4. Require spent RXJ2129 multiplicity, both parities, critical curves,
   heldout RMS below `3 arcsec`, and stability on three resolutions plus fixed
   baryon sensitivities.
5. Only then open P0633/P0640 once under one global parameter vector and run
   the metric/PPN Solar analysis.

## Public researcher simulator

Once the scientific kernel and validation contracts are stable, the same
runner will be offered through a safe typed API. Researchers will be able to
load named real galaxies/clusters, create seeded synthetic systems, submit
unit-aware formula expressions, launch immutable galaxy/cluster/Solar/topology
jobs, and compare results with Newton/GR, MOND/RAR, and compact-halo baselines.
The web interface and thin gateway can run on Vercel; heavy solvers belong in
versioned Cloud Run Jobs or Modal workers with content-addressed artifacts.

See [`the public API plan`](../../docs/PUBLIC_SIMULATOR_API_PLAN.md) for the
endpoint schemas, security boundary, deployment split, and launch criteria.

## Canonical evidence

- [`P0682 deflection atlas`](../../docs/P0682_SPENT_MULTICLUSTER_DEFLECTION_ATLAS_RESULTS.md)
- [`P0683 QUMOND reconnaissance`](../../docs/P0683_POTENTIAL_CHANNEL_QUMOND_RESULTS.md)
- [`P0682 snapshot`](../2026-08-02-p0682-spent-deflection-atlas/README.md)

