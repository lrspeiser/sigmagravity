# P0631 observation-matched galaxy replicas

**Replica gate: PASS**

- Galaxies: 131 (81 train / 27 development / 23 holdout).
- Median angular-photometry reconstruction: 0.000031 dex.
- Median continuous rotation reconstruction: 0.000000 km/s.
- Median finite-grid rotation loss: 0.221 km/s.
- Median absolute total-light integration error: 0.37%.
- Deterministic replay: True.

## Meaning

This establishes that the simulator can generate an axisymmetric galaxy whose radial 3.6 μm light profile, projected inclination, and velocity field reproduce the supplied SPARC observables. Rotation is supplied in replica mode, so this is a reconstruction test—not a successful gravity prediction.

For a gravity test, the identical light seed is retained but the observed velocity is removed. `render_replica` then requires an explicit theory-predicted circular-speed curve. The theory is scored only against the hidden observed curve.

## Current observational limit

The downloaded SPARC products contain radial profiles rather than raw two-dimensional 3.6 μm and H I images. The simulator therefore does not yet claim to reproduce observed bars, spiral arms, warps, gas clumps, or lopsidedness. Resolved survey cutouts and velocity cubes are the next data layer for that test.
