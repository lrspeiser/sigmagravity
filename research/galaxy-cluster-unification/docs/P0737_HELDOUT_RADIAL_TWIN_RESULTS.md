# P0737 held-out observed-galaxy radial twins

Date: 2026-08-02

## Outcome

The hosted simulator can now answer two different questions without mixing
them:

1. Does the submitted gravity formula predict the observed rotation curve when
   evaluated on the published SPARC baryonic mass-model channels?
2. Does that prediction survive when those baryonic channels are compressed
   into a small parameter package and regenerated as a radial galaxy twin?

`POST /api/v1/twin-runs` and the browser workbench show the measured velocity
points and uncertainties, the submitted formula on measured baryons, the same
formula on the generated twin, fixed MOND on the twin, and Newtonian baryons on
the twin. A second chart panel shows the submitted twin residual at every
radius against the one-sigma observational band.

## Non-circular protocol

The twin extractor is allowed to read radius, gas, stellar-disk, bulge, disk
surface-brightness, bulge surface-brightness, morphology, distance,
inclination, and catalog quality. It is forbidden to read observed rotation
speed, speed uncertainty, or catalog flat speed until scoring.

Each physical radial channel is represented by six least-squares
piecewise-linear controls in log radius. Signed component velocities are
encoded as signed squared velocity, matching the way their acceleration terms
combine. Surface-brightness channels use a nonnegative logarithmic encoding.
The parameter package records zero gravity parameters.

The leakage audit changed every observed velocity, uncertainty, and flat-speed
value for every one of the 175 galaxies. Every twin content hash remained
unchanged.

## All-galaxy result

| Quantity | Result |
|---|---:|
| SPARC galaxies | 175 |
| radial points | 3,391 |
| median baryonic-acceleration reconstruction error | 0.0097% |
| worst baryonic-acceleration reconstruction error | 12.50% (`UGC06787`) |
| median change in fixed-MOND prediction | 0.75 km/s |
| worst change in fixed-MOND prediction | 6.96 km/s (`NGC2903`) |
| median fixed-MOND RMSE on measured baryons | 13.98 km/s |
| median fixed-MOND RMSE on generated twins | 13.98 km/s |

The status is **needs improvement** because the worst prediction-transport
error is above the preregistered 5 km/s gate. All other frozen gates pass. The
workbench deliberately reports the transport penalty rather than hiding it.

## What the chart means

If the measured-baryon and twin predictions overlap but both miss the observed
points, the formula is the dominant failure. If the formula works on measured
baryons but moves away on the twin, the generator is the dominant failure. If
the source reconstruction score is small and the twin curve follows the
observations within their uncertainties, that is a clean radial success for
the tested formula-plus-source combination.

This still does not simulate individual stars. SPARC rotation curves are
inferred circular speeds of gas and stellar tracers. The twin is a compressed
one-dimensional baryonic surrogate, not a photorealistic or uniquely inferred
three-dimensional galaxy. The existing P0720/P0731 local pipeline remains the
route toward resolved 2D baryonic maps, prior-based 3D realizations, and real
velocity-field scoring.

## Reproduction

```powershell
cd research/galaxy-cluster-unification/hosted-simulator
node scripts/audit-heldout-twins.mjs
node --test test/*.test.mjs
node scripts/build-static.mjs
node scripts/verify-build.mjs
```

Evidence is in
[`../results/p0737_heldout_radial_twin_validation`](../results/p0737_heldout_radial_twin_validation/SUMMARY.md).
