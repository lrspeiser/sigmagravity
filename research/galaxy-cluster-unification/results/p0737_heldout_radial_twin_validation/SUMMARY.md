# P0737 held-out radial twin validation

- Status: **NEEDS IMPROVEMENT**
- Public SPARC galaxies: **175** (3,391 radial points)
- Observed speed targets used to build a twin: **no**
- Gravity parameters used to build a twin: **0**
- Median baryonic-acceleration reconstruction error: **0.010%**
- Worst baryonic-acceleration reconstruction error: **12.50%** (UGC06787)
- Median fixed-MOND prediction transport: **0.75 km/s**
- Worst fixed-MOND prediction transport: **6.96 km/s** (NGC2903)
- Median fixed-MOND RMSE on measured baryons: **13.98 km/s**
- Median fixed-MOND RMSE on generated twins: **13.98 km/s**

The simulator can now make a non-circular radial twin of every catalog galaxy,
apply the same submitted formula to both the measured baryonic source and the
generated source, and reveal the measured rotation speeds only for scoring.
Perturbing every observed speed and uncertainty leaves every twin package hash
unchanged.

The commissioning result is **needs improvement**, rather than pass, because
the six-control-point compression changes fixed MOND's prediction by
6.96 km/s for NGC2903, above the frozen
5 km/s worst-case gate. The public report therefore keeps the source error,
formula error, and transport error separate. A radial twin is not yet a full
2D/3D simulated galaxy or an individual-star orbit model.
