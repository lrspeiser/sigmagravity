# P0634 real-field solver validation

- Overall: **PASS** (12/12 gates)
- Poisson convergence order: 2.001544
- Newtonian Plummer force error: 0.667% median, 1.421% p95
- QUMOND spherical force error: 0.999% median
- AQUAL spherical force error: 1.712% median
- QUMOND/AQUAL Newtonian-limit differences: 0.000154% / 0.000154%
- Runtime: 7.521 seconds

This passes the solver prerequisite frozen in P0633. It validates the numerical
machinery on analytic synthetic cases; it is not an observational validation
and supplies no relativistic light-bending law.
