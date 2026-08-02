# P0695B cubic-path audit snapshot

Frozen before synthetic metrics: 2026-08-02

## Bottom line

Cubic interpolation improves spherical radial error to `0.652%` and reduces
the 24/48 quadrature difference to `0.0043%`, while retaining machine-level
rotation covariance, curl, identity, and boundary behavior. It still fails
with `6.61%` tangential/radial power and `6.01%` maximum angular scatter.

The straight-ray Cartesian implementation is retired before observational
testing. The next generator replaces interpolated rays with a coherent
monopole completion: boost the spherical baryonic monopole, retain measured
Newtonian multipoles, and add the cluster routing correction as the same
zero-boundary potential difference.

## Canonical evidence

- [`P0695B result`](../../docs/P0695B_CUBIC_RADIAL_PATH_AUDIT_RESULTS.md)
- [`P0695 linear result`](../../docs/P0695_RADIAL_PATH_POTENTIAL_MATH_AUDIT_RESULTS.md)
- [`public simulator/API plan`](../../docs/PUBLIC_SIMULATOR_API_PLAN.md)
