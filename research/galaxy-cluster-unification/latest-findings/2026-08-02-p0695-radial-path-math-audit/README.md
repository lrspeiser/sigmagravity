# P0695 radial-path mathematical-audit snapshot

Frozen before synthetic metrics: 2026-08-02

## Bottom line

The path-potential concept recovers the spherical algebraic radial force to
`1.17%` RMS, is stable from 24 to 48 quadrature points, rotates covariantly to
machine precision, remains curl-free, and preserves the zero-boundary routing
correction exactly.

The first-order interpolation implementation does not advance. It creates
`9.15%` tangential/radial power and `5.79%` maximum angular scatter, exceeding
the frozen `3%` and `5%` limits. No observational score was opened.

The next audit changes only numerical interpolation from linear to cubic under
the identical equation, fields, masks, and rejection thresholds.

## Canonical evidence

- [`P0695 mathematical result`](../../docs/P0695_RADIAL_PATH_POTENTIAL_MATH_AUDIT_RESULTS.md)
- [`P0694 endpoint retirement`](../../docs/P0694_SPENT_DDO154_ROUTING_CONTINUUM_RESULTS.md)
- [`public simulator/API plan`](../../docs/PUBLIC_SIMULATOR_API_PLAN.md)
