# Sigma V19J automated-front implementation failure

## Outcome

The first completely automatic source-only X-ray front implementation produced
a **formal machine-gate pass and a mandatory post-hash visual-audit failure**.
The formal result is preserved unchanged in
`results/sigma_v19j_automated_fronts/report.json` with SHA-256
`06eb0add35583f8ee45c17ba65432b0d082922cb842a28631ff78a73cf8f4923`.
It must not be interpreted as a physical front or shock catalog.

No published front coordinate, hand sector, lensing target, gravity equation or
gravity parameter entered the calculation.

## What worked

The implementation was frozen before a science array was read.  It used the
same four physical Gaussian scales, exposure-aware two-sided count statistic,
`5 sigma` threshold, `100 kpc` length, `20 kpc` gap, curvature range and `90%`
valid-area rule in both clusters.  Its manufactured tests behaved as intended:

- zero ridges in a noiseless uniform field;
- one recovered circular step with radius error `0.505 kpc`; and
- zero retained ridges when the step lacked valid data on one side.

All parent and map hashes matched.  Both science calculations were numerically
finite.

## Why the formal pass is invalid

The post-hash audit is required by V19H specifically to detect software,
coordinate and nonsensical-topology failures.  It found dense nested contour
forests in both clusters.

| Diagnostic | Bullet | Abell 2146 |
|---|---:|---:|
| Pixels above 5 sigma at 64-kpc scale | 136,034 | 82,655 |
| Nonmaximum candidate pixels | 23,737 | 9,660 |
| Linked skeleton pixels | 14,581 | 6,562 |
| Formally retained ridges | 12 | 18 |
| Primary pixels selecting 64 kpc | 12,222 / 12,249 | 347 / 347 |
| Primary reported edge length | 72,927 kpc | 1,720 kpc |

The Bullet primary edge length is `18.38` times the entire map diagonal.  This
is not a long shock.  It is the accumulated edge length of a highly branched
network.

The root cause is mathematical.  The implemented statistic was

\[
Z={|r_+-r_-|\over\sqrt{V_++V_-}}.
\]

It tests whether the two sides have different rates.  A smooth cluster radial
gradient also has different rates, and at large scale the enormous number of
photons makes that difference highly significant.  The statistic therefore
answered “is there a gradient?” rather than “is a discontinuity required over
a smooth gradient?”

## Decision

The V19J report and arrays remain immutable as a failed implementation
artifact.  We do not lower thresholds, choose a visually attractive arc, or
continue to profile fitting and spectra from these ridges.

The required V19K successor must be frozen before another science execution
and must compare two explicit source models:

1. a continuous local brightness profile with gradient and curvature; and
2. the same profile plus a discontinuous compression.

It must also add smooth linear-gradient and smooth radial-profile negative
fixtures.  Ridge length must be the geodesic length of a simple arc, not the
sum of every edge in a branched skeleton.  These corrections address the
audited failure class; they do not use either cluster’s known shock position.

The machine-readable audit is
`results/sigma_v19j_automated_fronts/visual_audit.json`.

## Reproduction

```powershell
python scripts/run_sigma_v19j_automated_fronts.py
python -m pytest -q tests/test_sigma_v19j_automated_fronts.py
python -m ruff check scripts/run_sigma_v19j_automated_fronts.py tests/test_sigma_v19j_automated_fronts.py
```
