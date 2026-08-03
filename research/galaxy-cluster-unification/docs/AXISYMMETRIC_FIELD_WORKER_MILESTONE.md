# Axisymmetric cylindrical field-worker milestone

Date: 2026-08-02

## Outcome

The formula-independent reference worker can now solve stationary scalar
elliptic equations on a uniform axisymmetric cylindrical `(r,z)` grid. This is
the efficient geometry needed to distinguish a thin disk, a thick disk, and an
axisymmetric bulge without replacing each one by a spherical or planar proxy.

The implementation is not a new gravity formula. It executes the submitted
expression tree through the same general path used for Cartesian Poisson,
Refracted Gravity, AQUAL-like, and coupled-field manifests.

## Exact coordinate contract

An axisymmetric array bundle must declare:

```json
{
  "coordinateSystem": "axisymmetric_cylindrical",
  "dimensions": 2,
  "axisOrder": ["r", "z"],
  "origin": [0, 0],
  "spacing": [0.03125, 0.03125]
}
```

The first radial sample is the physical symmetry axis `r=0`. It receives the
regularity condition `partial u / partial r = 0`; it is not treated as a
Dirichlet wall. The declared outer radial and upper/lower vertical surfaces
retain the manifest's boundary value.

The finite-volume operator is

```text
div(a grad(u)) = (1/r) d[r a du/dr]/dr + d[a du/dz]/dz.
```

At the axis, the radial term uses its regular limit,
`4 a_(1/2) (u_1-u_0)/dr^2`. Away from the axis, cylindrical face areas supply
the `(r +/- dr/2)/r` weights. Coefficients use harmonic face averages.

## Acceptance evidence

The primary manufactured answer is

```text
u(r,z) = J0(alpha r) sin(pi z),
alpha = first zero of J0,
laplacian(u) = -(alpha^2 + pi^2) u.
```

| Cells per axis | Relative field error | Discrete residual |
|---:|---:|---:|
| 25 | 1.259793e-3 | 1.844284e-15 |
| 49 | 3.148784e-4 | 3.776145e-15 |
| 97 | 7.871635e-5 | 7.432610e-15 |

Halving the spacing reduces the field error by approximately four, the
expected second-order behavior. The radial gradient is exactly zero at the
axis. A separate spatially varying coefficient case recovers the exact
discrete manufactured field to below `1e-11` relative error.

The immutable field-job test also verifies that `axisOrder`, `origin`, worker
version, input hashes, numerical metadata, and output artifacts survive the
complete job path. Ambiguous axes, a nonzero radial origin, and Cartesian
convolution semantics are rejected.

## What this makes possible

- Compare the same submitted law on thin disks, thick disks, and bulges.
- Test density- or field-dependent scalar coefficients in a galaxy-appropriate
  geometry.
- Run much cheaper disk/bulge resolution studies before spending resources on
  full Cartesian 3D calculations.
- Make the coordinate convention part of the immutable scientific identity.

## What remains outside this milestone

- Bars, spiral arms, lopsidedness, mergers, and irregular cluster structure
  require Cartesian 3D.
- Cylindrical nonlocal kernels need a real Hankel/azimuth-integrated operator;
  Cartesian `linear_same` convolution is deliberately rejected.
- Circular-speed and resolved velocity-field adapters now consume `(r,z)`
  fields through the separately documented v0.26 milestone. Axisymmetric
  photon lensing remains unbuilt.
- Vector/tensor field solves, mixed boundaries, refinement, and production
  workers remain unfinished.
- Numerical agreement with manufactured solutions validates the solver, not a
  gravity theory or an observational fit.
