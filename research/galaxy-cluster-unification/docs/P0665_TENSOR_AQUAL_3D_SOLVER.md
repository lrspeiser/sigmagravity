# P0665 three-dimensional tensor-AQUAL solver

## Equation and discretization

P0665 implements

\[
\nabla\cdot\left[\mu(|\nabla\Phi|/a_0)
(I-\sigma hh)\nabla\Phi\right]=4\pi G\rho_b
\]

on a three-dimensional Cartesian density grid. The constitutive tensor is
decomposed as

\[
\mu\left[(1-\sigma)I+\sigma(n_1n_1+n_2n_2)\right],
\]

where `(h,n1,n2)` is an orthonormal frame. Cartesian graph links carry the
isotropic eigenvalue and symmetric trilinear links carry the two perpendicular
directions. Every linearized operator is symmetric positive definite after
Dirichlet boundaries remove the constant mode.

## Frozen result

All 15 progression gates pass:

- 25-grid manufactured relative RMS error: `1.429e-3`;
- measured convergence order: `2.0065`;
- rotation covariance error: `8.63e-15`;
- direction-reversal error: `0.0`;
- exact scalar graph-AQUAL limit at `sigma=0`;
- nonlinear residual: `7.87e-6` after eight iterations;
- positive minimum constitutive eigenvalue;
- operator symmetry within the frozen threshold; and
- surface-to-volume column reconstruction error: `1.34e-16`.

No spent RX J2129 image or sealed target was opened.

## Why this matters for lensing

P0664's projected potential could not honestly be converted into photon
deflection without an empirical amplitude. P0665 removes that shortcut. Under
a zero-slip weak-field metric, the next stage can calculate

\[
\boldsymbol\alpha={2\over c^2}
\int \nabla_\perp\Phi\,dz
\]

directly from the solved three-dimensional field. The photon coupling and its
point-mass normalization must be validated before any image catalog is scored.

## Claim boundary

This is numerical machinery, not yet a covariant theory. A zero-slip weak-field
metric is still a closure assumption, and the physical construction of the
three-dimensional baryonic `sigma` field remains the next checkpoint.

## Reproduction

```powershell
python scripts/run_p0665_tensor_aqual_3d_solver.py
python -m pytest tests/test_tensor_aqual_3d.py tests/test_p0665_tensor_aqual_3d_solver.py -q
```
