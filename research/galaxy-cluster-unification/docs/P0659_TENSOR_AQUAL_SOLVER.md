# P0659 projected tensor-AQUAL solver

## One constitutive equation

P0659 replaces the additive transport correction with a single elliptic field
equation:

\[
\nabla_i\left[\mu\left(\delta_{ij}-\sigma h_i h_j\right)
\nabla_j\Phi\right]=4\pi G\rho_b.
\]

Equivalently, in the projected plane,

\[
\mu\left[(1-\sigma)I+\sigma\,\hat n\hat n\right],
\qquad \hat n\perp\hat h.
\]

The scalar `mu=x/(1+x)` is ordinary simple AQUAL. The anisotropy is the
already-frozen P0643 baryonic quantity

\[
\sigma={a_0\over a_0+|g_N|}\,C_{\rm transverse}
\left(1-e^{-\ell/10\,\mathrm{kpc}}\right).
\]

When stellar and gas fields align, `sigma=0` and the equation is exactly
ordinary AQUAL. Persistent cluster-scale component disagreement lowers the
constitutive eigenvalue along `h` without adding a second equation or an
object-specific strength.

## Variational solver

The symmetric graph discretization is the Euler equation of a positive
quadratic energy at each Picard step:

- Cartesian links carry `mu(1-sigma)`;
- bilinear links perpendicular to `h` carry `mu sigma`;
- the graph Laplacian is symmetric and positive semidefinite;
- explicit Dirichlet boundaries remove its constant null mode; and
- conjugate gradient solves each linearized system.

The constitutive eigenvalues are `mu(1-sigma)` and `mu`. They remain positive
for `mu>0` and `0<=sigma<1`, so the field equation remains elliptic.

## Frozen result

All 14 progression gates pass:

- 65-grid manufactured relative RMS error: `2.0082e-4`;
- measured grid-convergence order: `2.00130`;
- 90-degree rotation relative error: `8.15e-16`;
- direction-reversal relative error: `0.0`;
- aligned `sigma=0` difference from scalar AQUAL: `0.0`;
- nonlinear normalized residual: `4.19e-6` after eight iterations;
- minimum nonlinear constitutive eigenvalue: `9.84e-7`;
- inherited registered cluster/galaxy activation ratio: `18.655`; and
- inherited one-AU anisotropy proxy: `2.024e-8`.

There are no new universal constants beyond P0643 and no per-object gravity
parameters. No galaxy velocity or raw-lensing target was opened.

## What it proves—and does not

P0659 establishes a mathematically coherent and numerically convergent
projected tensor generalization of AQUAL. Unlike P0652-P0657, the geometric
response is inside the constitutive field equation rather than added to an
empirical lens potential afterward.

It does not establish observational accuracy, a three-dimensional tensor
solver, a relativistic photon metric, PPN safety, or a first-principles origin
for the inherited 10 kpc coherence length. The next authorized stage may run
the equation on registered baryonic maps while continuing to keep the sealed
velocity and lens catalogs opaque.

## Reproduction

```powershell
python scripts/run_p0659_tensor_aqual_solver.py
python -m pytest tests/test_tensor_aqual.py tests/test_p0659_tensor_aqual_solver.py -q
```
