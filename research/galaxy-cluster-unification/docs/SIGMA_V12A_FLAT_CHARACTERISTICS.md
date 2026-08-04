# Sigma v12A direct flat characteristic regression

## Decision

The complete local quadratic Euler pencil independently reproduces the
corrected flat AeST spectrum and its linear six-degree count. All nine frozen
gates pass:

- one zero-frequency scalar sector;
- one finite-frequency scalar with principal squared speed `1/2`;
- four tensor/vector modes with principal squared speed `1`;
- positive quadratic energy for every finite-frequency root; and
- no dependence on the sign of `lambda_D` at the flat clock background.

This is the first direct v12A characteristic regression. It validates the
flat finite-frequency statement but does not establish finite-tilt or
nonconstant-background viability.

## Euler pencil

For the real sine/cosine Fourier amplitudes, the quadratic Lagrangian is

$$
L={1\over2}\dot q^TK\dot q+\dot q^TAq+{1\over2}q^TBq.
$$

Every lapse, shift, spatial-metric, and aether component is retained while the
Euler equation is formed:

$$
K\ddot q+(A-A^T)\dot q-Bq=0.
$$

For `q=exp(s t)u`, the characteristic polynomial and its linearization are

$$
P(s)=s^2K+s(A-A^T)-B,
$$

$$
\begin{pmatrix}0&I\\B&-(A-A^T)\end{pmatrix}U
=s
\begin{pmatrix}I&0\\0&K\end{pmatrix}U.
$$

Only after constructing these complete matrices do we impose the spatial
diffeomorphism gauge `h13=h23=h33=0` in each real phase. The generalized
pencil is singular because lapse, shifts, and the DHOST primary retain their
constraint roles.

## Root count

The gauge-fixed first-order pencil has dimension 40. It produces:

| Root class | Generalized roots | Physical configuration modes |
|---|---:|---:|
| finite | `24` | `6` |
| constraint roots at infinity | `16` | -- |

A physical configuration mode generates four finite roots in this real
representation: positive/negative time roots times sine/cosine phases. The
finite roots split as:

| Sector | Roots | Modes |
|---|---:|---:|
| zero-frequency scalar | `4` | `1` |
| finite-frequency scalar | `4` | `1` |
| tensor plus vector | `16` | `4` |

This reproduces the published AeST local linear count of six. It is not a
replacement for the nonlinear Hamiltonian constraint count.

## Frozen numerical result

The simple interpolation is represented using `sqrt(Y)`, whose exact `Y=0`
endpoint is not twice differentiable in automatic differentiation. The audit
therefore uses a frozen positive aether-tilt sentinel `10^-5` and wave numbers
`100,300,1000`, then extrapolates the group values linearly in `1/k^2`.

| Quantity | Result | Target |
|---|---:|---:|
| Scalar principal squared speed | `0.5000024999` | `0.5` |
| Luminal principal lower edge | `0.9999999998` | `1` |
| Luminal principal upper edge | `1.0000000002` | `1` |
| Zero-sector principal squared speed | `-5.71e-20` | `0` |
| Minimum finite-frequency mode energy | `0.4000021` | positive |
| Maximum Euler-polynomial residual | `1.06e-11` | below `1e-8` |
| Maximum imaginary part of squared speeds | `2.29e-12` | below `1e-8` |
| Positive/negative `lambda_D` spectral difference | `0` | `0` |

The residual `2.50e-6` shift of the scalar intercept is the declared finite
tilt regulator effect. It lies within the frozen `10^-5` regression tolerance.

For an oscillatory root with frequency `omega` and unit-normalized mode
amplitude `u`, the conserved quadratic energy is evaluated as

$$
E={1\over4}u^\dagger(\omega^2K-B)u.
$$

All twenty roots belonging to the five finite-frequency physical modes have
positive energy. The four zero-sector roots are deliberately excluded from
that statement: the published AeST constant/linearly-growing Jeans-like
sector remains unresolved.

## Interpretation

This direct calculation corrects and strengthens the earlier formula-level
statement. The local simple interpolation has `f_y(0)=0`, so the scalar front
is `c_s^2=1/2`; the previously reported `3/4` used the interpolation's
asymptotic coefficient. The v12A interaction and its first derivative vanish
at the background clock, so neither sign of `lambda_D` alters the quadratic
flat spectrum.

The result does **not** show that v12A is viable. It does not yet prove:

- a common Cauchy covector for finite scalar/aether tilt;
- positive reduced energy for tilted physical modes;
- causal cones on arbitrary wave orientations;
- resolution of the zero-frequency infrared sector;
- the nonlinear six-degree constraint count; or
- regularity with scalar Hessian, aether gradient, extrinsic curvature, or
  spacetime curvature.

The subsequent scalar-unitary finite-tilt grid fails, so the next kill gate is
now the general-time invariant common-cone and reduced-energy calculation; see
[`SIGMA_V12A_TILTED_CHARACTERISTICS.md`](SIGMA_V12A_TILTED_CHARACTERISTICS.md).
No astronomical data or holdout were opened.

## Reproduction

```powershell
$env:PYTHONPATH='src'
python scripts/audit_sigma_v12a_flat_characteristics.py
python -m pytest -q tests/test_sigma_v12a_flat_characteristics.py
```

Machine-readable evidence is in
`results/sigma_v12a_flat_characteristics/report.json`.
