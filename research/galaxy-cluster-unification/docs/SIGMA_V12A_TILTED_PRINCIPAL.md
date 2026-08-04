# Sigma v12A constant-background tilted Dirac block

## Decision

Both `lambda_D=+1` and `lambda_D=-1` survive the complete local
constant-background primary-secondary rank gate. Across the frozen scan, the
clock-normalized two-phase Dirac block remains negative and nonzero. This gate
does not select a sign.

The result advances v12A to its physical-characteristic calculation. It does
not establish theory viability.

## Why unitary gauge retains the tilted physics

For a timelike scalar first gradient, a regular local coordinate choice can
use the scalar as time:

$$
\phi=q t.
$$

This does not align the aether with the scalar. Write

$$
\widehat A^\mu=\chi n^\mu+A^\mu,
\qquad
\chi=\sqrt{1+|A|^2}.
$$

A passive spatial rotation places the Fourier wave vector on the third axis
and the aether in the first/third plane. `A_parallel` and `A_perpendicular`
therefore preserve the complete relative angle between the aether, scalar
clock, and wave vector.

No spatial metric component is deleted before the primary nullspace is
constructed. Doing so would remove part of the conformal Class-Ia direction
and falsely lift the degeneracy.

## Quadratic system

Each real Fourier mode is represented by sine and cosine amplitudes. Per phase
the retained ADM configurations are

$$
\{\delta N,\delta N^i,h_{ij},\delta A_i\},
$$

or 13 variables. The exact local quadratic Lagrangian is written

$$
L={1\over2}\dot q^TK\dot q+\dot q^TAq+{1\over2}q^TBq.
$$

The implementation obtains its Hessian by automatic differentiation of:

- the Einstein-Hilbert Fierz-Pauli density with coefficient `F0`;
- the exact AeST ADM Maxwell and `J^mu nabla_mu phi` terms;
- the frozen simple AeST scalar function;
- the exact unit-aether solution `chi=sqrt(1+A_iA^i)`; and
- the v12A `L3,L4,L5` terms with the luminal Class-Ia identities.

The doubled 26-variable kinetic matrix has exactly eight null directions:

- six sine/cosine shift primaries associated with spatial diffeomorphisms;
- two sine/cosine copies of the DHOST primary.

The latter are normalized to unit perturbation of the physical normal clock
`delta r`. Their conformal metric component reproduces

$$
\boxed{
\delta\zeta=-{r^3\bar A_3\over4}\delta r
}
$$

to machine precision throughout the scan.

## Primary-secondary block

Let `Z` contain the two clock-normalized DHOST null vectors and let `K+` be
the Moore-Penrose inverse on the kinetic range. The finite-dimensional Dirac
block is

$$
\boxed{
\Delta_Z
=Z^T\left[-B-(A^T-A)K^+(A^T-A)\right]Z.
}
$$

The antisymmetric mixing term is essential. It includes the longitudinal
aether momentum and cancels the apparent Maxwell lapse-gradient term. After
this reduction the constant-background block is independent of wave number.

The DHOST primary self-bracket and its cross-brackets with the spatial
diffeomorphism primary/secondary directions vanish to numerical precision.

## Frozen scan

The audit used 48 signed random backgrounds per coupling sign with:

- `|q/q_sigma|` from `10^-2` to `10^2`;
- aether tilt magnitude from `10^-2` to `10^2`;
- arbitrary wave-vector/aether angle;
- `k_bar={0.1,1,10}` on the wave-invariance subset.

All nine gates pass. Principal results are:

| Quantity | Result |
|---|---:|
| Positive-sign closest Dirac eigenvalue to zero | `-1.2975747367` |
| Negative-sign closest Dirac eigenvalue to zero | `-4.0032986306` |
| Rank/sign failures | `0` |
| Null-structure failures | `0` |
| Maximum wave-number dependence residual | `3.15e-13` |
| Maximum normalized null residual | `2.34e-16` |
| Maximum Class-Ia conformal-ratio residual | `1.67e-16` |
| Aligned continuum eigenvalue | `-8.0000000400` |
| Aligned analytic target `-4K2` | `-8` |

The tiny aligned offset is caused by evaluating the nondifferentiable
`Y=0` endpoint at the preregistered differentiable tilt sentinel `10^-4`.

## Additional correction: flat scalar tangent

The frozen simple interpolation has

$$
f_y={\sqrt y\over1+\sqrt y}.
$$

Its asymptotic coefficient is one, but its local Minkowski tangent is
`f_y(0)=0`. The published AeST scalar-front formula must therefore use local
`lambda_s=0`, not the asymptotic value one. At `K_B=1,K2=2`, the corrected
flat squared scalar speed is

$$
c_s^2={2-K_B\over K_2K_B}
\left(1+{K_B\lambda_s\over2}\right)={1\over2},
$$

not the previously reported `3/4`. It remains positive and subluminal. The
tensor and vector fronts remain one. The subsequent direct flat
characteristic regression reproduces all of these values from the complete
quadratic Euler pencil; see
[`SIGMA_V12A_FLAT_CHARACTERISTICS.md`](SIGMA_V12A_FLAT_CHARACTERISTICS.md).

## What remains

This audit does not yet prove:

- positive kinetic energy of every reduced physical mode away from the flat
  finite-frequency branch;
- causal physical cones on tilted backgrounds;
- well-posed evolution when the chosen scalar slicing leaves a mode's
  hyperbolicity cone;
- regularity with background scalar Hessian, aether gradient, extrinsic
  curvature, or spacetime curvature;
- the final six-mode degree count on those nonconstant backgrounds.

The flat finite-frequency characteristic and energy subgate now passes. The
remaining items above are the next kill gates. No astronomical data or holdout
were opened.

## Reproduction

```powershell
$env:PYTHONPATH='src'
python scripts/audit_sigma_v12a_tilted_principal.py
python -m pytest -q tests/test_sigma_v12a_tilted_principal.py
```

Machine-readable evidence is in
`results/sigma_v12a_tilted_principal/report.json`.
