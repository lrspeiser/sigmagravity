# Sigma v8B preferred-time causal completion selection

## Decision

Sigma v8B passes a narrow fixed-aether scalar selection gate. It advances only
to a full covariant variation, constraint count, and arbitrary-background
characteristic audit. No astronomical map is authorized.

The candidate keeps the one-metric AeST base and the v8A static cubic geometry
equation, but adds a kinetic partner using the timelike direction already
present in AeST. Its coefficient is derived from the existing scalar speed and
adds no sixth physical constant.

## Action addition

Define

$$
Q=A^\mu\nabla_\mu\phi,
\qquad
q^{\mu\nu}=g^{\mu\nu}+A^\mu A^\nu,
\qquad
D^2\phi=q^{\mu\nu}\nabla_\mu\nabla_\nu\phi.
$$

In addition to the v8A action, select

$$
\boxed{
\Delta\mathcal L_C=(\alpha-1)L_H^2(Q-Q_0)^2D^2\phi
}
$$

with

$$
\alpha={1\over3c_s^2(1-c_s^2)}.
$$

At the frozen AeST point `c_s^2=0.75`, this gives

$$
\alpha={16\over9}.
$$

Because `Q=Q0` on the static background, this term and its first variation
vanish there. It does not alter the static spherical solution used in the v8A
failure. Its quadratic variation adds time kinetic energy to perturbations.

## Why the preferred-frame term is not arbitrary

For a fixed timelike vector in local Cartesian coordinates, the operator reduces
to the structure

$$
(\dot\phi)^2\nabla^2\phi.
$$

Although it contains a second derivative, its apparent third derivatives cancel
in the scalar Euler--Lagrange equation:

$$
\mathcal E_C\propto
2\left[(\nabla\dot\phi)^2-\ddot\phi\nabla^2\phi\right].
$$

This fixed-vector identity is only a necessary check. Since `A^mu` is dynamical
in AeST, the full covariant vector and constraint variation must still be done.

## Spherical cone closure

The spatial principal matrix remains

$$
Z_{ij}=c_s^2\delta_{ij}
+2L_H^2\left[(\nabla^2\phi)\delta_{ij}-\nabla_i\nabla_j\phi\right],
$$

while the time coefficient becomes

$$
Z_t=1+2\alpha L_H^2\nabla^2\phi.
$$

Maximizing `Z_r/Z_t` over the complete positive cubic spherical branch gives the
minimum universal completion

$$
\alpha_{\min}={1\over3c_s^2(1-c_s^2)}.
$$

For `c_s^2=0.75`, the numerical scan gives:

| Quantity | Result |
|---|---:|
| Spherical points scanned | `20,001` |
| Maximum radial speed squared | `1.000000` |
| Location of maximum, `u=L_H^2 phi'/r` | approximately `0.1875` |
| Deep radial speed squared | `0.750000` |
| Deep tangential speed squared | `0.187500` |
| Added physical constants | `0` |
| Total provisional physical constants | `5` |

The equal-trace isotropic and rank-one construction probes are also positive and
subluminal. This repairs exactly the spherical scalar failure that retired v8A.

## Arbitrary static Hessian bound

The result is not limited to spherical Hessians. Write the three eigenvalues of
`L_H^2 nabla_i nabla_j phi` as `lambda_i`, their trace as `T`, and the
dimensionless nonnegative baryonic source as `R`. The static equation is

$$
R=c_s^2T+T^2-\sum_i\lambda_i^2\ge0.
$$

On the continuous branch connected to the zero-field boundary this implies
`T>=0`. For fixed `T`, the smallest possible Hessian eigenvalue obeys

$$
\lambda_{\min}\ge
{T-\sqrt{4T^2+6c_s^2T}\over3}.
$$

Since the largest spatial principal eigenvalue is
`c_s^2+2(T-lambda_min)`, this gives a global upper bound on every directional
speed. The `20,001`-point trace scan from `0` through `10^8` reaches a maximum
squared speed of one and never exceeds it. The extremizer has two equal Hessian
eigenvalues and is the same spherical-vacuum geometry that fixed `alpha=16/9`.

The smallest spatial eigenvalue stays positive at every finite trace, although
its extremal lower bound approaches zero as the trace tends to infinity. That
asymptotic strong-coupling warning is carried into the nonlinear validity gate;
the present result proves cone closure, not a uniform EFT cutoff.

## What remains capable of killing v8B

The selection pass is deliberately narrow. v8B must be rejected before data if
any of the following occurs:

1. varying the dynamical vector makes its equation higher than second order or
   destroys AeST's constraint count;
2. lapse, vector, or scalar kinetic mixing has a negative eigenvalue;
3. planar, saddle, separated-source, rotating, or time-dependent backgrounds
   develop a superluminal or nonhyperbolic mode;
4. the completion changes `c_T=1`, the one-metric Weyl identity, or the static
   source equation;
5. Solar and PPN screening cannot use the same five constants;
6. fixed baryons and universal boundary conditions do not select a unique state.

The scalar-sector construction is project-specific, but no novelty claim is
made. AeST, cubic Horndeski interactions, preferred-frame scalar operators,
and characteristic-cone engineering all have substantial prior art. A complete
prior-art comparison waits until the covariant operator survives.

## Reproduction

```powershell
$env:PYTHONPATH='src'
python scripts/audit_sigma_v8b_causal_completion_selection.py
python -m pytest -q tests/test_sigma_v8b_causal_completion.py
```

Machine-readable evidence is stored in
`results/sigma_v8b_causal_completion_selection/report.json`.
