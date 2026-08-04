# Sigma v8A one-metric AeST--Galileon selection

> **Superseded gate result (2026-08-04):** the selected cubic interaction fails
> its spherical nonlinear-characteristic gate and is retired before data.  The
> one-metric AeST base remains available for a different interaction.  See
> [`SIGMA_V8A_CUBIC_CHARACTERISTIC_GATE.md`](SIGMA_V8A_CUBIC_CHARACTERISTIC_GATE.md).

## Decision

Sigma v8A passes a narrow pre-data action-selection gate and advances only to a
complete variation, constraint, and nonlinear-characteristic audit.  It is not
authorized to open galaxy or cluster maps.

The selected envelope is materially different from v7.  It does not add a
positive massive-spin-2 carrier and it does not identify a conformal auxiliary
scalar with the lensing potential.  Matter, photons, and tensor gravitational
waves use one metric.  The scalar changes that physical metric through the
published AeST kinetic mixing, while a single cubic Horndeski term supplies the
three-dimensional Hessian response.

## Prior-art boundary

The base action is the one-metric relativistic MOND theory of
[Skordis and Zlosnik](https://arxiv.org/abs/2007.00082), often called AeST.  Its
unit timelike vector, scalar, fixed MOND interpolation, one-metric lensing
projection, luminal tensor mode, and published flat-background health region are
prior art.

The shift-symmetric cubic interaction `X box(phi)` is also established
Horndeski/kinetic-gravity-braiding physics.  This project has not established
that combining it with this fixed AeST row is novel.  The project-specific
hypothesis is narrower: can the cubic Hessian interaction make the already
Weyl-active AeST response sensitive to separated baryonic geometry without
destroying AeST's health and universality?

## Why classic TeVeS was not selected

Classic TeVeS uses the disformal physical metric

$$
\widetilde g_{\mu\nu}
=e^{-2\phi}g_{\mu\nu}-2\sinh(2\phi)U_\mu U_\nu.
$$

It gives the desired same-sign scalar contribution to dynamics and lensing, but
the later one-metric construction was motivated in part by the incompatibility
of classic TeVeS tensor propagation with the gravitational-wave/light-speed
constraint.  AeST minimally couples all matter to `g_mn` and has the
Einstein-Hilbert tensor quadratic action, so `c_T=1` structurally.

## Frozen action envelope

With `A_m A^m=-1`, define

$$
Y=(g^{\mu\nu}+A^\mu A^\nu)
\nabla_\mu\phi\nabla_\nu\phi,
\qquad
Q=A^\mu\nabla_\mu\phi.
$$

The selected action is

$$
\begin{aligned}
S={1\over16\pi G}\int d^4x\sqrt{-g}\Big[&R
-{K_B\over2}F_{\mu\nu}F^{\mu\nu}
+2(2-K_B)J^\mu\nabla_\mu\phi
-(2-K_B)Y-\mathcal F(Y,Q)\\
&-\lambda(A^2+1)
-{L_H^2\over2}(\nabla\phi)^2\Box\phi\Big]+S_m[g].
\end{aligned}
$$

The last term is the only addition to the fixed AeST envelope.  Its sign is
fixed positive in the scalar equation; it is not a lensing multiplier.

The free function is frozen rather than fit:

$$
\mathcal F(Y,Q)
=(2-K_B)a_\Sigma^2 f(Y/a_\Sigma^2)
-2K_2(Q-Q_0)^2,
$$

$$
f(y)=y-2\sqrt y+2\ln(1+\sqrt y),
\qquad
f_y={\sqrt y\over1+\sqrt y}.
$$

Thus the quasistatic interpolation is the fixed simple function and introduces
no shape parameter.  There are five provisional universal parameters:

$$
\{a_\Sigma,\mu_\Sigma,K_B,K_2,L_H\},
$$

with `lambda_s=1` fixed and

$$
Q_0=\mu_\Sigma\sqrt{\frac{2-K_B}{2K_2}}.
$$

## Physical-metric projection

The AeST quasistatic action diagonalizes with

$$
\Phi_{\rm phys}=\widehat\Phi+\varphi,
\qquad
\Psi_{\rm phys}=\Phi_{\rm phys}.
$$

Therefore a unit scalar response gives

$$
\boxed{
\delta\Psi=1,\qquad
\delta\Phi=1,\qquad
\delta W=1.
}
$$

This is the central difference from v7C, where the leading scalar shifts were
opposite and canceled from `W`.  Here light and slow matter respond to the same
physical metric; there is no photon-only rule.

## Geometry interaction

Variation of

$$
-{L_H^2\over2}(\nabla\phi)^2\Box\phi
$$

adds the second-order scalar equation term

$$
L_H^2\left[(\Box\phi)^2
-\nabla_\mu\nabla_\nu\phi\,
 \nabla^\mu\nabla^\nu\phi\right].
$$

On a static Cartesian background, compare two Hessians having the same trace:

$$
H_{\rm iso}=\operatorname{diag}(1,1,1),
\qquad
H_{\rm rank1}=\operatorname{diag}(3,0,0).
$$

Both have `tr(H)=3`, but

$$
(\operatorname{tr}H)^2-\operatorname{tr}(H^2)
=6\quad\hbox{and}\quad0.
$$

The activation therefore cannot be reduced to local density or spherical
`M/r^3`.  It responds to how curvature is distributed among principal
directions, with no galaxy/cluster label.

## Base linear health

The construction point is

$$
K_B=1,\qquad K_2=2,\qquad\lambda_s=1.
$$

Using the published flat-background AeST spectrum gives

| Mode | Squared speed |
|---|---:|
| Tensor | `1` |
| Vector | `1` |
| Scalar | `0.75` |

The base linear spectrum is positive and subluminal.  The cubic term starts at
third perturbative order on the constant flat background and therefore does not
change that quadratic spectrum or the tensor light cone.

This is not enough to declare the combined theory healthy.  Static scalar
gradients activate the cubic term and change the principal symbol.  The full
nonlinear characteristic cones, lapse/vector constraints, and branch uniqueness
remain untested.

## Next gate

Before any observational array is opened, v8A must:

1. derive the complete metric, scalar, vector, and multiplier equations from the
   combined action;
2. verify the diffeomorphism identity and one physical matter metric;
3. compute the full kinetic Hessian and degree-of-freedom count;
4. scan the principal symbol on time-gradient, static spherical, planar, and
   separated-source backgrounds;
5. require positive kinetic and gradient eigenvalues and no characteristic
   faster than the physical metric light cone;
6. derive the PPN and Solar-screened solution without changing the five
   constants;
7. prove that fixed baryons and universal boundary data select a unique state.

Any failure retires this exact v8A envelope before it reaches a galaxy or lens
map.

## Reproduction

```powershell
$env:PYTHONPATH='src'
python scripts/audit_sigma_v8a_aest_galileon_selection.py
python -m pytest -q tests/test_sigma_v8_aest_galileon.py
```

Machine-readable evidence is stored in
`results/sigma_v8a_aest_galileon_selection/report.json`.
