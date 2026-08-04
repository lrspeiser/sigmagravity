# Sigma v11A tilted nonlinear kinetic falsification

## Decision

Exact Sigma v11A is retired before observational data. Its fixed-background
principal block was positive, but the bounded alignment coefficient depends on
the projected AeST scalar gradient `S`. On a foliation tilted relative to the
aether, `S` contains the coordinate-time velocity of the AeST scalar. The
memory-gradient energy therefore becomes a nonlinear function of that
velocity.

The bounded alignment has a concave region. Because its coefficient is
multiplied by an allowed, arbitrarily large but finite spatial gradient of the
memory field, the scalar velocity Hessian reaches zero at a finite field
configuration and is negative immediately beyond it.

At the frozen audit point, the zero occurs at

$$
v_A={1\over2},
\qquad
\dot\phi=\sqrt3,a_\Sigma,
\qquad
|\partial_x\chi|=\sqrt{1056},a_\Sigma=32.4962,a_\Sigma,
$$

using the conservative positive base scalar Hessian `H_phi=8`. The local
Lagrangian is finite, `36` in normalized units. At `0.99`, `1`, and `1.01`
times the critical memory gradient, the scalar Hessian is respectively
`0.1592`, `0`, and `-0.1608`.

This is the first failed closure after the v10 mechanism reset. It does not
trigger another three-closure reset by itself, but it closes the exact v11A
action and rules out a broad implementation pattern: do not make one field's
kinetic tensor a nonconstant bounded function of another dynamical field's
projected gradient unless a complete degeneracy identity removes the resulting
velocity curvature.

## Local tilted configuration

Use a local Minkowski metric and tilt the unit aether along `x`:

$$
A^\mu=\gamma(1,v,0,0),
\qquad
\gamma=(1-v^2)^{-1/2}.
$$

Choose derivatives

$$
\partial_\mu\phi=(\dot\phi,0,0,0),
\qquad
\partial_\mu\chi=(0,d,0,0).
$$

These are finite ordinary Cauchy data. The projected gradients are

$$
S:S=\gamma^2v^2\dot\phi^2,
\qquad
D\chi:D\chi=\gamma^2d^2,
$$

and they are parallel in the aether-spatial subspace.

Normalize `a_Sigma=1` and define

$$
c=\gamma^2v^2.
$$

V11A's bounded alignment becomes

$$
z(\dot\phi)={c\dot\phi^2\over1+c\dot\phi^2}.
$$

Its exact second derivative is

$$
\boxed{
z''(\dot\phi)
={2c(1-3c\dot\phi^2)\over(1+c\dot\phi^2)^3}.
}
$$

It is positive near zero, vanishes at `c dot(phi)^2=1/3`, and is negative
beyond that point.

## Scalar velocity Hessian

Let `H_phi>0` denote the complete finite positive base scalar velocity Hessian
at the chosen event. The velocity-dependent terms along this one-dimensional
Rayleigh direction are

$$
\mathcal L_{\rm vel}
={1\over2}H_\phi\dot\phi^2
+{1\over2}s\alpha\gamma^2d^2 z(\dot\phi),
$$

where

$$
s={3\over11},
\qquad
\alpha={1\over4}.
$$

Therefore

$$
\boxed{
{\partial^2\mathcal L\over\partial\dot\phi^2}
=H_\phi
+{1\over2}s\alpha\gamma^2d^2z''(\dot\phi).
}
$$

At the convenient finite point `c dot(phi)^2=1`, one has

$$
z''=-{c\over2},
$$

and the Hessian is

$$
H_\phi-{s\alpha\gamma^2c\over4}d^2.
$$

It vanishes at

$$
\boxed{
d_*^2={4H_\phi\over s\alpha\gamma^2c}.
}
$$

Every factor in the denominator is finite and positive for a nonzero
subluminal tilt. Thus `d_*` is finite for every finite positive base Hessian.
Making the AeST scalar kinetic coefficient larger only moves the surface as
`d_* proportional sqrt(H_phi)`; it never removes it.

At `v=1/2`,

$$
\gamma^2={4\over3},
\qquad
c={1\over3}.
$$

For `H_phi=8`, this gives

$$
\dot\phi_*=\sqrt3,
\qquad
d_*^2=1056.
$$

A five-point finite difference reproduces the analytic zero with absolute
error `1.18e-9`.

## Why omitted fields cannot restore positivity

The calculation evaluates the second variation along a direction in velocity
space where only `dot(phi)` changes. Once that diagonal Rayleigh quotient is
negative, the complete symmetric Hessian cannot be positive definite,
regardless of off-diagonal mixing with other velocities.

The source interaction `beta D_chi.J` is only linear in the aether/metric
velocity in this local block and does not supply a positive term growing as
`d^2`. The memory mass and all nonderivative potentials are lower order and do
not enter the velocity Hessian.

This failure is also not an inaccessible infinite-energy limit. The critical
gradient, scalar velocity, Lagrangian, and canonical derivatives are all
finite.

## General lesson

The immediate formula uses `z=Y/(a_Sigma^2+Y)`, but the issue is broader. A
nonconstant bounded smooth function of an unbounded velocity cannot remain
globally convex. Wherever its curvature has the unfavorable sign, multiplying
it by an independent unbounded gradient energy eventually overwhelms any
finite base Hessian.

The exact v11A escape choices are not acceptable:

- Setting the anisotropy fraction to zero removes the directional mechanism
  and returns to the isotropic scalar-memory lane already closed by v4.
- Imposing `|D chi|<d_*` is a new unexplained state cutoff and is not preserved
  by arbitrary sources or memory waves.
- Increasing the base scalar coefficient merely moves the finite surface.
- Fitting the bounded function after seeing data violates the frozen-action
  rule and does not establish global convexity.

## Successor requirement

The next post-reset candidate must not place a baryon-sensitive dynamical
gradient inside another field's kinetic coefficient unless the full action has
an exact all-background degeneracy/positivity identity. More promising
architectures include a gauge/exterior-derivative carrier whose principal
symbol contains no metric or alignment connection, or an orientation-dependent
source with a fixed healthy kinetic operator. Either choice still needs a
unique baryon-forced state and a one-metric weak lensing derivation.

No v11B observational work is authorized until its kinetic and characteristic
identities are analytic.

## Reproduction

```powershell
$env:PYTHONPATH='src'
python scripts/audit_sigma_v11a_tilted_rank.py
python -m pytest -q tests/test_sigma_v11a_tilted_rank.py
```

Machine-readable evidence is in
`results/sigma_v11a_tilted_rank/report.json`.
