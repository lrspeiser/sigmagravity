# Sigma v8A cubic characteristic gate

## Decision

The exact cubic Horndeski geometry interaction selected in v8A is retired before
observational use.  The one-metric AeST base is retained as a possible carrier,
but it needs a different geometry interaction.

This is not a data-fit failure.  It is a pre-data conflict between making the
cubic term dynamically important and keeping every characteristic inside the
physical matter light cone.

## Reduced spherical equation

Normalize the flat AeST scalar time coefficient to one.  At the selected point,
its spatial coefficient is

$$
c_s^2=0.75.
$$

For a static spherical exterior, write

$$
b={\varphi'\over r},\qquad a=\varphi'',\qquad u=L_H^2 b.
$$

The cubic scalar equation integrates once to

$$
c_s^2 r^2\varphi'+2L_H^2r(\varphi')^2=\mathcal C.
$$

Differentiating this conserved flux fixes the Hessian ratio:

$$
{a\over b}=-2{c_s^2+u\over c_s^2+4u}.
$$

The perturbation principal coefficients are

$$
Z_t=1+2L_H^2(a+2b),
$$

$$
Z_r=c_s^2+4u,
\qquad
Z_\Omega=c_s^2+2L_H^2(a+b).
$$

The squared radial characteristic speed is `Z_r/Z_t`.

## Positive sign result

The analytic light-cone crossing occurs at

$$
u_\star={1-2c_s^2+\sqrt{3c_s^4-3c_s^2+1}\over2}
=0.0807189.
$$

At this point the cubic-to-linear conserved-flux ratio is

$$
{2u_\star\over c_s^2}=0.21525,
$$

so the cubic term supplies only

$$
{0.21525\over1+0.21525}=0.17713
$$

of the total scalar flux.  Any larger fraction is radially superluminal on this
branch.  In the nonlinear limit,

$$
c_r^2\longrightarrow {4\over3},
\qquad
c_\Omega^2\longrightarrow {1\over3}.
$$

Thus the sign that supplies a regular Vainshtein-like positive-source branch
cannot provide an order-unity geometry correction while satisfying this
project's strict causal gate.

## Opposite sign result

Reversing the cubic sign changes the conserved flux to

$$
c_s^2 b-2|L_H|^2b^2.
$$

This expression has a maximum at

$$
|L_H|^2b={c_s^2\over4}=0.1875.
$$

At exactly that point the radial principal coefficient

$$
Z_r=c_s^2-4|L_H|^2b
$$

vanishes.  A compact positive source demanding a larger flux has no continuation
on this branch.  The opposite sign therefore trades superluminality for branch
termination and loss of ellipticity.

## What is and is not falsified

The result agrees with the established Galileon concern that successful
Vainshtein screening and fully subluminal perturbations are difficult to combine;
see [Garcia-Saenz (2013)](https://arxiv.org/abs/1303.2905) and
[Goon and Hinterbichler (2017)](https://arxiv.org/abs/1609.00723).

The audit falsifies the exact single cubic interaction as Sigma's strong
geometry carrier.  It does not falsify:

- the one-metric AeST projection, where the same physical potential governs
  matter and light;
- every Horndeski or degenerate scalar-vector-tensor interaction;
- a geometry term whose nonlinear principal symbol is bounded by construction.

The next candidate must preserve the one-metric Weyl-active projection while
bounding its geometry response before any kinetic or spatial eigenvalue changes
sign or crosses the matter light cone.

## Reproduction

```powershell
$env:PYTHONPATH='src'
python scripts/audit_sigma_v8a_cubic_characteristics.py
python -m pytest -q tests/test_sigma_v8a_cubic_characteristics.py
```

Machine-readable evidence is stored in
`results/sigma_v8a_cubic_characteristic_gate/report.json`.
