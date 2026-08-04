# Sigma v12A tilted AeST clock susceptibility

## Decision

The complete lower-derivative AeST contribution to the v12A
primary-secondary bracket remains strictly nonzero for every finite aether
tilt and scalar-gradient orientation at the selected constants. This removes
another possible constraint-rank failure.

It is not yet the full `Delta_eff`: the spatial differential operator from the
new DHOST terms remains to be derived and could still produce a zero mode or
rank change.

## Reduced tilted invariants

After solving the unit-timelike condition, write the aether as

$$
\widehat A^\mu=\chi n^\mu+A^\mu,
\qquad
\chi=\sqrt{1+|A|^2},
$$

and decompose the scalar gradient into its normal and spatial pieces

$$
q=\nabla_n\phi,
\qquad
s_i=D_i\phi.
$$

Then

$$
Q=\chi q+A\cdot s,
$$

$$
Y=-q^2+|s|^2+Q^2.
$$

Holding `A_i` and `s_i` fixed while differentiating with respect to the normal
clock gives

$$
Y_q=2\left(|A|^2q+\chi A\cdot s\right),
\qquad
Y_{qq}=2|A|^2.
$$

The exact identity

$$
\boxed{
|A|^2Y-{Y_q^2\over4}
=|A|^2|s|^2-(A\cdot s)^2\ge0
}
$$

is simply Cauchy-Schwarz. It controls every relative orientation, including
the exact `Y=0` axis where the scalar gradient is parallel to the aether.

## Eliminating the AeST auxiliaries

The frozen AeST scalar function is

$$
\mathcal F(Y,Q)
=(2-K_B)a_\Sigma^2f(Y/a_\Sigma^2)-2K_2(Q-Q_0)^2,
$$

with

$$
f_y={\sqrt y\over1+\sqrt y},
\qquad
f_{yy}={1\over2\sqrt y(1+\sqrt y)^2}.
$$

Eliminating the published `mu,nu` auxiliary pairs is equivalent to returning
to this original function. Their Schur contribution is therefore already
included by differentiating the reduced Lagrangian

$$
L_{\rm AeST}
=-(2-K_B)Y
-(2-K_B)a_\Sigma^2 f(Y/a_\Sigma^2)
+2K_2(Q-Q_0)^2.
$$

The aether-acceleration term `J.B` is affine in `q` and has zero second
derivative. The exact normal-clock susceptibility is

$$
\boxed{
{d^2L_{\rm AeST}\over dq^2}
=4K_2\chi^2
-(2-K_B)\left[
(1+f_y)Y_{qq}
+{f_{yy}\over a_\Sigma^2}Y_q^2
\right].
}
$$

This is the zeroth-spatial-derivative part of `-Delta_eff`, up to the fixed
canonical sign convention.

## Global lower bound

Let `r=sqrt(Y)/a_Sigma`. Cauchy-Schwarz gives

$$
{f_{yy}\over a_\Sigma^2}Y_q^2
\le {2|A|^2r\over(1+r)^2}
\le {|A|^2\over2}.
$$

Since `0<=f_y<1`, the full negative density contribution is at most

$$
{9\over2}(2-K_B)|A|^2.
$$

Therefore

$$
\boxed{
{d^2L_{\rm AeST}\over dq^2}
\ge
4K_2+\left[4K_2-{9\over2}(2-K_B)\right]|A|^2.
}
$$

A sufficient all-tilt condition is

$$
\boxed{
K_2\ge {9\over8}(2-K_B).
}
$$

For the selected `K_B=1,K2=2` row,

$$
{d^2L_{\rm AeST}\over dq^2}
\ge 8+{7\over2}|A|^2>0.
$$

No finite aether tilt, scalar-gradient magnitude, or relative orientation can
make this lower-derivative susceptibility vanish.

## Stress test

The executable audit uses 50,000 signed random backgrounds with clock,
spatial-gradient, and aether amplitudes spanning `10^-6` through `10^6`.
Every fifth sample lies exactly on the projected `Y=0` axis. It checks the
Cauchy identity, interpolation-curvature bound, global lower bound, and direct
susceptibility.

All six frozen gates pass. The minimum sampled susceptibility is
`8.00000000000402`, the minimum analytic bound is `8.000000000003505`, the
maximum relative Cauchy-identity error is `5.95e-16`, and both the absolute and
relative lower-bound violations are zero. The implementation evaluates `Y` as
a manifest sum of parallel and perpendicular squares so these results are not
artifacts of subtracting nearly equal `q^2` and `Q^2` terms on the projected
axis.

This uses no astronomical data and introduces no parameter.

## Remaining kill gate

The positive zeroth-order term does not by itself prove an invertible
differential operator. The new `L3-L5` terms can contribute spatial derivatives
to `Delta_eff`. We must derive their principal symbol on anisotropic
backgrounds and determine whether a finite wave vector can cancel the positive
AeST susceptibility or change operator rank. That is the next calculation.

The AeST invariants and auxiliary reduction follow the published
[AeST Hamiltonian formulation](https://arxiv.org/abs/2307.15126). The use of a
primary-secondary constraint pair follows the
[DHOST Hamiltonian analysis](https://arxiv.org/abs/1512.06820). The bound above
is the project-specific result; no novelty claim is made.

## Reproduction

```powershell
$env:PYTHONPATH='src'
python scripts/audit_sigma_v12a_tilted_clock.py
python -m pytest -q tests/test_sigma_v12a_tilted_clock.py
```

Machine-readable evidence is in
`results/sigma_v12a_tilted_clock/report.json`.
