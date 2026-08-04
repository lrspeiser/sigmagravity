# Sigma v9A bounded alignment falsification

## Verdict

Exact Sigma v9A is **retired as the standalone galaxy--cluster completion
before new observational use**.

The direct first-derivative interaction has a finite static principal-rank
surface for every tested nonzero coupling.  A minimal saturation in both
gradients removes that failure throughout the declared necessary scan, but it
cannot supply the missing cluster monopole: its action and both first-variation
fluxes vanish identically whenever the scalar and aether gradients are aligned.
Every spherical system therefore remains exactly the fixed AeST/MOND solution,
independent of the coupling.

This is a mechanism-null result, not a parameter-fit failure.  No new galaxy,
cluster, image, shear, or holdout array was opened.

## Covariant candidate

Keep the one-metric AeST base.  With the unit timelike aether $A^\mu$, define

$$
q_{\mu\nu}=g_{\mu\nu}+A_\mu A_\nu,
\qquad
S_\mu=q_\mu{}^\nu\nabla_\nu\phi,
\qquad
J_\mu=A^\nu\nabla_\nu A_\mu,
$$

and

$$
Y=S_\mu S^\mu,
\qquad
Z=J_\mu J^\mu,
\qquad
U=S_\mu J^\mu,
\qquad
N=YZ-U^2.
$$

The projector makes $S_\mu$ and $J_\mu$ spatial in the aether rest frame, so
Cauchy--Schwarz gives $N\ge0$.  In that frame,

$$
N=|\mathbf S|^2|\mathbf J|^2\sin^2\theta.
$$

The direct proposal was

$$
\boxed{
\Delta\mathcal L_{9A}
=-4\eta_\Sigma a_\Sigma^2
{N\over(a_\Sigma^2+Y)^2}.
}
$$

It uses only first derivatives and is quartic around the constant Minkowski
background.  It therefore does not alter the published quadratic AeST tensor,
vector, or scalar spectrum, and does not alter $c_T=1$ at that order.  The five
provisional constants are

$$
\{a_\Sigma,\mu_\Sigma,K_B,K_2,\eta_\Sigma\}.
$$

There is no object label, object-specific force parameter, second matter
metric, or photon multiplier.

## Exact weak-static fluxes

Write $D_Y=a_\Sigma^2+Y$.  Direct differentiation gives

$$
{\partial\Delta\mathcal L_{9A}\over\partial J_\mu}
=-8\eta_\Sigma a_\Sigma^2
{YJ^\mu-US^\mu\over D_Y^2},
$$

$$
{\partial\Delta\mathcal L_{9A}\over\partial S_\mu}
=-8\eta_\Sigma a_\Sigma^2
\left[
{ZS^\mu-UJ^\mu\over D_Y^2}
-{2N S^\mu\over D_Y^3}
\right].
$$

The scalar and aether equations receive the corresponding divergences, in
addition to the AeST base fluxes.  Automatic differentiation, analytic fluxes,
and independent centered finite differences agree in the tests.

## Why the apparently safe kinetic bound is insufficient

At small $J$ in the aether rest frame, decompose a vector perturbation parallel
and perpendicular to $S_\mu$.  The two Lagrangian coefficients are exactly

$$
K_\parallel=K_B,
\qquad
K_\perp=K_B-4\eta_\Sigma{y\over(1+y)^2},
\qquad
y={Y\over a_\Sigma^2}.
$$

Since $4y/(1+y)^2\le1$, $0\le\eta_\Sigma<K_B$ keeps this isolated vector
coefficient positive.  At the selected $K_B=1$, $\eta_\Sigma=2/3$, its minimum
is $1/3$.  This looks capable of a factor-three perpendicular response.

It is not a sufficient coupled-field health condition.  The coefficient is
still quadratic and unbounded in $J$, while its curvature with respect to
$S$ changes sign across the transition.  The full six-variable quasistatic
principal matrix covers

$$
(J_x,J_y,J_z,S_x,S_y,S_z)
$$

and includes the exact AeST static mixing.  At $Y/a_\Sigma^2=1$ and $U=0$, its
first rank surfaces are:

| $\eta_\Sigma$ | $Z/a_\Sigma^2$ | $|J|/a_\Sigma$ |
|---:|---:|---:|
| 0.25 | 9.023588 | 3.003929 |
| 0.50 | 4.540129 | 2.130758 |
| 2/3 | 3.498466 | 1.870419 |
| 0.80 | 3.033953 | 1.741825 |
| 0.95 | 2.771520 | 1.664788 |

At every crossing the raw static inertia changes from `(3,0,3)` to `(2,0,4)`.
For the selected row the null direction is `54.81%` aether acceleration and
`45.19%` scalar gradient.  It is not a decoupled or coordinate-only direction.
The one-sided action is therefore retired.

## Minimal double saturation

The smallest repair that preserves the desired small-$J$ coefficient is

$$
\boxed{
\Delta\mathcal L_{9A,s}
=-4\eta_\Sigma a_\Sigma^4
{N\over(a_\Sigma^2+Y)^2(a_\Sigma^2+Z)}.
}
$$

It is globally bounded because $N\le YZ$:

$$
|\Delta\mathcal L_{9A,s}|\le\eta_\Sigma a_\Sigma^2.
$$

Across 2,212 deterministic and frozen-random points covering

$$
10^{-4}\le {Y\over a_\Sigma^2},{Z\over a_\Sigma^2}\le10^4,
\qquad -1\le\cos\theta\le1,
$$

the saturated term preserves the base `(3,0,3)` static inertia.  Its minimum
singular value is `1.369856`, and its minimum combined/base determinant ratio
is `0.257337`.  An independent finite-difference Hessian agrees with automatic
differentiation below `2e-7` relative error.

This is a necessary local static pass, not a full nonlinear Hamiltonian or
characteristic proof.  The next mechanism result makes that expensive audit
unnecessary for this exact candidate.

## Exact spherical null

For every aligned configuration $J_\mu=\kappa S_\mu$,

$$
N=0,
\qquad
YJ_\mu-US_\mu=0,
\qquad
ZS_\mu-UJ_\mu=0.
$$

Consequently the action density and both first-variation fluxes vanish for
both variants.  In a static spherical system,

$$
S_i=S(r)\hat r_i,
\qquad
J_i=J(r)\hat r_i,
$$

so this identity holds at every radius.  The v9A term cannot change the
spherical AeST solution, the radial force, or the spherical lensing monopole
for any value of $\eta_\Sigma$.

The existing development result already shows that fixed galaxy-scale MOND
provides only `0.318` of the required CLASH-derived cluster field on average,
corresponding to a target amplification of

$$
{1\over0.318}=3.14465.
$$

At $\eta_\Sigma=2/3$, even the best-case *orthogonal*, transition-scale,
small-$J$ response is only `3.0`; the full target is unreachable.  Closing 75%
of the amplitude gap would require at least

$$
\sin^2\theta=0.92495,
\qquad
\theta=74.10^\circ.
$$

The spherical value is $\theta=0$, hence amplification one.

A deliberately favorable two-source geometry control confirms that separated
sources can create local angular activation, but not a generic monopole.  A
single radial source has maximum $\sin^2\theta<6\times10^{-16}$.  The
two-source field reaches a local maximum near one, but its field-weighted mean
is `0.0243`, its 90th percentile is `0.0477`, and only `0.62%` of its weighted
domain exceeds $\sin^2\theta=0.5`.  This synthetic construction is illustrative,
not an observational or AeST solution; the exact spherical null alone decides
the gate.

## Prior-art boundary

The one-metric AeST base is the published theory of
[Skordis and Zlosnik](https://arxiv.org/abs/2007.00082).  Broad scalar--vector--
tensor action classes and aether-acceleration derivative invariants are also
established; relevant examples include
[ghost-free scalar--vector--tensor theories](https://arxiv.org/abs/1801.01523),
[Einstein--Maxwell--aether derivative interactions](https://arxiv.org/abs/1407.6014),
and [Einstein--aether--axion cross-couplings](https://arxiv.org/abs/1606.09286).

A targeted primary-literature search did not establish that the exact rational
Gram function above has been published.  That is not a novelty finding, and no
novelty claim is made.  Algebraic novelty would not rescue the exact spherical
null.

## Decision and successor constraint

- Retire the one-sided term on the finite principal-rank surface.
- Retire the saturated term as the standalone galaxy--cluster completion on
  the exact spherical monopole null.
- Do not tune $\eta_\Sigma$; multiplying an identically zero source cannot
  change it.
- Do not append this term as an unproved sixth-parameter topology correction.

The successor must provide **both** a baryon-forced nonzero spherical monopole
and orientation/shear transport.  It must use bounded first-order dynamics or
an exact arbitrary-background degeneracy identity.  Another pure angular gate
cannot meet the goal.

## Reproduction

```powershell
$env:PYTHONPATH='src'
python scripts/audit_sigma_v9a_bounded_alignment.py
python -m pytest -q tests/test_sigma_v9a_bounded_alignment.py
```

The frozen protocol is
[`../configs/sigma_v9a_bounded_alignment_gate.json`](../configs/sigma_v9a_bounded_alignment_gate.json),
and the machine-readable report is
[`../results/sigma_v9a_bounded_alignment_gate/report.json`](../results/sigma_v9a_bounded_alignment_gate/report.json).
