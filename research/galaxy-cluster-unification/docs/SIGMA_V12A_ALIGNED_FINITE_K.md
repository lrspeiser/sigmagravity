# Sigma v12A corrected aligned finite-wave-vector Dirac subgate

## Correction and decision

The earlier project conclusion that `lambda_D=+1` has an aligned finite-wave-
vector constraint zero is withdrawn. That calculation retained the
`A4 q^2|Dq|^2` clock-gradient term but omitted the spatial-metric component of
the Class-Ia primary null direction. It was a zero of one unprojected matrix
block, not a zero of the Dirac operator.

Including the conformal metric displacement, its Einstein curvature terms,
and the AeST Maxwell term gives the complete aligned symbol

$$
\boxed{
{\Delta(\bar k)\over F_0}
=-\left[4K_2+{2K_B\over r^2}\bar k^2\right],
\qquad r={q\over q_\Sigma}.
}
$$

It is strictly nonzero for every finite nonzero aligned clock and wave number
when `K2>0` and `K_B>0`, independently of the sign or magnitude of `lambda_D`.
The positive and negative sign sentinels therefore both survive this aligned
subgate. Neither has yet passed the tilted, nonzero-gradient, anisotropic
principal matrix.

## Clock and metric must be projected together

On the aether-aligned, zero-spatial-scalar-gradient auxiliary branch, write

$$
q=\nabla_n\phi,\qquad
V_*=\phi_{nn},\qquad
u_i=D_iq.
$$

The covariant Hessian invariants are

$$
L_3=-q^2V_*^2-q^3KV_*,
$$

$$
L_4=-q^2V_*^2+q^2|Dq|^2,
$$

$$
L_5=q^4V_*^2.
$$

The velocity terms form the Class-Ia degenerate square

$$
\kappa K^2+2bKV_*+{b^2\over\kappa}V_*^2,
$$

with

$$
\kappa=-{2F_0\over3},
\qquad
b=-{q^3A_3\over2}.
$$

The null velocity therefore does not change `q` alone. For a conformal
spatial-metric perturbation `zeta`, it obeys

$$
\delta\zeta
=-{q^3A_3\over4F_0}\delta q.
$$

Any spatial principal calculation that varies `q` while holding this metric
direction fixed is not the primary-secondary bracket.

For a nonzero timelike clock, unitary gauge may then be imposed regularly.
The scalar definition gives `delta N/N=-delta q/q`. The published DHOST
Hamiltonian analysis shows that the gauge-fixed Dirac determinant contains
the same primary-secondary factor (multiplied only by nonzero lapse powers),
so this is a convenient way to compute the complete aligned operator. The
clock-lapse relation is what generates the Einstein clock-metric cross term
and lets the AeST electric/Maxwell lapse gradient enter the final Schur
complement. Imposing the relation while omitting the metric null displacement,
as the superseded audit effectively did, is inconsistent.

## Complete aligned spatial quadratic form

Use the normalized variables

$$
r={q\over q_\Sigma},
\qquad
\bar A_3={q_\Sigma^4A_3\over F_0},
\qquad
\eta=-{r^3\bar A_3\over4},
$$

so that `delta zeta=eta delta r`. There are four spatial-gradient
contributions along the null direction:

| source | normalized coefficient |
|---|---:|
| direct DHOST clock term | `r^2 A4_bar` |
| Einstein clock-metric cross term | `-4 eta/r` |
| Einstein metric term | `2 eta^2` |
| AeST Maxwell term | `K_B/r^2` |

The first three sum to

$$
r^2\bar A_4-{4\eta\over r}+2\eta^2.
$$

Substituting

$$
\bar A_4=-\bar A_3-{r^4\bar A_3^2\over8}
$$

and `eta=-r^3 A3_bar/4` gives the exact Class-Ia cancellation

$$
\boxed{
r^2\bar A_4-{4\eta\over r}+2\eta^2=0.
}
$$

The entire `lambda_D`-dependent aligned gradient coefficient disappears. The
only remaining spatial term is

$$
\boxed{C_{\rm aligned}={K_B\over r^2}>0.}
$$

Combining this with the AeST clock susceptibility `4K2` yields the complete
aligned symbol quoted above.

## Reassessment of the old counterexample

The old clock-only block at `r=2`, `lambda_D=+1`, and `K2=2` has

$$
\bar A_4=-0.0785562607610
$$

and vanishes at `k_bar=3.56787473663`. The corrected calculation deliberately
reproduces that incomplete zero and then adds the missing terms. At the same
background and wave number:

- the Class-Ia DHOST-plus-Einstein sum is zero to numerical precision;
- the AeST Maxwell coefficient is `K_B/r^2=0.25`;
- the full positive core is greater than 8, not zero.

Thus the old point is a regression fixture demonstrating why the correction
matters, not a physical counterexample.

## Executable audit

The corrected audit evaluates 50,000 signed clock and wave-number backgrounds
over `10^-6` through `10^6` for both `lambda_D=+1` and `lambda_D=-1`. It checks:

1. exact reproduction of the superseded clock-block zero;
2. explicit incomplete-projection labeling of that block;
3. Class-Ia cancellation after adding the metric null direction;
4. removal of the old counterexample in the complete symbol;
5. positive complete symbols for both sign sentinels; and
6. a positive AeST Maxwell coefficient.

The full test suite passes. No astronomical data or holdout was opened.

## Remaining kill gate

This correction is exact only on the constant, zero-momentum, aligned branch.
It does not establish the sign of the full principal matrix when the scalar
has a spatial first gradient, the aether is tilted, the wave vector has an
arbitrary orientation, or tensor/vector metric and aether directions mix.
Those components must be projected onto the complete Class-Ia/AeST constraint
surface before any sign conclusion is valid.

The next gate therefore scans both orientation signs; the aligned calculation
does not select one. The theory remains explicitly marked not viable.

The covariant degeneracy and primary-null construction follow the published
[DHOST Hamiltonian analysis](https://arxiv.org/abs/1512.06820). The remaining
positive lapse/aether gradient follows the published
[AeST ADM action](https://arxiv.org/abs/2307.15126). The explicit correction
and cancellation audit are project calculations; no novelty claim is made.

## Reproduction

```powershell
$env:PYTHONPATH='src'
python scripts/audit_sigma_v12a_aligned_finite_k.py
python -m pytest -q tests/test_sigma_v12a_aligned_finite_k.py
```

Machine-readable evidence is in
`results/sigma_v12a_aligned_finite_k/report.json`.
