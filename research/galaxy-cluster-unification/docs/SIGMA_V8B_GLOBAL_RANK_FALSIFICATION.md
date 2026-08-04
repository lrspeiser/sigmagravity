# Sigma v8B global Legendre-rank falsification

## Verdict

The exact Sigma v8B action is **retired before observational use**. Its
preferred-time completion has a globally singular velocity Legendre map for
every nonzero completion coupling. The failure is visible in two independent
ways:

1. At the frozen $K_B=1$, large scalar velocity forces a finite determinant
   zero above a finite, subluminal aether tilt.
2. Raising $K_B$ can reverse that large-velocity sign, but cannot rescue the
   action. On a generic tilted background the determinant is exactly affine in
   isotropic extrinsic curvature and crosses zero at finite $K$ throughout the
   healthy open interval $0<K_B<2$.

At each representative crossing, the Lagrangian, canonical energy, and all
momenta are finite. The kinetic inertia gains a second negative direction. No
observational array or held-out target was opened.

## Why the bounded scan looked healthy

The preceding tilted-ADM gate scanned $v_A\le0.9$, $0.5\le Q/Q_0\le1.5$,
$L_HQ_0\le1$, and $|K|/Q_0,|E|/Q_0\le0.1$. All 1,409 points were regular. That
result remains correct for that bounded patch.

The global calculation finds the frozen-row critical tilt at

$$
v_{A,*}=0.9020884486,
$$

only slightly beyond the scanned maximum. More importantly, the curvature
surface exists even at ordinary $v_A=0.5$ once $K/Q_0$ is allowed to reach its
finite root. A healthy bounded patch therefore does not extend to a globally
regular theory.

## Large-$Q$ analytic no-go for the frozen row

At zero background $K_{ij}$ and $E_i$, write the ten-velocity Hessian in block
form,

$$
\mathcal H=
\begin{pmatrix}
A & b\\
b^T & d
\end{pmatrix},
$$

where $A$ covers the six metric and three aether velocities and $d$ is the
scalar-velocity entry. At large scalar velocity,

$$
b=-C\sigma^2u+O(\sigma),
\qquad
C=(\alpha-1)L_H^2,
$$

with $x=A_iA^i$ and

$$
u=\left(
3+5x,
3+4x+x^2,
3+4x+x^2,
0,0,0,
2\sqrt{x}(1+2x),
0,0
\right).
$$

The automatic-differentiation Hessian agrees with this analytic leading vector
to relative error below $3\times10^{-5}$ at $Q/Q_0=10^5$.

The scalar Schur complement contains

$$
d-b^TA^{-1}b
=-C^2\sigma^4\mathcal S(x,K_B)+O(\sigma^3),
$$

where

$$
\boxed{
\mathcal S(x,K_B)=
\frac{
(32-20K_B)x^3
+(32-67K_B)x^2
+(8-78K_B)x
-27K_B
}{4K_B}.
}
$$

For the frozen $K_B=1$, $\mathcal S$ changes sign at

$$
x_*=4.3695187102,
\qquad
v_{A,*}=\sqrt{\frac{x_*}{1+x_*}}=0.9020884486.
$$

Above this tilt, every $C\neq0$ makes the Schur complement negative at
sufficiently large but finite $Q$. At $Q=Q_0$ the completion Hessian reduces to
the regular base result. Continuity therefore requires a finite rank-zero
surface.

At $v_A=0.97$, $a_\Sigma/Q_0=1$, the exact roots are:

| $L_HQ_0$ | First $Q/Q_0$ rank surface |
|---:|---:|
| 0.25 | 9.664817 |
| 0.50 | 5.133731 |
| 1.00 | 2.864943 |
| 2.00 | 1.740328 |

Changing $a_\Sigma/Q_0$ from $10^{-4}$ to $100$ shifts the unit-length root
only from `2.859899` to `2.923798`. The acceleration scale is not an escape.

At the reference root the null-mode power is `51.92%` metric, `47.99%` aether,
and `0.096%` scalar. It is not the pure GR conformal direction. The canonical
energy is finite, `14.5741` in audit units, and the raw inertia changes from
`(1,0,9)` to `(2,0,8)`.

## Exact curvature no-go closes the high-$K_B$ escape

For $K_B\ge1.6$, every coefficient in $\mathcal S$ is nonpositive, so the
large-$Q$ argument alone no longer forces a zero. This was tested as a genuine
theory-only escape, not dismissed by assumption. The escape fails for a more
general reason.

Set

$$
K_{ij}=\frac{\kappa}{3}q_{ij},
\qquad E_i=0.
$$

The first-order completion from the tilted gate becomes

$$
\mathcal L_C^{(1)}=-C\sqrt q\,\kappa\,G(x,\sigma),
$$

where

$$
G=F+\sigma B^2
+\frac{x}{3}\left(\sigma B^2-2F_{,x}\right),
\qquad
F=\frac{xB^3}{3\chi}.
$$

Consequently only the scalar diagonal entry gains a background-curvature term,

$$
\mathcal H_{\sigma\sigma}(\kappa)
=\mathcal H_{\sigma\sigma}(0)-C\kappa G_{,\sigma\sigma},
$$

while $A$ and $b$ are independent of $\kappa$. The determinant is therefore
exactly affine:

$$
\det\mathcal H(\kappa)
=\det A\left[
d_0-b^TA^{-1}b-C\kappa G_{,\sigma\sigma}
\right].
$$

For $0<K_B<2$, the base block $A$ is nonsingular. At the generic audit point
$v_A=0.5$, $Q/Q_0=1.2$, $L_HQ_0=1$, one has
$G_{,\sigma\sigma}\neq0$. Thus every $C\neq0$ has the finite root

$$
\kappa_*=\frac{d_0-b^TA^{-1}b}{C G_{,\sigma\sigma}}.
$$

The numerical determinant satisfies the affine identity at $\kappa=2$ below
$2\times10^{-15}$ across the scan:

| $K_B$ | Derived $c_s^2$ | Derived $\alpha$ | $K/Q_0$ rank surface |
|---:|---:|---:|---:|
| 1.00 | 0.750000 | 1.777778 | 2.899272 |
| 1.60 | 0.225000 | 1.911589 | 2.804103 |
| 1.70 | 0.163235 | 2.440402 | 1.848835 |
| 1.80 | 0.105556 | 3.530566 | 1.164108 |
| 1.95 | 0.025321 | 13.506550 | 0.933690 |

Every crossing has finite momenta and changes the raw inertia from `(1,0,9)`
to `(2,0,8)`. The higher-$K_B$ rows are healthy in the published flat linear
spectrum but are not globally regular nonlinear theories with this completion.

## Why this is a retirement result

A full Dirac analysis could label the zero surface as an additional primary
constraint, but that would make the constraint and degree-of-freedom count
change across field space. It cannot make the demonstrated Legendre map
globally regular. Avoidance would require a separately derived invariant domain
restriction proving that no allowed initial data or baryonic solution can reach
the surface. The exact action supplies no such restriction, and the crossings
occur at finite timelike tilt, scalar velocity, curvature, Lagrangian, energy,
and momenta.

The zero-coupling limits do not preserve the proposed theory:

- $L_H=0$ removes both the cubic geometry response and its causal partner,
  leaving the AeST base without the proposed cluster-Hessian mechanism.
- $\alpha=1$ retains the nonzero v8A cubic while removing its causal partner,
  restoring the already-demonstrated superluminal nonlinear scalar cone.

Changing $K_B$, $a_\Sigma$, $L_H$, or $\alpha$ therefore does not rescue exact
v8B. Moving the surface after observational fitting is explicitly forbidden.

## Lesson for the successor

The desired static Hessian response cannot be made healthy by appending a term
that is linear in ADM curvature but nonlinearly dependent on the scalar clock.
The successor must instead satisfy a degeneracy identity on arbitrary tilted,
time-dependent backgrounds. Suitable theory-only directions are:

1. a complete degenerate scalar-vector-tensor operator whose lapse, metric,
   aether, and scalar terms cancel the dangerous Schur complement identically;
2. a bounded auxiliary field with its own positive kinetic term and a unique
   baryon-forced state, rather than a higher-derivative clock completion;
3. a nonlocal causal response represented by a healthy localized carrier,
   provided its state is fixed by baryons and universal retarded boundaries.

No new observational holdout should be opened until one of these alternatives
passes global degeneracy, characteristic, source-uniqueness, and Solar gates.

## Reproduction

```powershell
python scripts/audit_sigma_v8b_global_rank.py
python -m pytest tests/test_sigma_v8b_global_rank.py
```

The frozen protocol is
[`../configs/sigma_v8b_global_rank_falsification.json`](../configs/sigma_v8b_global_rank_falsification.json),
and the machine-readable report is
[`../results/sigma_v8b_global_rank_falsification/report.json`](../results/sigma_v8b_global_rank_falsification/report.json).
