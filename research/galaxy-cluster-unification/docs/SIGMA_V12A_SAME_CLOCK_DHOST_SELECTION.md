# Sigma v12A same-clock DHOST mechanism selection

## Decision

Sigma v12A passes a narrow, theory-only mechanism-selection gate. It advances
only to complete covariant variation and a joint AeST--DHOST Hamiltonian audit.
It is not a viable theory and is not authorized to see astronomical data.

The post-v10 material-memory family is closed. V12A does not add another
material coordinate, auxiliary memory, tensor carrier, response multiplier, or
homogeneous halo state. It uses the scalar already present in AeST and places
its Hessian inside the exact luminal Class-Ia DHOST degeneracy relations.

## Why this lane remains after the reset

The mechanism audit gives the following disposition:

| Mechanism | Existing result | Disposition |
|---|---|---|
| Positive massive spin two | Healthy exchange cannot supply the required universal infrared amplitude; scalar proxy does not define lensing | closed v7 lane |
| Cubic AeST/Horndeski | Geometry-sensitive, but order-unity response becomes superluminal | closed v8A interaction |
| Retarded inverse operators | Causal effective rule, but minimal fundamental localization has six negative kinetic directions | closed v6 fundamental lane |
| Aether-tidal tensor carrier | Instantaneous tail, finite-amplitude ghost, then widened tensor cone | closed v10 lane |
| Independent scalar/material memory | Three distinct finite-background kinetic-rank failures | closed v11 lane |
| Same AeST scalar plus exact DHOST degeneracy | No new state; galaxy exterior remains AeST; directional Hessian terms available | selected v12A lane |

This explicitly retains a MOND/AeST-like galaxy limit. An earlier project
closure that prohibited any MOND/AQUAL galaxy equation does not apply to this
goal: the present target permits MOND-level galaxy performance and asks whether
one universal theory can also predict cluster lensing.

## Covariant action

Retain the published one-metric AeST action and scalar `phi`. Define

$$
X=g^{\mu\nu}\nabla_\mu\phi\nabla_\nu\phi,
\qquad
\phi_{\mu\nu}=\nabla_\mu\nabla_\nu\phi.
$$

The quadratic DHOST basis required here is

$$
L_3=(\Box\phi)\phi^\mu\phi_{\mu\nu}\phi^\nu,
$$

$$
L_4=\phi^\mu\phi_{\mu\rho}\phi^{\rho\nu}\phi_\nu,
\qquad
L_5=(\phi^\mu\phi_{\mu\nu}\phi^\nu)^2.
$$

The selected envelope is

$$
\boxed{
S_{12A}=S_{\rm AeST}
+\int d^4x\sqrt{-g}\,[A_3L_3+A_4L_4+A_5L_5].
}
$$

With the Einstein coefficient `F0>0` constant, exact luminal Class-Ia
degeneracy requires

$$
A_1=A_2=0,
$$

$$
\boxed{
A_4=-A_3-{X^2A_3^2\over8F_0},
\qquad
A_5={XA_3^2\over2F_0}.
}
$$

These identities are published DHOST physics, not a Sigma invention. They
remove the extra higher-derivative scalar mode algebraically and set the tensor
front to the matter-metric light cone.

## Background-zero coefficient

Let

$$
q_\Sigma={a_\Sigma\over c^2},
\qquad
x={X\over q_\Sigma^2},
\qquad
x_0=-{Q_0^2\over q_\Sigma^2},
\qquad
d=x-x_0.
$$

Choose the provisional theory-driven shape

$$
\mathcal A(x,x_0)
={d^2\over(1+d^2)^{3/2}\sqrt{1+x^2}},
$$

and

$$
\boxed{A_3={\lambda_DF_0\over q_\Sigma^4}\mathcal A(x,x_0).}
$$

The reasons for this shape precede any data:

1. `A3` and its first derivative vanish on the flat AeST clock background;
2. it is smooth on timelike and spacelike branches;
3. the total-`x` factor bounds the `X^2 A3^2` dependent combination; and
4. on the static branch `d=Y/q_sigma^2=(g/a_sigma)^2`, so it is suppressed by
   far more than `10^-5` at the declared Solar high-acceleration boundary.

The provisional constants are

$$
\boxed{\{a_\Sigma,\mu_\Sigma,K_B,K_2,\lambda_D\}.}
$$

There is no sixth length, cluster label, lensing amplitude, or object-specific
state. `Q0` is derived from the AeST constants as before.

## Flat-vacuum selection result

At the fixed AeST construction row

$$
K_B=1,\qquad K_2=2,\qquad\lambda_s=1,
$$

the published finite-frequency squared speeds remain

| mode | squared speed |
|---|---:|
| tensor | `1` |
| vector | `1` |
| scalar | `3/4` |

Because the v12A coefficient and its first derivative vanish at `X=X0`, the
new interaction does not enter that quadratic spectrum. The published AeST
zero-frequency Jeans-like warning remains; this selection does not relabel it
as a pass.

Across the signed coefficient scan and three representative clock-background
ratios, the normalized Class-Ia residuals are below `10^-12`, every normalized
coefficient is finite, and `A1=A2=0` keeps the added tensor contribution
luminal. This is an algebraic selection result, not yet a proof that the
combined AeST and DHOST constraint algebras coexist.

## Directional capability

On a locally Cartesian static background, with scalar spatial gradient `g`
and Hessian `H`, the operators reduce to

$$
L_3=(\operatorname{tr}H)(g^THg),
\qquad
L_4=g^TH^2g,
\qquad
L_5=(g^THg)^2.
$$

For `g=(1,0,0)`, compare

$$
H_{\rm iso}=\operatorname{diag}(1,1,1),
\qquad
H_{\rm rank1}=\operatorname{diag}(3,0,0).
$$

Both have trace three, but their `(L3,L4,L5)` values are `(3,1,1)` and
`(9,9,9)`. Thus the interaction distinguishes how the same scalar curvature is
distributed among directions. Random common rotations preserve all three
invariants to the frozen numerical tolerance.

This is the needed capability for cluster topology: it can respond differently
to a smooth central field and to aligned, overlapping member-galaxy fields
without asking whether the object is a galaxy or cluster. It does not prove the
response has the correct sign or amplitude.

## Prior-art boundary

AeST and its nonlinear Hamiltonian structure are published work; see the
[original one-metric theory](https://arxiv.org/abs/2007.00082) and its
[Hamiltonian formulation](https://arxiv.org/abs/2307.15126). Quadratic DHOST
classification and the absence of the Ostrogradsky mode by degeneracy are also
established; see [Ben Achour, Langlois, and Noui](https://arxiv.org/abs/1602.08398)
and the luminal subclass relations in
[Langlois et al.](https://arxiv.org/abs/1711.07403).

The search performed for this selection found no paper explicitly analyzing
this AeST-plus-background-zero Class-Ia combination. That is not proof of
originality. The only potentially project-specific content is the coefficient
shape, the exact combination with AeST, and—if it survives—the resulting
baryon-shaped lensing prediction.

## Decisive next gate

The shared metric makes the next calculation non-optional. Adding two
individually healthy actions does not prove their constraints remain healthy
together. Before observations, v12A must:

1. perform the full `3+1` decomposition using `B_mu=nabla_mu phi` as an
   auxiliary first-derivative variable;
2. combine the published AeST metric/aether Hessian with the Class-Ia
   metric/scalar Hessian;
3. prove the DHOST primary-secondary constraint pair survives arbitrary aether
   tilt and nonzero scalar gradient;
4. retain the AeST four first-class and four second-class constraints with no
   additional physical mode;
5. derive the complete metric, scalar, vector, and multiplier equations and
   their diffeomorphism identity;
6. compute physical characteristics on tilted and anisotropic backgrounds;
7. require positive energy and every cone inside the one matter-metric cone.

Any rank change, ghost, superluminal cone, or loss of the published constraint
count retires the exact row before Solar, galaxy, or cluster calculations. No
observational product or holdout was opened.

## Reproduction

```powershell
$env:PYTHONPATH='src'
python scripts/audit_sigma_v12a_same_clock_dhost_selection.py
python -m pytest -q tests/test_sigma_v12a_same_clock_dhost.py
```

Machine-readable evidence is in
`results/sigma_v12a_same_clock_dhost_selection/report.json`.
