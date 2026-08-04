# Sigma v10C hyperbolic aether-tidal selection

> **Superseded gate result (2026-08-04):** exact v10C fails its nonlinear
> kinetic gate and is retired before data. On a nonzero spatial carrier
> background, the physical aether-vector kinetic matrix is
> `K_B I-beta P`; it crosses zero at the finite isotropic amplitude
> `P=sqrt(11 K_B/2) I` and becomes negative beyond it. See
> [`SIGMA_V10C_NONLINEAR_KINETIC_FALSIFICATION.md`](SIGMA_V10C_NONLINEAR_KINETIC_FALSIFICATION.md).

## Decision

Sigma v10C passes a narrow theory-only selection gate and advances to complete
covariant variation, nonlinear ADM constraint counting, and arbitrary-
background characteristic analysis.  No astronomical fitting is authorized.

V10C preserves the positive static aether-tidal block discovered in v10B while
replacing its instantaneous auxiliary constraint with a hyperbolic tensor.  Its
speed and mixing are not independently fitted numbers: they are the unique
solution of the declared threefold-capacity and luminal-upper-cone equations at
the frozen AeST scalar speed.

At the selected point:

- carrier squared speed: $c_P^2=3/11$;
- normalized mixing: $\beta^2/K_B=2/11$;
- worst static determinant: $1/11$;
- worst static Schur complement: $1/3$;
- longitudinal mixed squared speeds: $9/44$ and $1$;
- transverse mixed squared speeds: `0.232009` and `0.881627`;
- unmixed carrier squared speed: $3/11$;
- physical flat TT squared speed: $1$;
- longitudinal static response capacity: `3`;
- spent unit-to-cluster amplitude-gap closure: `93.26%`.

These results establish neither nonlinear viability nor observational
adequacy.  Full metric/aether/scalar/P equations, tilted and inhomogeneous
cones, stress-energy, PPN, Solar screening, cosmology, and a convergent PDE
solver remain mandatory.

## Construction logic

The previous two results leave complementary lessons:

1. V10A's propagating tensor coupled to the MOND scalar failed because the
   scalar's static stiffness vanishes in the deep regime.
2. V10B's auxiliary tensor coupled to the constant-stiffness aether
   acceleration had positive energy and useful geometry, but its finite-range
   elliptic constraint produced a physical equal-time tail.

V10C retains the v10B source

$$
J_\mu=A^\nu\nabla_\nu A_\mu,
\qquad
H_{\mu\nu}=D_{(\mu}J_{\nu)},
$$

and restores a positive time kinetic term for the six-component symmetric
spatial polarization $P_{\mu\nu}$.

The source remains physically appropriate.  In static unitary gauge,

$$
J_i=D_i\ln N=D_i\Psi+O(2),
$$

so

$$
H_{ij}=D_iD_j\Psi+O(2).
$$

It contains isotropic curvature, tidal eigenvalues, and shear orientation,
while distinguishing equal-$M/r^2$ systems through $M/r^3$.

## Frozen action

Define the aether-spatial projector and spatial aether field strength

$$
q_{\mu\nu}=g_{\mu\nu}+A_\mu A_\nu,
$$

$$
B_{\mu\nu}=q_\mu{}^\alpha q_\nu{}^\beta F_{\alpha\beta},
\qquad
F_{\alpha\beta}=2\nabla_{[\alpha}A_{\beta]}.
$$

The selected addition is

$$
\begin{aligned}
\Delta\mathcal L_{10C}={}&
{1\over2}\dot P_{\mu\nu}\dot P^{\mu\nu}
-{c_P^2\over2}D_\lambda P_{\mu\nu}D^\lambda P^{\mu\nu}\\
&-{1\over L_P^2}\left[
{P_{\mu\nu}P^{\mu\nu}\over2}
+{(P_{\mu\nu}P^{\mu\nu})^2\over4}
\right]\\
&+\beta P^{\mu\nu}D_{(\mu}J_{\nu)}
+{K_B(1-u)\over2}B_{\mu\nu}B^{\mu\nu}
+\zeta^\nu A^\mu P_{\mu\nu}.
\end{aligned}
$$

The last aether term reduces the bare transverse-aether spatial coefficient
without changing its positive time kinetic coefficient.  At the frozen point,
it makes the bare vector speed equal to the existing AeST scalar speed.

Because $P^{\mu\nu}$ is symmetric and spatial,

$$
P^{\mu\nu}D_{(\mu}J_{\nu)}
=P^{\mu\nu}\nabla_\mu J_\nu.
$$

Up to a covariant boundary term,

$$
\int\sqrt{-g}\,P^{\mu\nu}\nabla_\mu J_\nu
=-\int\sqrt{-g}\,(\nabla_\mu P^{\mu\nu})J_\nu.
$$

The first-order form contains only first derivatives of $P$, $A$, and the
metric.  A complete variation is still required because the projectors,
spatiality constraint, and connections create nonlinear kinetic mixing.

The physical constants remain

$$
\{a_\Sigma,\mu_\Sigma,K_B,K_2,L_P\},
$$

with no object-specific gravity state, lens-only parameter, or object label.

## Deriving the coefficients

Let

$$
u={3\over4}
$$

be the largest squared speed among the sourced frozen AeST scalar and modified
vector channels.  Let

$$
s=c_P^2,
\qquad
q={\beta^2\over K_B}.
$$

To retain v10B's longitudinal static capacity of three, require

$$
{q\over s}={2\over3}.
$$

The high-frequency mixed characteristic equation is

$$
(u-y)(s-y)-qy=0,
\qquad
y={\omega^2\over k^2}.
$$

The upper root is no greater than the physical metric cone precisely when

$$
q\le(1-u)(1-s).
$$

V10C fixes the fastest root at the boundary rather than leaving a free cone
coefficient:

$$
q=(1-u)(1-s).
$$

Solving the two equations gives

$$
\boxed{s={3\over11}},
\qquad
\boxed{q={2\over11}},
\qquad
\boxed{\beta=\sqrt{{2K_B\over11}}}.
$$

The aether magnetic counterterm fraction is

$$
1-u={1\over4}.
$$

All of these coefficients are derived from the frozen base row and the
declared capacity/cone requirements.  None is a sixth fitted constant.

## Static positivity

The worst longitudinal static principal matrix at $K_B=1$ is

$$
K_{\rm static,L}=
\begin{pmatrix}
1&-\sqrt{2/11}\\
-\sqrt{2/11}&3/11
\end{pmatrix}.
$$

It has determinant

$$
\det K_{\rm static,L}={1\over11}
$$

and eigenvalues

$$
{7-\sqrt{38}\over11}=0.0759624,
\qquad
{7+\sqrt{38}\over11}=1.196765.
$$

The aether Schur complement is

$$
1-{(2/11)\over(3/11)}={1\over3},
$$

so the longitudinal high-$k$ response capacity is exactly three.

The canonical transverse mixing is smaller by $1/\sqrt2$.  Its static
eigenvalues are `0.163986` and `1.108741`, its determinant is `2/11`, and its
response capacity is `1.5`.

The convex potential has five Hessian eigenvalues $1+p^2$ and one
$1+3p^2$.  Combined with the positive gradient block and a zero static boundary
condition, it gives one stationary carrier profile for fixed sources.

## Flat characteristic cones

### Longitudinal sourced channel

Substitution of $u=3/4$, $s=3/11$, and $q=2/11$ gives

$$
y_-={9\over44}=0.204545,
\qquad
y_+=1.
$$

The positivity margin is

$$
us-q={1\over44}>0,
$$

and the upper-cone margin is exactly zero by construction.

### Transverse sourced channels

The canonical off-diagonal tensor normalization changes the mixing to
$q_T=q/2=1/11$.  The roots are

$$
y_{T\pm}={49\pm\sqrt{817}\over88},
$$

or

$$
0.232009,
\qquad0.881627.
$$

Both are strictly inside the metric cone.

### Unmixed modes

The remaining polarization modes have

$$
c_P^2={3\over11}.
$$

A pure flat TT metric perturbation has $J_i=0$ and does not source $P$.  The
Einstein-Hilbert tensor cone therefore remains

$$
c_T^2=1
$$

at this background and perturbative order.

These calculations are necessary local principal checks.  The zero margin in
the longitudinal channel makes arbitrary-background analysis especially
important: any nonlinear coefficient shift toward a wider cone retires the
exact construction.

## Causal and source boundary condition

Unlike v10B, the $P$ equation contains a positive second time derivative.  Its
mass and self-interaction are lower-order terms; the front is fixed by the
hyperbolic principal cone above.  A localized source therefore uses a retarded
Green function rather than an equal-time elliptic Yukawa kernel.

For static sources, strict convexity and $P\to0$ at spatial infinity select one
stationary solution.  Dynamically, the universal rule is retarded/no incoming
carrier radiation.  The theory permits physical free $P$ waves, but they are
universal initial data, not a separately chosen halo profile for each object.

Global nonlinear well-posedness and cosmological initial data are not yet
proved.

## Linear physical-metric projection

The frozen static interaction remains

$$
\Delta\mathcal L^{(2)}
=\beta P_{ij}\partial_i\partial_j\Psi.
$$

It changes the lapse equation by

$$
\beta\partial_i\partial_jP_{ij}.
$$

At this order the traceless spatial metric equation is unchanged.  On the AeST
no-slip branch,

$$
\delta\Psi=\delta\Phi=\delta W,
\qquad
W={\Psi+\Phi\over2}.
$$

Thus the same metric correction affects slow massive bodies and photons.  This
is a linear static identity, not the complete nonlinear stress-energy or Weyl
derivation.

## Geometry retained

The local convex-response probes at the selected mixing give:

| Probe | Result |
|---|---:|
| Isotropic response trace | `0.972563` |
| Isotropic STF norm | `0` |
| Rank-one STF norm | `0.643941` |
| Spherical exterior tidal norm | `0.700593` |

The construction therefore retains nonzero monopole/trace response and
orientation-preserving shear information while escaping the v9B local-force
theorem.

## What can still kill v10C

V10C must be retired before data if any of the following occurs:

1. the complete spatiality/multiplier/metric/aether constraint algebra changes
   rank or exposes a negative mode;
2. a tilted, static-gradient, nonzero-$P$, separated-source, or FLRW background
   widens any characteristic beyond the physical metric cone;
3. nonlinear projection changes the physical TT speed or destroys the one-
   metric relation between dynamics and Weyl lensing;
4. the retarded/no-incoming prescription fails to select a unique physical
   branch;
5. Solar, PPN, compact-source, or cosmological stability fails with the same
   five constants;
6. a convergent PDE implementation cannot meet the existing numerical gates.

No observation may be used to adjust $3/11$, $2/11$, or the magnetic
counterterm.

## Prior-art boundary

The AeST base and its scalar/vector modes are established by
[Skordis and Zlosnik](https://arxiv.org/abs/2007.00082) and their
[Minkowski stability analysis](https://arxiv.org/abs/2109.13287).  General
aether interactions containing acceleration, shear, vorticity, second
derivatives, and curvature are broad effective-field prior art; see
[Balakin and Lemos](https://arxiv.org/abs/1407.6014).  Preferred-foliation
constraint and degeneracy analysis is also an established field, including
[Gao and Yao](https://arxiv.org/abs/1910.13995).

The fixed `3/11`, `2/11` combination and proposed use as a baryonic
trace/shear lensing response are project-specific hypotheses.  No novelty claim
is made before a much broader prior-art audit and survival of the full action.

## Reproduction

```powershell
$env:PYTHONPATH='src'
python scripts/audit_sigma_v10c_hyperbolic_aether_tidal.py
python -m pytest -q tests/test_sigma_v10c_hyperbolic_aether_tidal.py
```

Machine-readable evidence is in
`results/sigma_v10c_hyperbolic_aether_tidal_selection/report.json`.

No observational product or holdout was opened.
