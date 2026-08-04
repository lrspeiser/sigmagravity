# Sigma v10B auxiliary aether-tidal selection and falsification

## Verdict

Sigma v10B repairs the exact static sign failure that retired v10A, preserves
the useful trace-plus-shear geometry, and removes all six polarization
components through a healthy second-class constraint system.  It is
nevertheless **retired before observational fitting** because its finite-range
auxiliary constraint produces a physical equal-preferred-time Yukawa tail.

The distinction is important:

- v10B has a positive static principal block;
- its carrier potential is strictly convex;
- fixed sources and zero boundary data select a unique carrier state;
- the carrier adds no initial data or propagating frequency root;
- its reduced flat Hamiltonian is positive;
- but eliminating it transmits a nonzero physical response to every spatial
  radius on the same aether-time slice.

No galaxy, cluster, map, or holdout was opened.  The only phenomenological
number used was the already-spent scalar amplitude target `3.14465`.

## Why v10B was attempted

V10A coupled the spatial tensor to the Hessian of the deep-MOND scalar.  That
scalar's transverse and longitudinal constitutive stiffnesses both vanish as
the field tends to zero.  Every nonzero constant derivative mixing therefore
made the high-wave-number static determinant negative.

The AeST aether has another geometrically meaningful vector,

$$
J_\mu=A^\nu\nabla_\nu A_\mu,
$$

the four-acceleration of the preferred timelike congruence.  Its Maxwell-type
principal stiffness is the positive constant $K_B$, rather than a MOND
coefficient that vanishes in the target regime.

On a static slice aligned with the aether,

$$
J_i=D_i\ln N=D_i\Psi+O(2).
$$

Consequently,

$$
D_{(i}J_{j)}=D_iD_j\Psi+O(2)
$$

contains exactly the spatial information the project needs:

- trace for isotropic convergence;
- traceless components for tidal shear;
- eigenvectors for orientation;
- $M/r^3$ scale information that distinguishes systems with the same
  $M/r^2$.

## Frozen action

The selected addition to one-metric AeST was

$$
\begin{aligned}
\Delta\mathcal L_{10B}={}&
-{1\over2}D_\lambda P_{\mu\nu}D^\lambda P^{\mu\nu}\\
&-{1\over L_P^2}\left[
{P_{\mu\nu}P^{\mu\nu}\over2}
+{(P_{\mu\nu}P^{\mu\nu})^2\over4}
\right]\\
&+\beta P^{\mu\nu}D_{(\mu}J_{\nu)}
+\zeta^\nu A^\mu P_{\mu\nu},
\end{aligned}
$$

with

$$
P_{\mu\nu}=P_{\nu\mu},
\qquad
A^\mu P_{\mu\nu}=0.
$$

Unlike v10A, there is no $\dot P^2$ term.  $P_{\mu\nu}$ is intended to be a
spatial elliptic polarization constraint, not a freely specifiable halo or a
new propagating particle.

The field normalization fixes the spatial-gradient coefficient to one.  The
mixing prescription is

$$
\boxed{\beta^2={2K_B\over3}}.
$$

At $K_B=1$,

$$
\beta=\sqrt{2\over3}=0.816497.
$$

This rational condition was selected from the principal matrix, not fitted to
an object.  It leaves a worst-channel Schur complement $K_B/3$ and a maximum
longitudinal response of three.  The physical constant budget remains

$$
\{a_\Sigma,\mu_\Sigma,K_B,K_2,L_P\}.
$$

There are no object-specific field parameters, lens-only coefficients, or
galaxy/cluster labels.

## Exact static principal block

After integration by parts and freezing the local projectors, the principal
Euclidean energy is

$$
E_{\rm pr}=
{K_B\over2}J_iJ_i
+{1\over2}D_kP_{ij}D_kP_{ij}
-\beta J_jD_iP_{ij}.
$$

For a wave along $x$, the worst longitudinal component $P_{xx}$ has block

$$
K_L=
\begin{pmatrix}
1&-\sqrt{2/3}\\
-\sqrt{2/3}&1
\end{pmatrix}.
$$

Its eigenvalues are

$$
0.183503,
\qquad1.816497,
$$

and its determinant is exactly $1/3$.  The canonical transverse off-diagonal
component has mixing $\beta/\sqrt2=1/\sqrt3$, giving eigenvalues

$$
0.422650,
\qquad1.577350,
$$

and determinant $2/3$.  Unmixed components have unit stiffness.

The general divergence inequality

$$
|n_iP_{ij}|\le |P|_F
$$

makes the longitudinal block the global frozen-projector worst case.  A
deterministic 256-tensor check reached maximum ratio `0.963681`; the analytic
operator norm is one and is saturated by $P=n\otimes n$.

This repairs v10A's deep-field failure because $K_B$ does not vanish with the
AQUAL scalar stiffness.

## Response capacity and geometry

In the linear static limit, solving the finite-range carrier gives

$$
K_{\rm eff}(k)=
K_B-b^2{k^2\over k^2+L_P^{-2}},
$$

where $b=\beta$ in the longitudinal channel and
$b=\beta/\sqrt2$ transversely.  The response relative to the unmixed channel is

$$
\mathcal A(k)={K_B\over K_{\rm eff}(k)}.
$$

The asymptotic capacities are

| Channel | Capacity |
|---|---:|
| Longitudinal | `3.0` |
| Transverse | `1.5` |
| Unmixed | `1.0` |

The longitudinal value closes

$$
{3-1\over3.14465-1}=0.932553
$$

or `93.26%` of the already-spent unit-to-cluster amplitude gap.  The 75% gap
target is reached at $kL_P=3.51072$.  These are channel-capacity checks, not an
astronomical prediction.

The same strictly convex potential used in v10A gives five local Hessian
eigenvalues $1+p^2$ and one $1+3p^2$.  The frozen geometry probes retain:

- nonzero isotropic trace `1.44460`;
- nonzero rank-one STF norm `0.901398`;
- nonzero spherical exterior tidal norm `1.0`.

## Linear one-metric projection

Around the static flat branch,

$$
\Delta\mathcal L^{(2)}
=\beta P_{ij}\partial_i\partial_j\Psi.
$$

Variation with respect to the lapse potential adds

$$
\beta\partial_i\partial_jP_{ij}
$$

to the linear $\Psi$ equation.  Dependence on the spatial metric multiplies
the already first-order product $P\Psi$ and begins one order higher, so the
linear traceless spatial equation is unchanged.  On the AeST no-slip branch,

$$
\delta\Psi=\delta\Phi=\delta W,
\qquad
W={\Psi+\Phi\over2}.
$$

Thus the construction is not a photon-only rule: the same physical metric
changes slow-body dynamics and lensing at this order.  A pure flat TT wave has
$J_i=0$, so the new source does not change the flat tensor light cone at
quadratic order.  Full nonlinear metric variation remained mandatory.

## Dirac constraint result

For one canonical flat Fourier component, write

$$
L={K_B\over2}(\dot a^2-k^2a^2)
-{\Omega^2\over2}p^2
+bkp\dot a,
\qquad
\Omega^2=k^2+L_P^{-2}.
$$

Because $p$ has no velocity,

$$
\pi_p=0
$$

is a primary constraint.  Preserving it gives a secondary constraint.  Their
Poisson bracket has magnitude

$$
\Omega^2+{b^2k^2\over K_B}>0.
$$

Therefore the pair is second class for every finite $k$ and $L_P$.  Repeated
for six spatial tensor components, v10B has:

- six primary constraints;
- six secondary constraints;
- twelve second-class constraints;
- zero additional polarization configuration degrees of freedom.

The reduced Hamiltonian momentum coefficient is positive.  Equivalently,
eliminating $p$ increases the aether time kinetic coefficient:

$$
K_{\rm time}(k)
=K_B+b^2{k^2\over k^2+L_P^{-2}}.
$$

The worst longitudinal vector squared speed decreases smoothly from one to
$3/5$; the transverse value decreases from one to $3/4$.  No superluminal
root or extra frequency appears in this flat block.

## Exact causality failure

The healthy constraint count does not make the new constraint a gauge
constraint.  Eliminating $p$ makes the response of the physical aether
acceleration to a localized source proportional in Fourier space to

$$
C(k)={k^2+m^2\over(K_B+b^2)k^2+K_Bm^2},
\qquad m=L_P^{-1}.
$$

This decomposes in position space into a local delta response plus

$$
{b^2m^2\over(K_B+b^2)^2}
{e^{-m\sqrt{K_B/(K_B+b^2)}r}\over4\pi r}.
$$

The second term is nonzero at every $r>0$ on the same preferred-time slice.
It is exponentially small at large distance, but a causal front must be
exactly zero outside its cone, not merely small.

At the selected row, the longitudinal kernel has

$$
C_{\delta,L}={3\over5},
\qquad
m_{\rm eff,L}=\sqrt{3\over5},L_P^{-1},
\qquad
C_{{\rm tail},L}={6\over25}L_P^{-2}.
$$

The transverse channel has

$$
C_{\delta,T}={3\over4},
\qquad
m_{\rm eff,T}=\sqrt{3\over4},L_P^{-1},
\qquad
C_{{\rm tail},T}={3\over16}L_P^{-2}.
$$

This transverse aether mode is a physical AeST mode, not a lapse-coordinate
artifact.  The $P$ constraints are second class rather than generators of a
gauge symmetry, so there is no first-class cancellation analogous to a
coordinate representation of the GR lapse.  A localized perturbation changes
the transverse acceleration at all distances at the same aether time.

The two ways to remove the tail also remove essential parts of the proposal:

- $\beta=0$ decouples the geometry response;
- $m=0$ removes the finite transition length and the finite-range nonlinear
  screening while leaving an uncontrolled massless auxiliary sector.

Exact finite-range v10B therefore fails the project's causal-propagation gate.

## Prior-art boundary

AeST and its aether acceleration are established in
[Skordis and Zlosnik](https://arxiv.org/abs/2007.00082), with its flat stability
analyzed in their [follow-up](https://arxiv.org/abs/2109.13287).  General
higher-derivative interactions built from aether acceleration, shear,
vorticity, and expansion are broad prior art; an explicit effective-field
catalog appears in
[Balakin and Lemos](https://arxiv.org/abs/1407.6014).  The need to prove both
degeneracy and secondary-constraint consistency in preferred-foliation gravity
is also established in the spatially covariant analyses of
[Gao and Yao](https://arxiv.org/abs/1910.13995) and their
[dynamic-lapse work](https://arxiv.org/abs/1806.02811).

The targeted search did not locate this exact coefficient and auxiliary tensor
combination.  No novelty claim is made: the exact finite-range construction is
already falsified by its causal kernel.

## Next mechanism requirement

The useful pieces remain unusually clear:

1. The aether-acceleration Hessian avoids the deep-AQUAL static no-go.
2. The fixed ratio $\beta^2=2K_B/3$ gives a positive block and useful capacity.
3. Trace plus STF tensor response is still the right representation for
   convergence and shear.
4. The auxiliary implementation must be replaced by a hyperbolic causal
   completion.

The next construction should therefore add a time kinetic completion for
$P_{\mu\nu}$ and derive—not tune—the compensating cone coefficients so that
every mixed scalar/vector/tensor characteristic remains inside the physical
metric cone.  It must retain the positive static Schur complement, the one
metric, five constants, and universal no-incoming boundary conditions.

## Reproduction

```powershell
$env:PYTHONPATH='src'
python scripts/audit_sigma_v10b_auxiliary_aether_tidal.py
python scripts/audit_sigma_v10b_constraint_causality.py
python -m pytest -q tests/test_sigma_v10b_auxiliary_aether_tidal.py
```

Machine-readable evidence is in
`results/sigma_v10b_auxiliary_aether_tidal_selection/report.json` and
`results/sigma_v10b_constraint_causality_gate/report.json`.
