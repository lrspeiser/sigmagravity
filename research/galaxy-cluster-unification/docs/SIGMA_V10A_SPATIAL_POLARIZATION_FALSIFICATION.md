# Sigma v10A spatial-polarization selection and falsification

## Verdict

The six-component aether-spatial polarization tensor is a useful geometry
construction, but the exact v10A constant derivative coupling is **retired
before observational fitting**.

It passes its deliberately narrow selection checks:

- the necessary normalized flat scalar--carrier squared speeds are
  `0.0493061` and `0.950694`;
- the other five carrier polarizations have squared speed `0.25`;
- its fixed-source potential is strictly convex;
- it has nonzero trace and traceless responses, is rotation covariant to
  `1.58e-16`, and is nonlinear under source addition by `34.08%` in the frozen
  two-source probe;
- it distinguishes two spherical points having the same $M/r^2$ because their
  tidal source $M/r^3$ differs by a factor of ten;
- its normalized flat linear response capacity is `4`, above the existing
  spent cluster-amplitude target `3.14465`.

The next exact quasistatic gate fails.  On the simple-$\mu$ AeST branch, the
scalar constitutive stiffness vanishes with acceleration, whereas the selected
mixing remains nonzero.  The high-wave-number static principal matrix then has
a negative eigenvalue.  The carrier mass and convex quartic potential are
order-$k^0$ and cannot repair an order-$k^2$ sign failure.

This result does not retire every spatial tensor carrier.  It closes the exact
constant mixing $\beta P^{\mu\nu}D_{(\mu}S_{\nu)}$ as a healthy completion of
the deep-MOND scalar.

## First-principles motivation

The v9B theorem showed why another local function of acceleration magnitude
cannot unify galaxies and clusters.  In spherical symmetry, every regular
first-gradient completion reduces to one enhancement $E(g_{\rm bar})$.  Yet
the spent development products require galaxy and cluster enhancements that
differ by a median factor `3.231` at almost identical $g_{\rm bar}$.

A Hessian contains the missing spatial-scale information.  Two points with

$$
{M_1\over r_1^2}={M_2\over r_2^2}
$$

can still have

$$
{M_1\over r_1^3}\ne {M_2\over r_2^3}.
$$

A symmetric spatial tensor also has the right representation content:

- its trace can carry an isotropic interior or density response;
- its traceless part carries tidal magnitude and orientation;
- its off-diagonal components can retain shear axes rather than reducing the
  source to one scalar amplitude.

That is a principled response to the raw-lensing failure, where the missing
information is image topology and oriented curvature rather than mean radial
amplitude alone.

## Frozen action

Start with the one-metric AeST action and its unit timelike aether $A^\mu$.
Define

$$
q_{\mu\nu}=g_{\mu\nu}+A_\mu A_\nu,
\qquad
S_\mu=q_\mu{}^\nu\nabla_\nu\phi.
$$

The new field is symmetric and spatial,

$$
P_{\mu\nu}=P_{\nu\mu},
\qquad
A^\mu P_{\mu\nu}=0,
$$

so it has six components on each preferred spatial slice.  Its projected
derivatives are

$$
\dot P_{\mu\nu}
=q_\mu{}^\alpha q_\nu{}^\beta
A^\lambda\nabla_\lambda P_{\alpha\beta},
$$

$$
D_\lambda P_{\mu\nu}
=q_\lambda{}^\rho q_\mu{}^\alpha q_\nu{}^\beta
\nabla_\rho P_{\alpha\beta}.
$$

The frozen addition was

$$
\begin{aligned}
\Delta\mathcal L_P={}&
{1\over2}\dot P_{\mu\nu}\dot P^{\mu\nu}
-{c_P^2\over2}D_\lambda P_{\mu\nu}D^\lambda P^{\mu\nu}\\
&-{1\over L_P^2}\left[
{P_{\mu\nu}P^{\mu\nu}\over2}
+{(P_{\mu\nu}P^{\mu\nu})^2\over4}
\right]
+\beta P^{\mu\nu}D_{(\mu}S_{\nu)}
+\zeta^\nu A^\mu P_{\mu\nu}.
\end{aligned}
$$

The coefficient prescription added no physical constant:

$$
c_P^2=1-c_s^2,
\qquad
\beta={c_s^2\over2}.
$$

At the frozen AeST point

$$
K_B=1,
\qquad K_2=2,
\qquad\lambda_s=1,
\qquad c_s^2={3\over4},
$$

this gives

$$
c_P^2={1\over4},
\qquad
\beta={3\over8}.
$$

The provisional physical constants remained

$$
\{a_\Sigma,\mu_\Sigma,K_B,K_2,L_P\},
$$

with one minimally coupled matter metric, no object labels, no lens-only
coefficient, and no object-specific carrier state.

Integration by parts changes the source interaction schematically to

$$
\int\!\sqrt{-g}\,P^{\mu\nu}D_\mu S_\nu
=-\int\!\sqrt{-g}\,S_\nu D_\mu P^{\mu\nu}
+\hbox{boundary/projector terms}.
$$

This avoids introducing a second time derivative of the scalar in the local
flat preferred frame.  It was not treated as a substitute for the full
covariant constraint variation.

## Necessary flat selection result

For a scalar plane wave, only
$P_L=P_{ij}\hat k_i\hat k_j$ mixes.  With normalized time kinetic terms, the
high-frequency spatial block is

$$
K_{\rm flat}=
\begin{pmatrix}
3/4&3/8\\
3/8&1/4
\end{pmatrix}.
$$

Its determinant is

$$
\det K_{\rm flat}={3\over64}=0.046875,
$$

and its squared speeds are

$$
c_-^2=0.0493061,
\qquad
c_+^2=0.950694.
$$

Both are positive and subluminal.  The other five normalized carrier
polarizations retain $c_P^2=0.25$.  A pure flat transverse-traceless metric
wave has $S_\mu=0$ at linear order, so this source does not by itself alter the
flat TT cone.  The vector/aether and constraint blocks were still unresolved.

Eliminating the massive carrier in this normalized two-field block gives

$$
K_{\rm eff}(k)
=c_s^2-{\beta^2\over c_P^2+(kL_P)^{-2}}.
$$

The corresponding asymptotic response capacity is

$$
{c_s^2\over c_s^2-\beta^2/c_P^2}=4.
$$

The already-spent amplitude `3.14465` occurs at

$$
kL_P=6.33383,
$$

and 75% closure of the unit-to-target gap occurs at `4.30056`.  These are
capacity statements only, not fits and not predictions.

## Convex carrier and geometry checks

For the dimensionless potential

$$
V(P)={P:P\over2}+{(P:P)^2\over4},
$$

the six-dimensional Hessian is

$$
V''=(1+p^2)I+2pp^T.
$$

It has five eigenvalues $1+p^2$ and one eigenvalue $1+3p^2$.  All are positive
for every finite $P$.  Therefore the carrier energy is strictly convex for a
fixed source and fixed boundary data.

In the local algebraic limit,

$$
(1+P:P)P=H,
$$

has one solution, parallel to $H$.  At large source it grows as
$|P|\sim|H|^{1/3}$, so the relative susceptibility falls as $|H|^{-2/3}$.
This offered a possible high-curvature screening mechanism, but it cannot
repair the gradient failure below.

The construction probes give:

| Probe | Result |
|---|---:|
| Isotropic source response trace | `1.60970` |
| Isotropic response STF norm | `0` |
| Traceless spherical-tidal response norm | `1.10398` |
| Rank-one response STF norm | `0.99075` |
| Rotation-covariance relative error | `1.58e-16` |
| Two-source nonadditivity | `0.34085` |

For $(M,r)=(1,1)$ and $(100,10)$, $M/r^2$ is identical while the Hessian norm
ratio is `0.1`.  Thus the source genuinely escapes the v9B local-force theorem.

## Exact quasistatic failure

The flat calculation used the finite-frequency AeST scalar normalization.  The
intended galaxy branch is instead governed quasistatically by

$$
\mu(x)={x\over1+x},
\qquad
x={|S|\over a_\Sigma}.
$$

Linearizing its constitutive flux around a constant spatial field gives

$$
K_T=\mu={x\over1+x}
$$

for perturbations transverse to the background field and

$$
K_L=\mu+x{d\mu\over dx}
={x(x+2)\over(1+x)^2}
$$

for longitudinal perturbations.  For a wavevector making angle $\theta$ with
the background,

$$
K(\theta)=K_T+(K_L-K_T)\cos^2\theta.
$$

The v10A high-$k$ static block is therefore

$$
K_{\rm static}=
\begin{pmatrix}
K(\theta)&\beta\\
\beta&c_P^2
\end{pmatrix}.
$$

Strict ellipticity requires

$$
K(\theta)>{\beta^2\over c_P^2}.
$$

At the selected coefficients,

$$
{\beta^2\over c_P^2}={9\over16}=0.5625.
$$

This yields the analytic thresholds

$$
x_T>{9\over7}=1.285714
$$

and

$$
x_L>{4\over\sqrt7}-1=0.511858.
$$

The entire lower-field interval fails for at least one propagation direction.
At $x=0$ the matrix becomes

$$
\begin{pmatrix}
0&3/8\\
3/8&1/4
\end{pmatrix},
$$

with eigenvalues

$$
-0.270285,
\qquad 0.520285.
$$

The deterministic 10,010-point scan finds 4,979 non-elliptic rows.  More
importantly, the analytic limit is conclusive:

$$
K_T,K_L\longrightarrow0
\quad\hbox{as}\quad x\longrightarrow0.
$$

For every constant $\beta\ne0$, the determinant eventually becomes negative.
Choosing $\beta=0$ is the only globally elliptic member of this exact family,
but it decouples the proposed mechanism.

The carrier mass and quartic potential add finite terms to the $P$ block.  At
arbitrarily large $k$, the displayed gradient matrix grows as $k^2$ while those
terms remain $k^0$.  They cannot change the principal sign.

## Prior-art boundary

The one-metric scalar--vector--tensor base and its MOND quasistatic limit are
from [Skordis and Zlosnik](https://arxiv.org/abs/2007.00082); its Minkowski
stability analysis is also established
[work](https://arxiv.org/abs/2109.13287).  Preferred-foliation theories and the
importance of Hamiltonian degeneracy/secondary constraints are broad prior
art, including the spatially covariant analyses of
[Gao and Yao](https://arxiv.org/abs/1910.13995) and their
[dynamic-lapse analysis](https://arxiv.org/abs/1806.02811).

The search did not identify this exact fixed $P^{\mu\nu}D_{(\mu}S_{\nu)}$
construction, but absence from a targeted search is not evidence of novelty.
No novelty claim is made.  The exact action is now falsified before that
question becomes scientifically important.

## What the result teaches us

The geometry idea survived; the constant derivative implementation did not.
The useful constraints for a successor are now sharper:

1. Retain a full trace plus traceless orientation response.  A scalar cannot
   encode cluster shear topology.
2. Couple it through a manifestly positive or degenerate principal form.  A
   constant off-diagonal gradient coefficient cannot be attached to a scalar
   whose own stiffness vanishes in the target regime.
3. Do not repair the failure with a fitted acceleration switch.  Such a switch
   would merely add another interpolation and risks nonregular behavior at
   $S=0$.
4. A promising source is a sector whose principal stiffness does not vanish,
   or a constrained/auxiliary tensor whose elimination preserves the complete
   positive form by construction.
5. The successor still must modify the one physical metric's $\Psi$ and $\Phi$
   equations.  A healthy carrier that never enters the Weyl potential cannot
   solve lensing.

## Reproduction

```powershell
$env:PYTHONPATH='src'
python scripts/audit_sigma_v10a_spatial_polarization.py
python scripts/audit_sigma_v10a_quasistatic_ellipticity.py
python -m pytest -q tests/test_sigma_v10a_spatial_polarization.py
```

Machine-readable reports are in
`results/sigma_v10a_spatial_polarization_selection/report.json` and
`results/sigma_v10a_quasistatic_ellipticity_gate/report.json`.

No new observational product or holdout was opened in either audit.
