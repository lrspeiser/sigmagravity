# Sigma v3A local DHOST edge audit

## Outcome

Sigma v3A asked whether the smallest known local, one-parameter relativistic
mechanism that carries density-gradient information could supply the missing
cluster-lensing curvature.  The candidate is the luminal, degenerate
`beta_1=0` beyond-Horndeski subclass.  It uses one physical metric and adds one
universal coefficient, `alpha_H`; no galaxy, cluster, lensing, center, shear,
scale, or orientation parameter is fitted.

The candidate passes its action-identity screen.  The quadratic DHOST
degeneracy relations are satisfied to a maximum relative residual of
`4.23e-16`, its transverse tensor speed is exactly luminal at this order, and
the weak spherical equations reduce to a baryonic density-gradient correction
to the Weyl potential.

It fails the preregistered amplitude screen.  Positivity of the matter response
inside a uniform core requires `alpha_H < 1/3`.  Under that restriction the
largest possible Weyl enhancement for any smooth power-law density profile is
`18.75%`.  Applied to the actual Newtonian baryonic convergence, it closes only
`1.53%` of the Sigma-v1-to-halo gap.  Even the deliberately unphysical gift of
letting the bound multiply the full halo convergence closes at most `39.82%`,
below the frozen `75%` threshold.

The smooth local edge term is therefore retired as the **sole broad cluster
response** before a 2D solver or cluster fit is attempted.  This is a pre-fit
structural screen, not a third raw-holdout failure.  It does not exclude a
localized DHOST edge effect as a subleading correction, but such a correction
cannot justify another free constant unless a different mechanism already
passes the main galaxy--cluster gates.

## Action envelope

Let

\[
X=g^{ab}\nabla_a\phi\nabla_b\phi<0
\]

on a universal timelike background.  The local scalar-tensor sector is

\[
S_{\rm D} = \int d^4x\sqrt{-g}\left[
F(X)R+P(\phi,X)+Q(\phi,X)\Box\phi+
\sum_{I=1}^{5}A_I(X)L_I
\right],
\]

with

\[
F(X)=F_0\left({-X\over-X_0}\right)^{-\alpha_H/2},
\]

\[
A_1=A_2=A_5=0,\qquad
A_3=-{4F_X\over X},\qquad
A_4={4F_X\over X}.
\]

The full action envelope also contains the already tested Sigma-v1 nonlinear
nonmetricity correction and minimally coupled baryons.  The Einstein kinetic
term is counted only once: the Sigma-v1 part contributes its nonlinear
departure from its GR-equivalent linear term, while `F R` supplies the total
metric kinetic sector.  The scalar background functions `P` and `Q` were not
fixed because this amplitude screen fails first; consequently this is not a
complete frozen Sigma-v3 action and makes no claim of full scalar stability.

For the chosen coefficients,

\[
\alpha_H=-{2XF_X\over F},\qquad \beta_1=0,
\]

and the `c_T=1` quadratic-DHOST degeneracy identities give

\[
A_4={48F_X^2-8(F-XF_X)A_3-X^2A_3^2\over8F},
\]

\[
A_5={(4F_X+XA_3)A_3\over2F}=0.
\]

These identities remove the higher-derivative Ostrogradsky mode at the
quadratic-DHOST level.  They do not by themselves prove a healthy cosmological
background, positive scalar sound speed, or a well-posed nonlinear initial
value problem.

## Weak spherical prediction

Use the project convention

\[
ds^2=-(1+2\Psi/c^2)c^2dt^2+(1-2\Phi/c^2)d\mathbf x^2,
\qquad W={\Psi+\Phi\over2}.
\]

Massive matter responds to `Psi`; photons respond to the Weyl potential `W`.
On the screened spherical branch, the published beyond-Horndeski laws become

\[
{d\Psi\over dr}={GM\over r^2}-{\alpha_H\over2}GM'',
\]

\[
{d\Phi\over dr}={GM\over r^2}+\alpha_H G{M'\over r}.
\]

With `M'=4 pi r^2 rho`, their Weyl average is

\[
\boxed{
\Delta {dW\over dr}
=-\pi\alpha_H G r^2\rho'(r)
}.
\]

This is the attractive feature of the mechanism: a declining baryonic density
adds photon curvature specifically at an edge.  It is not a uniform lensing
multiplier.  It also exposes the limitation.  Outside matter, `M'=M''=0`, so
the correction vanishes rather than redistributing curvature over the broad
regions where cluster critical curves and shear are observed.

Inside a constant-density sphere,

\[
{g_\Psi\over g_N}=1-3\alpha_H,\qquad
{g_\Phi\over g_N}=1+3\alpha_H,\qquad
{g_W\over g_N}=1.
\]

Requiring an inward, nonnegative matter response gives `alpha_H<1/3`.  For a
smooth density `rho proportional to r^-n`, `0<n<3`,

\[
{\Delta g_W\over g_N}={\alpha_H\over4}n(3-n).
\]

The expression peaks at `n=3/2`, giving

\[
\sup {\Delta g_W\over g_N}
={9\alpha_H\over16}<{3\over16}=18.75\%.
\]

## Frozen amplitude audit

The audit reused only already-spent aggregate diagnostics; no raw holdout was
unsealed and no coefficient was optimized.

| Quantity | Value |
|---|---:|
| Median baryon-only GR convergence | 0.04730 |
| Median Sigma-v1/AQUAL convergence | 0.10238 |
| Median optimistic existing convergence | 0.37987 |
| Median halo convergence | 0.68868 |
| Actual-source maximum added convergence | 0.00887 |
| Sigma-v1 gap closed using the actual source | 1.53% |
| Sigma-v1 gap closed under the halo-scaled gift | 22.22% |
| Best optimistic gap closed under the halo-scaled gift | 39.82% |
| Frozen advancement requirement | 75% |

The halo-scaled rows are upper-bound stress tests, not physical predictions:
they pretend the local fractional correction multiplies the field we are
trying to explain rather than the baryonic field that actually sources it.
Failing even that gift makes the negative decision robust.

## Prior-art and claim boundary

The `c_T=1` DHOST action class, its degeneracy conditions, and the screened
spherical `Xi_i` laws are published prior art.  The primary reference is
[Langlois et al., *Scalar-tensor theories and modified gravity in the wake of
GW170817*](https://arxiv.org/abs/1711.07403).  Galileon/k-mouflage mechanisms
that reproduce MOND-like dynamics while screening the Solar System also
predate this project; see
[Babichev, Deffayet, and Esposito-Farese](https://arxiv.org/abs/1106.2538).

The project result is narrower: the exact density-gradient reduction, the
frozen comparison with the spent cluster convergence deficit, and the
quantitative retirement of this local edge term as the sole broad response.
It is not a claim to have invented beyond-Horndeski gravity.

## Decision and next mechanism

The missing cluster information must be spread away from a local baryonic edge
and must carry orientation.  The next derivation target is therefore a causal,
baryon-forced **nonlocal tidal response**.  Its added state must be uniquely
fixed by a retarded prescription and universal boundary condition; any free
homogeneous profile would be observationally equivalent to an invisible halo.

The nonlocal candidate must be rejected before empirical fitting unless it:

1. has no free auxiliary initial data or extra propagator pole;
2. preserves a positive physical spectrum and causal response;
3. gives `c_T=c` and automatic high-acceleration Solar suppression;
4. derives both `Psi` and `W` from one metric equation; and
5. produces a nonzero, baryon-registered trace-free shear response.

## Reproduction

```powershell
python scripts/check_sigma_v3a_dhost_edge.py
python -m pytest tests/test_sigma_v3a_dhost_edge.py -q
python -m ruff check src/voidscreen/sigma_dhost_edge.py scripts/check_sigma_v3a_dhost_edge.py tests/test_sigma_v3a_dhost_edge.py
```
