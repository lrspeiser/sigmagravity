# Sigma v1 pure-nonmetricity action and weak-field decision

## Outcome

The first action cycle under the renewed Sigma Gravity goal is complete.  It
tested the smallest geometry-only theory that can recover the successful galaxy
low-acceleration limit while retaining one metric and no invisible material
source.

The action is covariant, uses one universal acceleration scale, has a regular
GR limit, conserves minimally coupled baryonic stress energy, and passes every
declared weak-field numerical and asymptotic check.  It does **not** advance as
a galaxy--cluster unifier.  Its unique regular isolated weak-field branch has

\[
\Phi=\Psi
\]

and is exactly standard-\(\mu\) AQUAL.  It therefore inherits AQUAL's successful
galaxy behavior and its already measured raw cluster-lensing topology failure.
No parameter fit can change that conclusion.

This is a useful first-principles result: making the GR-equivalent geometric
action nonlinear at low nonmetricity can derive MOND-like galaxy dynamics, but
one scalar geometric invariant does not provide the independent anisotropic
stress needed for cluster convergence and shear.

## Physical postulates tested

1. Baryonic stress energy is the only material source.
2. Matter and light couple minimally to one physical metric \(g_{ab}\).
3. The affine connection is flat and torsion-free.  It is the inertial
   connection of symmetric teleparallel geometry, not a freely specifiable
   halo field.
4. Sigma is a nonlinear gravitational self-interaction of nonmetricity, not a
   material fluid and not a collection of hidden particles.
5. The large-nonmetricity limit is the symmetric-teleparallel equivalent of GR.
6. The same isolated or cosmological boundary rule applies to every system.
7. No object class, per-object gravity constant, or lensing-only multiplier is
   allowed.

The theory has no separately conserved ``Sigma matter'' stress tensor.  Its
nonlinear contribution can be written as an effective geometric source after
rearranging the metric equation, just as one can rearrange nonlinear terms in
GR, but it is not an independently supplied mass distribution.  The action can
exchange energy among its geometric degrees of freedom while the total
diffeomorphism identity and minimally coupled matter equations enforce
\(\nabla^aT^{(b)}_{ab}=0\).

## Covariant action

Let the independent connection obey

\[
R^a{}_{bcd}(\Gamma)=0,\qquad T^a{}_{bc}(\Gamma)=0,
\]

and define

\[
Q_{a mn}=\nabla_a g_{mn},\qquad
Q_a=Q_{a\ m}^{\ \ m},\qquad
\widetilde Q_a=Q^m{}_{a m}.
\]

Use the STEGR scalar convention

\[
\mathbb Q={1\over4}Q_{a mn}Q^{a mn}
-{1\over2}Q_{a mn}Q^{m a n}
-{1\over4}Q_aQ^a
+{1\over2}Q_a\widetilde Q^a.
\]

With \(q_\Sigma=a_\Sigma/c^2\), define

\[
X={\mathbb Q\over2q_\Sigma^2},
\qquad
\mathcal F(X)=\sqrt{X(1+X)}-\operatorname{asinh}\sqrt X,
\]

and

\[
f_\Sigma(\mathbb Q)=2q_\Sigma^2\mathcal F
\left({\mathbb Q\over2q_\Sigma^2}\right).
\]

The tested action is

\[
\boxed{
S_{\Sigma1}=-{c^4\over16\pi G}
\int d^4x\sqrt{-g}\,f_\Sigma(\mathbb Q)
+S_b[g_{ab},\psi_b]
}
\]

with the single physical constant

\[
a_\Sigma=1.2\times10^{-10}\ {\rm m\,s^{-2}}.
\]

The derivative of the action is

\[
f_{\mathbb Q}=\mathcal F_X
=\sqrt{X\over1+X}\equiv\mu_\Sigma(X).
\]

For \(X\gg1\), \(f_{\mathbb Q}\rightarrow1\), so the metric equation tends to
STEGR and hence GR up to the usual boundary term.  For \(X\ll1\),
\(\mathcal F\sim(2/3)X^{3/2}\), producing the deep nonlinear flux law.

## Euler--Lagrange equations and conservation

Let \(P^a{}_{mn}\) be the nonmetricity conjugate defined by

\[
P^a{}_{mn}={\partial\mathbb Q\over\partial Q_a{}^{mn}}.
\]

Variation of the metric gives the standard \(f(\mathbb Q)\) equation in the
chosen sign convention,

\[
{2\over\sqrt{-g}}\nabla_a
\left(\sqrt{-g}\,f_{\mathbb Q}P^a{}_{mn}\right)
+{1\over2}g_{mn}f_\Sigma(\mathbb Q)
+f_{\mathbb Q}\left(
P_{mab}Q_n{}^{ab}-2Q_{abm}P^{ab}{}_n
\right)
=-{8\pi G\over c^4}T^{(b)}_{mn}.
\]

Variation of the flat, torsion-free connection gives

\[
\nabla_m\nabla_n
\left(\sqrt{-g}\,f_{\mathbb Q}P_a{}^{mn}\right)=0.
\]

The connection equation is essential: together with the metric equation it
supplies the diffeomorphism Noether identity.  Since the baryonic action uses
the same metric and contains no independent connection charge, its on-shell
identity is

\[
\nabla^mT^{(b)}_{mn}=0.
\]

The nonlinear theory's complete Hamiltonian degree-of-freedom and strong-
coupling analysis is not claimed here.  The empirical weak-field failure below
retires this action before that expensive stage.  This caution matters because
the degree-of-freedom count in nonlinear \(f(\mathbb Q)\) theories remains a
nontrivial research issue.

## Weak-field derivation

Use

\[
ds^2=-(1+2\Psi/c^2)c^2dt^2
+(1-2\Phi/c^2)d\mathbf x^2.
\]

In coincident gauge, keeping the leading static quadratic gradients gives

\[
\mathbb Q={2\over c^4}
\left[2\nabla\Psi\!\cdot\!\nabla\Phi
-|\nabla\Phi|^2\right].
\]

Therefore

\[
X={2\nabla\Psi\cdot\nabla\Phi-|\nabla\Phi|^2
\over a_\Sigma^2}.
\]

The weak action is

\[
S_{\rm wf}=\int dt\,d^3x\left[
-{a_\Sigma^2\over8\pi G}\mathcal F(X)-\rho_b\Psi
\right].
\]

Independent variation of the two potentials gives

\[
\nabla\cdot\left[\mu_\Sigma(X)\nabla\Phi\right]
=4\pi G\rho_b,
\]

\[
\nabla\cdot\left[
\mu_\Sigma(X)\nabla(\Psi-\Phi)
\right]=0.
\]

On the regular isolated branch \(\mu_\Sigma>0\).  Multiply the second equation
by \(\Psi-\Phi\), integrate over space, and use the common vanishing boundary
condition.  The result is

\[
\int d^3x\,\mu_\Sigma
|\nabla(\Psi-\Phi)|^2=0.
\]

Consequently \(\Psi=\Phi\), and the remaining equation is

\[
\boxed{
\nabla\cdot\left[
\mu_\Sigma(|\nabla\Psi|^2/a_\Sigma^2)\nabla\Psi
\right]=4\pi G\rho_b
}
\]

with

\[
\mu_\Sigma={|\nabla\Psi|/a_\Sigma
\over\sqrt{1+(|\nabla\Psi|/a_\Sigma)^2}}.
\]

This is standard-\(\mu\) AQUAL.  Slow matter responds to
\(-\nabla\Psi\).  Photons respond to
\(W=(\Psi+\Phi)/2=\Psi\).  Thus the theory has a legitimate one-metric photon
law, but no independent cluster lensing response.

## Executable checks

The frozen protocol is
`configs/sigma_v1_nonmetricity_cycle.json`; the deterministic report is
`results/sigma_v1_nonmetricity_cycle/report.json`.

| Check | Result | Gate | Decision |
|---|---:|---:|---|
| STEGR weak-invariant identity | \(4.26\times10^{-14}\) max error | \(10^{-12}\) | pass |
| independent slip-invariant identity | \(2.31\times10^{-14}\) | \(10^{-12}\) | pass |
| action derivative versus analytic \(\mu\) | \(1.42\times10^{-7}\) relative | \(2\times10^{-5}\) | pass |
| deep-limit error at \(g_b/a_\Sigma=10^{-3}\) | \(2.50\times10^{-4}\) | 5% | pass |
| high-field correction at \(g_b/a_\Sigma=10^5\) | \(5.00\times10^{-11}\) | \(10^{-5}\) | pass |
| external dwarf RMSE | 12.422 km/s | \(\le1.05\times\) best MOND | pass |
| ratio to best frozen MOND | 1.00157 | \(\le1.05\) | pass |
| raw cluster root convergence | 0.333 in both ready clusters | 1.000 | **fail** |
| all held-out topologies correct | false | true | **fail** |

The raw scores are inherited rather than rerun because the derived equations
are mathematically identical to the already frozen AQUAL comparator.  In both
AS295 and PLCKG287, AQUAL produces one usable root where each held-out family
contains three observed images; the finite position RMS is therefore undefined.

## Decision and next mechanism

Sigma v1 pure nonmetricity is retired as a galaxy--cluster unifier.  It remains
a compact demonstration that the galaxy sector can arise from an elegant
geometric action, but it has rediscovered the AQUAL weak-field equation and
does not solve the cluster problem.

The next candidate cannot be another regular scalar function of this same
first-derivative invariant.  It must add one baryon-predictable source of
anisotropic stress--a vector/tensor state, or a causal nonlocal/tidal state with
a well-defined continuum limit--so that \(\Phi-\Psi\), convergence, and shear
are predicted rather than multiplied.  Its state cannot be an independently
initialized halo in different notation.

## Prior-art boundary

Symmetric-teleparallel GR, scalar/nonlinear nonmetricity theories, and AQUAL are
prior art.  The result here is a project-specific equivalence-and-falsification
test, not a claim to have invented \(f(\mathbb Q)\) gravity.  Primary starting
points include:

- Järv et al., [Nonmetricity formulation of general relativity and its
  scalar-tensor extension](https://arxiv.org/abs/1802.00492).
- D'Ambrosio, Heisenberg, and Zentarra, [Hamiltonian Analysis of f(Q) Gravity
  and the Failure of the Dirac--Bergmann Algorithm](https://arxiv.org/abs/2308.02250).

## Reproduction

```powershell
python scripts/check_sigma_v1_nonmetricity.py
python -m pytest tests/test_sigma_nonmetricity.py -q
python -m ruff check src/voidscreen/sigma_nonmetricity.py scripts/check_sigma_v1_nonmetricity.py tests/test_sigma_nonmetricity.py
```
