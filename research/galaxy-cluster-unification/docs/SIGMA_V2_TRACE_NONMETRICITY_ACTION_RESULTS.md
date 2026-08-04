# Sigma v2 trace-nonmetricity action and raw-lensing decision

## Outcome

Sigma v2 tested the smallest geometry-only extension of Sigma v1 that lets the
two weak metric potentials respond differently.  It adds one of the independent
nonmetricity traces, rather than a new material field, free vector profile, or
lensing multiplier.

The result is decisive.  The action has one physical metric, one universal
constant, a GR high-field limit, conserved minimally coupled matter, and the
ordinary luminal tensor-wave cone at linear order.  Its weak equations are not
new: massive-particle dynamics reduce exactly to simple QUMOND, while photons
see the fixed average of the QUMOND time potential and the Newtonian spatial
potential.

That weak theory passes the external dwarf-galaxy gate with an RMSE of
`12.403 km/s`, exactly the best frozen MOND comparator.  It fails raw strong
lensing in both ready clusters.  The model recovers only one of three required
held-out image roots overall in each cluster and never predicts the observed
held-out topology.  Sigma v2 is therefore retired without fitting a cluster
parameter or completing an expensive nonlinear Hamiltonian analysis.

This is the second action-level closure under the renewed theory program:

- Sigma v1: one nonlinear STEGR scalar, no slip, exactly AQUAL, cluster failure;
- Sigma v2: a second trace scalar, derived slip, exactly QUMOND dynamics, cluster
  failure.

The common lesson is more specific than “MOND fails.”  A local scalar made only
from first metric derivatives can change radial strength, but these two minimal
geometric realizations do not create the baryon-registered trace-free curvature
and shear pattern required for multiple cluster images.

## Physical postulates

1. Observed baryonic stress energy is the only material source.
2. Massive matter and photons couple minimally to one physical metric
   \(g_{ab}\).
3. The independent affine connection is flat and torsion-free.  It has no
   object-specific initial profile that could function as a hidden halo.
4. Sigma is a nonlinear geometric self-interaction of nonmetricity, not an
   invisible material component.
5. Every object uses the same action, isolated boundary condition, and
   acceleration scale.
6. Object labels, per-object gravity constants, and lensing-only multipliers are
   prohibited.

## Covariant action

Let

\[
Q_{a mn}=\nabla_a g_{mn},\qquad
Q_a=Q_{a\ m}{}^m,\qquad
\widetilde Q_a=Q^m{}_{a m},
\]

with a flat, torsion-free connection.  Use the STEGR scalar

\[
\mathbb Q={1\over4}Q_{a mn}Q^{a mn}
-{1\over2}Q_{a mn}Q^{m a n}
-{1\over4}Q_aQ^a
+{1\over2}Q_a\widetilde Q^a
\]

and the independent trace scalar

\[
\mathbb V=\widetilde Q_a\widetilde Q^a.
\]

With \(q_\Sigma=a_\Sigma/c^2\), define

\[
Y={\mathbb V\over4q_\Sigma^2},
\qquad
\nu_s(y)={1\over2}+\sqrt{{1\over4}+{1\over y}},
\]

and fix the primitive by

\[
\mathcal H(0)=0,
\qquad
\mathcal H_Y(Y)=1-\nu_s(\sqrt Y).
\]

The frozen action is

\[
\boxed{
S_{\Sigma2}=-{c^4\over16\pi G}\int d^4x\sqrt{-g}
\left[\mathbb Q+2q_\Sigma^2\mathcal H(Y)\right]
+S_b[g_{ab},\psi_b]
}
\]

with the single physical parameter

\[
a_\Sigma=1.2\times10^{-10}\ {\rm m\,s^{-2}}.
\]

There is no free normalization in \(\mathcal H\): the additive constant is set
to zero and its derivative is fixed by the already frozen galaxy law.

## Exact variational structure

Define the two nonmetricity conjugates

\[
P^a{}_{mn}={\partial\mathbb Q\over\partial Q_a{}^{mn}},
\qquad
Z^{a mn}={\partial\mathbb V\over\partial Q_{a mn}}
=\widetilde Q^m g^{an}+\widetilde Q^n g^{am},
\]

and

\[
\Pi^a{}_{mn}=P^a{}_{mn}+{1\over2}\mathcal H_Y Z^a{}_{mn}.
\]

Variation of the flat, torsion-free connection gives

\[
\nabla_m\nabla_n\left(\sqrt{-g}\,\Pi_a{}^{mn}\right)=0.
\]

For a compact exact statement of the metric equation, write

\[
f(g,Q)=\mathbb Q+2q_\Sigma^2\mathcal H
\left({\mathbb V\over4q_\Sigma^2}\right).
\]

Holding \(Q_{a mn}\) fixed in the algebraic metric derivative, the Euler
equation is

\[
{1\over2}f g^{mn}
+{\partial f\over\partial g_{mn}}
-{1\over\sqrt{-g}}\nabla_a
\left(\sqrt{-g}\,{\partial f\over\partial Q_{a mn}}\right)
={8\pi G\over c^4}T_b^{mn}.
\]

The trace contribution is fully determined by

\[
{\partial\mathbb V\over\partial g_{mn}}
=-\widetilde Q^m\widetilde Q^n
-2\widetilde Q^r g^{a(m}g^{n)b}Q_{b r a},
\qquad
{\partial f\over\partial Q_{a mn}}=\Pi^{a mn}.
\]

The STEGR portion of this equation is the Einstein tensor, up to the standard
boundary identity.  No second potential equation has been inserted by hand.

Diffeomorphism invariance of the metric and connection equations supplies the
off-shell Noether identity.  With the connection equation imposed and matter
on shell, minimal coupling therefore gives

\[
\nabla_mT_b^{mn}=0.
\]

The added geometry can be moved to the right side and called an effective
stress tensor, but it is not an independently specified matter distribution.

## Weak-field derivation

Use

\[
ds^2=-(1+2\Psi/c^2)c^2dt^2
+(1-2\Phi/c^2)d\mathbf x^2.
\]

At leading static order in coincident gauge,

\[
\mathbb Q={2\over c^4}
\left[2\nabla\Psi\cdot\nabla\Phi-|\nabla\Phi|^2\right],
\qquad
\mathbb V={4\over c^4}|\nabla\Phi|^2.
\]

Thus \(Y=|\nabla\Phi|^2/a_\Sigma^2\) and

\[
S_{\rm wf}=\int dt\,d^3x\left\{
-{1\over8\pi G}\left[
2\nabla\Psi\cdot\nabla\Phi-|\nabla\Phi|^2
+a_\Sigma^2\mathcal H(Y)
\right]-\rho_b\Psi\right\}.
\]

Independent variation gives

\[
\boxed{\nabla^2\Phi=4\pi G\rho_b}
\]

and

\[
\boxed{
\nabla^2\Psi=
\nabla\cdot\left[
\nu_s(|\nabla\Phi|/a_\Sigma)\nabla\Phi
\right].
}
\]

These are exactly the simple-QUMOND equations, with \(\Phi\) playing the
Newtonian auxiliary potential and \(\Psi\) the physical time potential.
Massive particles respond to \(-\nabla\Psi\).  Because both potentials are
components of the same physical metric, photons respond to

\[
\boxed{W={\Psi+\Phi\over2}.}
\]

In a spherical deep field,

\[
g_\Psi\simeq\sqrt{a_\Sigma g_b},
\qquad
g_\Phi=g_b,
\qquad
g_W\simeq{1\over2}\sqrt{a_\Sigma g_b}.
\]

The action therefore derives a galaxy-strength matter force but only about
half that enhancement for light in the deep limit.  This is a prediction, not
an adjustable lensing factor.

## Gravitational-wave and health boundary

For a transverse-traceless perturbation,
\(\partial^jh_{ij}=0\) and \(h^i{}_i=0\).  Consequently
\(\widetilde Q_a=0\) at linear order, so the added trace term does not alter the
quadratic transverse-tensor action and \(c_T=c\) at that order.

This does **not** prove the full theory healthy.  General multi-invariant
nonmetricity theories can contain additional scalar or vector modes.  A full
Hamiltonian constraint count, positive-energy audit, and well-posedness proof
would be required if the empirical gate passed.  It did not, so those claims
are deliberately not made.

## Executable results

The frozen protocol is
`configs/sigma_v2_trace_nonmetricity_cycle.json`.  The deterministic report is
`results/sigma_v2_trace_nonmetricity_cycle/report.json`.

| Check | Result | Gate | Decision |
|---|---:|---:|---|
| Trace-invariant weak identity | 0 max error | \(10^{-12}\) | pass |
| Action derivative identity | \(2.03\times10^{-6}\) relative | \(2\times10^{-5}\) | pass |
| Deep matter-limit error | 1.59% | 5% | pass |
| High-field correction at \(g_b/a_\Sigma=10^5\) | \(9.9999\times10^{-6}\) | \(10^{-5}\) | pass |
| External dwarf RMSE | 12.403 km/s | \(\le1.05\times\) best MOND | pass |
| Ratio to best frozen MOND | 1.000 | \(\le1.05\) | pass |
| AS295 held-out root fraction | 0.333 | 1.000 | **fail** |
| PLCKG287 held-out root fraction | 0.333 | 1.000 | **fail** |
| All held-out topologies correct | false | true | **fail** |

The lensing calculation uses the registered map coordinates (array rows north,
columns east), profiles only the standard source-position nuisance for each
family, and fits no gravity amplitude, center, ellipticity, shear, orientation,
or scale.

Retuning \(a_\Sigma\) after seeing these images is not allowed by the frozen
protocol.  More importantly, the missing information is spatial: each held-out
family receives only one root where two to five images are observed.  A scalar
amplitude adjustment is not evidence for the missing baryon-registered shear
pattern.

## Prior-art boundary

The weak matter equation is QUMOND and must be credited as such.  Milgrom's
[original QUMOND paper](https://arxiv.org/abs/0911.5464) derives the
nonrelativistic bi-potential action and its conservation laws.  General
quadratic teleparallel/nonmetricity actions and their extra-mode problem are
also established research; see [General Teleparallel Quadratic
Gravity](https://arxiv.org/abs/1909.09045).  Pure-metric nonlocal MOND models
with enhanced lensing also already exist, including [Deffayet,
Esposito-Farese, and Woodard](https://arxiv.org/abs/1106.4984).

The project-specific contribution is the covariant trace-nonmetricity embedding
and its frozen raw-image falsification.  It is not a claim to have invented
QUMOND or multi-invariant nonmetricity gravity.

## Decision and next action class

Sigma v2 is retired as a galaxy--cluster unifier.  The third action cannot be
another local scalar function of first metric derivatives.  It must produce a
trace-free, orientation-carrying response sourced and boundary-fixed by the
baryons themselves.  A free vector/tensor concentration is inadmissible because
it would be a dark halo under another name.

Before freezing Sigma v3, the candidate must pass three pre-action checks:

1. its trace-free state is uniquely fixed by baryons and a universal boundary
   rule;
2. its localized kinetic/constraint system has no negative-energy mode; and
3. it can alter convergence and shear orientation, not only multiply a radial
   deflection map.

## Reproduction

```powershell
python scripts/check_sigma_v2_trace_nonmetricity.py
python -m pytest tests/test_sigma_v2_trace_nonmetricity.py -q
python -m ruff check src/voidscreen/sigma_nonmetricity.py scripts/check_sigma_v2_trace_nonmetricity.py tests/test_sigma_v2_trace_nonmetricity.py
```
