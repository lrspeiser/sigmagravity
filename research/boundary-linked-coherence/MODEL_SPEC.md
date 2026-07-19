# Minimal Model Specification

## 1. Scope and epistemic status

This document turns the boundary-link idea into quantities that can be computed and falsified. It
does not provide a covariant action, identify a new particle, or claim that electromagnetic phase
coherence exists between unrelated stars.

The first stage is deliberately nonrelativistic and quasi-static. A relativistic retarded model is
allowed to proceed only after the static kernel passes reciprocity and conservation checks.

## 2. Canonical anchor

The existing Σ-Gravity prediction is the locked null model:

\[
\nabla^2\Phi_N = 4\pi G\rho_b,
\]

\[
\Sigma_0(\mathbf x)=1+A(L)\,\mathcal C(\mathbf x)\,h(g_N),
\]

\[
\nabla^2\Phi = 4\pi G\rho_b +
\nabla\!\cdot\!\left[(\Sigma_0-1)\mathbf g_N\right].
\]

All BLC comparisons keep the repository values of `g†`, `A₀`, `L₀`, `n`, and the stellar
mass-to-light ratios fixed unless a test explicitly profiles over their published uncertainties.

## 3. Operational link statistic

For a field location `x`, define a bounded, mass-normalized link statistic

\[
Q_M(\mathbf x)=
\frac{\int d^3y\;\rho_b(\mathbf y)\,
K_R(r_{xy})\,C_{xy}\,B_{xy}}
{\int d^3y\;\rho_b(\mathbf y)\,K_R(r_{xy})+\epsilon},
\qquad 0\le Q_M\le1.
\]

The factors have distinct jobs:

- `K_R(r)` is a normalized range kernel. Candidate families must include compact, exponential,
  and scale-free tails.
- `C_xy` is a source-side kinematic correlation. It may use predicted flow or independent
  velocity-dispersion data, never the target rotation-curve residual.
- `B_xy` is a symmetric boundary-openness statistic derived from baryonic geometry and
  environment.
- The denominator prevents the response from growing as the square of the number of sources.

The first candidate for symmetric openness is

\[
B_{xy}=\sqrt{O(\mathbf x,\hat{\mathbf n}_{xy})
O(\mathbf y,-\hat{\mathbf n}_{xy})},
\]

where `O(x,n)` is the fraction of a specified cone or tube beyond `x` that remains below fixed
baryonic-density and acceleration thresholds. Thresholds are global and must be computed without
using `V_obs`. Alternatives based on solid angle, line-of-sight column density, or graph
connectivity are compared on synthetic data before any astronomical fit.

The BLC enhancement is

\[
\Sigma_M(\mathbf x)=1+A(L)\,Q_M(\mathbf x)\,h(g_N).
\]

This is an operational bridge to the current QUMOND-like pipeline, not yet a fundamental field
equation.

## 4. Literal luminosity alternative

The literal-radiation hypothesis replaces baryonic density in the numerator with a normalized
emissivity weight:

\[
Q_L(\mathbf x)=
\frac{\int d^3y\;j(\mathbf y)\,
K_R(r_{xy})\,C_{xy}\,B_{xy}}
{\int d^3y\;j(\mathbf y)\,K_R(r_{xy})+\epsilon}.
\]

Two amplitude choices must be tested separately:

1. **Phase-carrier version:** use `Q_L` only for path structure while retaining the canonical
   mass-normalized amplitude.
2. **Energy-source version:** scale the added source by radiative energy `j/c²`.

The second version faces an immediate magnitude test: the radiative energy in transit cannot be
silently promoted into a much larger gravitational source. Any extra interaction energy must
appear in the total stress-energy accounting.

For an approximately steady, optically thin source of luminosity `L`, the order-of-magnitude
radiative energy still inside radius `R` is

\[
E_{\rm rad}(<R)\simeq \frac{LR}{c},
\qquad
M_{\rm rad}(<R)\simeq \frac{LR}{c^3}.
\]

`HL-energy` must compare this available mass equivalent with the effective mass implied by its
additional acceleration. Time-integrated historical luminosity is a different memory hypothesis
and must not be substituted after seeing the result.

## 5. Directional prediction

A scalar `Q` can reproduce a radial window but cannot test the distinctive claim that links align
toward outer absorbers or external masses. Preserve the first angular moment:

\[
\mathbf q(\mathbf x)=
\frac{\int d^3y\;w_{xy}B_{xy}\hat{\mathbf n}_{xy}}
{\int d^3y\;w_{xy}+\epsilon},
\]

with `w_xy = ρ_b K_R C_xy` for `HM` and `w_xy = j K_R C_xy` for `HL`.

The first anisotropic perturbation may be written

\[
\mathbf g_{\rm BLC}=g_N A(L)h(g_N)
\left[Q\,\hat{\mathbf g}_N + \epsilon_q
\mathbf P_\perp(\mathbf q)\right],
\]

where `ε_q` is a single global coefficient and `P⊥` removes any component already degenerate with
the radial model. `ε_q = 0` is the preregistered null. This parameter is not fitted until the
isotropic model and synthetic recovery tests pass.

## 6. Required limits

Every candidate implementation must demonstrate these limits numerically and, where possible,
analytically:

| Limit | Required behavior |
|---|---|
| `g_N / g† → ∞` | `Σ → 1` |
| disordered flow | `C_xy → 0` and the BLC term vanishes |
| compact isolated source | no long-baseline enhancement |
| homogeneous isotropic distribution | vector link moment cancels |
| target mass rescaling | acceleration of the target is unchanged |
| source duplication at fixed density | no `N²` divergence |
| `B_xy → canonical window` | recover the canonical prediction |
| far outside a finite source | finite total effective mass or an explicitly justified nonlocal asymptotic law |

## 7. Conservation and causality gates

### Static reciprocity

The equal-time kernel must satisfy

\[
K(\mathbf x,\mathbf y)=K(\mathbf y,\mathbf x).
\]

A pairwise force implementation must sum to zero net internal force and torque for an isolated
synthetic system to numerical precision. If the QUMOND-like source formulation is used instead,
the equivalent field-stress boundary integral must close.

### Energy accounting

Interference or coherence may redistribute stress-energy but may not create integrated energy.
For every synthetic case, report the baryonic energy, radiative energy, field energy, and any
explicit interaction/binding term. A normalization that merely hides an unexplained energy source
does not pass.

### Retardation

A later covariant candidate may use a spacetime kernel with support only on or inside the past
light cone. Naively replacing equal-time arguments with `t-r/c` is not sufficient: it can break
reciprocity and conservation. The retarded stage requires either an action/in-in construction or
an explicit conserved effective stress-energy tensor.

### Propagation

The canonical gravitational-wave sector remains luminal. Any new propagating mode must be shown
not to violate multimessenger propagation constraints or introduce an observable gravitational
slip inconsistent with the one-potential lensing assumption.

## 8. Relationship to the existing one-dimensional kernel

`derivations/test_nonlocal_coherence_kernel.py` is a useful comparator, not the starting
definition. Its current density proxy, decoherence weights, source location, and local/nonlocal
mixing coefficient are exploratory. The BLC implementation should reproduce that script as a
named legacy variant, then replace each hand-set term with a measured source-side quantity or a
globally preregistered parameter.

## 9. Identifiability requirement

`Q`, `h(g_N)`, `A(L)`, disk radius, surface density, and environment can be strongly correlated.
Before interpreting any coefficient, the pipeline must report:

- the feature correlation matrix and variance-inflation factors;
- posterior or bootstrap covariance;
- performance when each BLC factor is shuffled within mass and surface-brightness strata;
- recovery from null and injected synthetic catalogs;
- ablations of `B`, `C`, `K_R`, luminosity, and the anisotropic moment.

A coefficient is not evidence for a link if the same held-out score is obtained from a shuffled or
purely radial proxy.
