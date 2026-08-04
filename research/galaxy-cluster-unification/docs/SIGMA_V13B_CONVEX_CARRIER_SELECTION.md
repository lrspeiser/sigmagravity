# Sigma v13B convex reduced-carrier selection

## Decision

Sigma v13B passes the frozen **reduced-carrier** advancement gates and is
selected for covariantization. It is not yet a gravity theory and no
observational data were opened.

The selected preferred-frame Hamiltonian density is

$$
\boxed{
\mathcal H_{13B}
={a_\Sigma^2\over2}F_\epsilon(z),
\qquad
z={\Pi^2+|\boldsymbol\nabla\sigma|^2\over a_\Sigma^2}
}
$$

with

$$
F_\epsilon(z)
=\epsilon z+(1-\epsilon)
\left[z-2\sqrt z+2\ln(1+\sqrt z)\right],
\qquad 0<\epsilon<1.
$$

Here $\sigma$ is the candidate nonlinear carrier, $\Pi$ is its canonical
momentum in the preferred frame, $a_\Sigma$ is the acceleration scale, and
$\epsilon$ is a universal positive response floor. The frozen reduced model
therefore uses two physical constants, below the project's five-constant
budget.

This construction succeeds at the precise mathematical gate that rejected
v12A and v13A: its local Hamiltonian is positive and globally strictly convex,
without a freely signed dust charge. Its static slice retains the intended
AQUAL-like nonlinear spatial response, its momentum--velocity map is unique,
and its scalar characteristics are real and no faster than the preferred-frame
unit cone.

The word **reduced** is essential. v13B does not yet explain covariantly where
the preferred frame comes from, how baryons source $\sigma$, how the carrier
couples to the metric, or how the same physical metric bends light. Those are
the next kill gates.

## Why this form was chosen

Let

$$
t=\sqrt z
={\sqrt{\Pi^2+|\boldsymbol\nabla\sigma|^2}\over a_\Sigma}.
$$

Differentiating the shape gives

$$
{dF_\epsilon\over dz}
=\mu_\epsilon(t)
=\epsilon+(1-\epsilon){t\over1+t}.
$$

The phase-space flux is consequently

$$
{\partial\mathcal H_{13B}\over
 \partial(\Pi,\boldsymbol\nabla\sigma)}
=\mu_\epsilon(t)(\Pi,\boldsymbol\nabla\sigma).
$$

On a static configuration, $\dot\sigma=0$ implies $\Pi=0$, so the implicit
Lagrangian is exactly $\mathcal L=-\mathcal H$. The spatial constitutive flux
is then

$$
\mu_\epsilon(|\boldsymbol\nabla\sigma|/a_\Sigma)
\boldsymbol\nabla\sigma.
$$

Once a covariant baryonic source is supplied, this is the left-hand structure
of an AQUAL-like nonlinear Poisson equation. It has three regimes:

- $t\gg1$: $\mu\rightarrow1$, giving the standard high-field response;
- $\epsilon\ll t\ll1$: $\mu\simeq t$, giving the MOND-like nonlinear window;
- $t\ll\epsilon$: $\mu\rightarrow\epsilon$, restoring a regular linear floor.

The last regime is the price of strict convexity at the origin. Whether a
single small universal $\epsilon$ is compatible with cosmology and the
largest low-acceleration systems is an open physical test, not something this
reduced audit establishes.

## Exact convexity result

Because the Hamiltonian is radially symmetric in the four local phase-space
components $(\Pi,\nabla_i\sigma)$, its Hessian has only two distinct
eigenvalues. The three transverse eigenvalues are

$$
\lambda_\perp
=\mu_\epsilon(t)
=\epsilon+(1-\epsilon){t\over1+t},
$$

and the radial eigenvalue is

$$
\lambda_\parallel
=\mu_\epsilon+t{d\mu_\epsilon\over dt}
=\epsilon+(1-\epsilon)
\left[1-{1\over(1+t)^2}\right].
$$

Therefore, for every finite $t\ge0$,

$$
\boxed{
\epsilon\le\lambda_\perp<1,
\qquad
\epsilon\le\lambda_\parallel<1.
}
$$

This proves all of the following locally and nonperturbatively:

1. $\mathcal H_{13B}\ge0$, with its minimum at the origin;
2. the Hamiltonian has no negative-energy phase-space direction;
3. $\dot\sigma=\partial\mathcal H/\partial\Pi$ is globally one-to-one;
4. the implicit first-derivative Lagrangian exists everywhere; and
5. there is no term linear in an arbitrary conserved charge near the origin.

The fifth point separates v13B from the exact v13A multiplier. Near the
origin,

$$
\mathcal H_{13B}
={\epsilon\over2}
(\Pi^2+|\boldsymbol\nabla\sigma|^2)
+O(a_\Sigma^2t^3),
$$

so changing the sign of $\Pi$ cannot turn positive energy into negative
energy.

## Arbitrary-background scalar cone

For a unit spatial propagation direction $n_i$, define the relevant Hessian
entries

$$
A=\mathcal H_{\Pi\Pi},
\qquad
b=n_i\mathcal H_{\Pi s_i},
\qquad
C=n_i\mathcal H_{s_i s_j}n_j,
$$

where $s_i=\nabla_i\sigma$. The principal Hamilton equations give

$$
(c+b)^2=AC,
\qquad
c_\pm=-b\pm\sqrt{AC}.
$$

The matrix

$$
M_n=\begin{pmatrix}A&b\\b&C\end{pmatrix}
$$

is a principal compression of the positive full Hessian. Consequently
$A C>0$, the two characteristics are real, and

$$
\max(|c_+|,|c_-|)
\le\lambda_{\max}(M_n)
\le\lambda_{\max}(\nabla^2\mathcal H_{13B})<1.
$$

This is stronger than a rest-background sound-speed check: it covers arbitrary
local mixtures of canonical momentum, spatial gradient, and propagation
direction within the reduced preferred-frame system.

## Frozen numerical audit

The preregistered audit used:

- five $\epsilon$ values from $10^{-8}$ through $0.5$;
- 802 phase-space radii from zero through $10^{10}a_\Sigma$;
- 81 momentum fractions and 81 propagation-direction cosines;
- 128 independent finite-difference Hessian checks; and
- signed Legendre inversions through $|\dot\sigma|=10$ and
  $|\boldsymbol\nabla\sigma|=100a_\Sigma$.

All reduced-carrier gates pass:

| Check | Frozen result |
|---|---:|
| Maximum scanned $|c|$ | `0.9999999999999899` |
| Minimum scanned $AC$ | `1.0e-12` |
| Maximum independent flux-Jacobian residual | `4.6734e-10` |
| Minimum random Hessian eigenvalue | `0.00106449` |
| Maximum random Hessian eigenvalue | `0.9999987851` |
| Static $\mathcal L+\mathcal H$ residual | `0` |
| Reduced physical constants | `2 / 5` |

The nonzero random minimum is not the global theoretical floor because the
finite-difference random sample did not include the exact origin. The analytic
global lower bound is the selected $\epsilon=10^{-6}$.

## What has and has not been established

Established for the reduced carrier:

- a positive Hamiltonian bounded below;
- strict convexity on all local phase-space backgrounds;
- a unique and nonsingular Legendre transform;
- an exact static AQUAL-like response with a universal floor;
- real, preferred-frame causal scalar characteristics; and
- no linear, object-level dust charge.

Not established:

- a manifestly covariant action;
- a dynamically determined foliation with no hidden integration charge;
- the joint metric--carrier constraint count and physical degrees of freedom;
- a baryonic source law and universal boundary conditions;
- one-metric equations for both massive matter and photons;
- $c_T=c$ after metric covariantization;
- Solar-System PPN and Mercury limits;
- cosmological background behavior; or
- galaxy rotation and raw cluster-lensing performance.

Accordingly, `theory_viable=false` remains the correct project status. v13B
is selected for the next derivation; it is not selected for observational
fitting.

## Relationship to known ideas

The static constitutive response is deliberately AQUAL/MOND-like and is not a
novel claim. Hamiltonian-first convexity and finite propagation have analogies
to nonlinear electrodynamics, while a future covariant preferred foliation
would overlap the design space of Einstein-aether, khronometric, cuscuton, or
degenerate scalar--tensor theories. A targeted literature search did not
identify this exact radial phase-space shape used as a gravity carrier, but
absence from that search is not evidence of novelty.

The relevant comparison literature includes:

- Bekenstein and Milgrom's AQUAL construction and later MOND reviews;
- causality analyses of Born--Infeld-type nonlinear electrodynamics;
- Hamiltonian and constraint studies of cuscuton-like fields;
- Class-Ia DHOST degeneracy conditions; and
- Einstein-aether propagation and positive-energy constraints.

Any novelty assessment must be repeated after a complete covariant action is
specified, because the covariant completion—not this isolated convex
function—determines the actual physical theory.

## Failure accounting and next gate

v13B is not a material failure. The post-v12 accounting therefore remains:

- materially distinct formulation failures: `2`;
- bounded-Hamiltonian failures: `2`;
- three-failure mechanism reset: not triggered;
- observations opened: no.

The next task is to construct a covariant foliation and metric completion that
does not introduce a free clock charge or hidden matter state. Before any data
are opened, that completion must pass the joint constraint count, reduced
energy, common-cone, luminal tensor, PPN, and one-physical-metric lensing gates.

## Reproduction

```powershell
python scripts/audit_sigma_v13b_convex_carrier.py
python -m pytest -q tests/test_sigma_v13b_convex_carrier.py
```

Machine-readable evidence is in
`results/sigma_v13b_convex_carrier/report.json`.
