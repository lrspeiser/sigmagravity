# Sigma v17Q pressure symmetron no-go

## Candidate

V17P showed that changing only the scalar kinetic flux cannot erase the Sun's
integrated pressure charge. The materially different successor is a standard
symmetry-restoring scalar:

\[
V(X)=-{\mu^2X^2\over2}+{\lambda X^4\over4},
\qquad
q(X)=-{X^2\over2M^2},
\]

with the same reciprocal physical metric

\[
\widetilde g_{\mu\nu}=e^{2q}
[g_{\mu\nu}+2qU_\mu U_\nu].
\]

For an isotropic fluid the dust term still cancels, but pressure changes the
effective potential:

\[
V_{\rm eff}(X)=
{1\over2}\left(-\mu^2+{3p/c^2\over M^2}\right)X^2
+{\lambda X^4\over4}.
\]

High pressure restores (X=0), where the coupling vanishes. Low pressure
allows (X\ne0). The scalar has a canonical kinetic term and bounded quartic
potential, and the physical metric is non-derivative, so this candidate avoids
the v17N kinetic theorem.

This is standard symmetron prior art. The only project-specific question is
whether replacing matter density by the reciprocal pressure source produces
the Solar/cluster selectivity we need.

## Pressure-column theorem

For a spherical source, the standard thin-shell coordinate is proportional to

\[
\mathcal A_p\propto {\rho_pR^2\over\rho_{\rm crit}L^2}
\propto {\Pi_pGM\over Rc^2},
\qquad \rho_p={3p\over c^2}.
\]

Let (S(\mathcal A_p)) be the surviving scalar-charge fraction. Symmetron
screening makes (S) decrease as (\mathcal A_p) increases.

The Model S Solar pressure profile gives

\[
\Pi_\odot=3.42083\times10^{-6}.
\]

Using the deliberately favorable 17 keV, all-baryon cluster pressure from
v17G, every one of the 84 spent CLASH points has a pressure-column coordinate
larger than the Sun's. The minimum, median, and maximum ratios are approximately

\[
1.288,\qquad23.42,\qquad78.12.
\]

Thus (S_{\rm cluster}\le S_\odot). Normalizing one universal coupling to
supply at least one missing baryonic field in the cluster gives

\[
\alpha^2\Pi_{\rm cluster}S_{\rm cluster}\ge1.
\]

The Solar slip must then satisfy

\[
\boxed{
|\gamma-1|_\odot
=2\alpha^2\Pi_\odot S_\odot
\ge {2\Pi_\odot\over\Pi_{\rm cluster}}.
}
\]

The screen therefore has the wrong ordering: the huge cluster size outweighs
its low pressure density.

## Executable profile control

The audit also solves the normalized quartic scalar equation through the full
Model S pressure profile. Every choice favors the candidate:

- the vacuum range is the smallest declared cluster value, 12.5 kpc;
- the critical pressure is the largest *enclosed mean* CLASH pressure, lower
  than a true local central pressure and therefore favorable to Solar
  screening; and
- the cluster is credited as fully unscreened when normalizing the coupling.

The protocol and exact numerical gates are frozen in
`configs/sigma_v17q_pressure_symmetron_no_go.json`. No holdout is opened.

## Prior-art boundary

The symmetry-restoring mechanism is the published
[symmetron](https://arxiv.org/abs/1001.4525). Screening in disformal theories
is treated in [Sakstein](https://arxiv.org/abs/1409.1734), and the one-metric
structure overlaps [TeVeS](https://arxiv.org/abs/astro-ph/0403694) and the
general [conformal/disformal metric](https://arxiv.org/abs/gr-qc/9211017).
The Solar input is the published
[Model S](https://doi.org/10.1126/science.272.5266.1286).

## Result

The hash-locked audit completed without opening a holdout or fitting an
observational target. The 2,402-point Model S profile gives

\[
\Pi_\odot=3.42082895\times10^{-6},
\qquad
\mathcal C_\odot=7.26093780\times10^{-12}.
\]

For the 84 spent CLASH diagnostic points, the minimum, median, and maximum
pressure columns are

\[
9.35312\times10^{-12},\quad
1.70068\times10^{-10},\quad
5.67247\times10^{-10}.
\]

Every point requires at least an order-unity extra Weyl field and every one
has a larger pressure column than the Sun. The coupling-independent theorem
therefore gives

\[
|\gamma-1|_\odot\ge0.0767803,
\]

which is 3,338.27 times the Cassini limit.

The independent Model S boundary-value control reaches the same conclusion.
At the deliberately favorable 12.5 kpc range, its numerical solution has

- central field fraction: `0.905115`;
- surface field fraction: `0.984204`;
- surviving Solar charge fraction: `0.937159`;
- charge fraction required by Cassini: `0.000299556`;
- resulting Solar slip proxy: `0.0719553`, or 3,128.49 Cassini limits;
- boundary residual: `4.08e-13`; and
- fractional resolution change: `4.73e-11`.

The failure is not marginal and is not caused by solver resolution. Standard
symmetry restoration leaves almost the entire Solar pressure charge when the
field range is long enough to operate across a cluster.

## Three-formulation decision

The direct pressure-only reciprocal metric has now failed the same universal
cluster-strength plus Solar-slip gate in three materially distinct ways:

| Formulation | What was changed | Necessary failure |
|---|---|---:|
| v17G | Unscreened linear pressure propagation | `0.0238204`, 1,035.67 Cassini limits |
| v17P | Conserved nonlinear AQUAL/K-mouflage kinetic flux | analytic floor `9.81846e-5`; numerical floor `6.72738e-4` |
| v17Q | Nonconserved symmetry-restoring scalar charge | theorem `0.0767803`; Model S control `0.0719553` |

The preregistered stopping rule is therefore triggered. We retire the direct
pressure-only reciprocal metric instead of adding a fourth screen. Pressure
may still be part of a different covariant baryonic source, but the next root
action must generate galaxy dynamics and lensing through a genuinely different
source or propagation mechanism.

Machine-readable results are in
`results/sigma_v17q_pressure_symmetron_no_go/report.json`.
