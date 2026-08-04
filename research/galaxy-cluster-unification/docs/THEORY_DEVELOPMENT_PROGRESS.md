# Theory-development progress against the stage gates

## 2026-08-04 Sigma v7 positive-carrier sequence

The v7 sequence replaced v6D's multiplier-localized retarded response with a
positive-norm massive spin-2 carrier.  The unscreened v7A spectrum has two
massless plus five massive healthy spin-2 modes, but Solar high-field bounds
limit its residue to `7.5e-6`, leaving less than `0.00075%` useful lensing.  Its
positive Yukawa response also decreases with radius.  V7A is retired before
data; see
[`SIGMA_V7A_POSITIVE_LOCAL_CARRIER_GATE.md`](SIGMA_V7A_POSITIVE_LOCAL_CARRIER_GATE.md).

The spherical Vainshtein v7B control restores GR at high enclosed density, but
its screening coordinate depends only on `M/r^3`.  Equal-density disk and
strong-lens archetypes have identical screening for every universal range, to
`6.54e-16` numerical precision.  The healthy bimetric exterior also caps light
deflection at a factor `1.5`, below the factor-`3` carrier target.  The spherical
control is retired; see
[`SIGMA_V7B_SPHERICAL_VAINSHTEIN_GATE.md`](SIGMA_V7B_SPHERICAL_VAINSHTEIN_GATE.md).

The full three-dimensional cubic Hessian v7C **construction** passes.  It
recovers an analytic spherical solution to `6.09e-11`, has maximum normalized
residual `7.997e-7`, minimum temporal coefficient `3.003`, minimum spatial
ellipticity eigenvalue `2.079`, `1.165%` double-resolution change, and `7.223%`
nonadditivity for separated sources with `1.26e-16` rotation error.  See
[`SIGMA_V7C_CUBIC_HESSIAN_CONSTRUCTION_GATE.md`](SIGMA_V7C_CUBIC_HESSIAN_CONSTRUCTION_GATE.md).

The subsequent physical-metric projection fails.  The leading helicity-zero
metric perturbation gives `delta Psi=-pi/2` and `delta Phi=+pi/2`, hence exactly
zero change in the Weyl potential.  A disformal term or residual `X^(3)` tensor
mixing could affect light, but v7C froze neither the complete disformal scalar
mapping nor the coupled tensor equation.  Its scalar nonadditivity cannot be
scored as lensing.  No map was opened; v7C is retained only as a dynamics
control.  See
[`SIGMA_V7C_PHYSICAL_METRIC_PROJECTION_GATE.md`](SIGMA_V7C_PHYSICAL_METRIC_PROJECTION_GATE.md).

This completes three materially distinct failures of the positive-spin-2
carrier objective.  The v7A unscreened pole fails Solar-safe amplitude, v7B
spherical screening fails amplitude and equal-density discrimination, and v7C
fails closure of a nonzero physical lensing projection.  The route is retired
under the planned mechanism-reset rule; no additional v7 response term will be
fit.  See
[`SIGMA_V7_POSITIVE_SPIN2_FALSIFICATION.md`](SIGMA_V7_POSITIVE_SPIN2_FALSIFICATION.md).

## 2026-08-03 v5C exterior-law failure

The fixed v5C row is retired before full variation or data. In the published
screened luminal-DHOST limit, its potential corrections are proportional to
`M'(r)` and `M''(r)` and vanish outside a source once enclosed baryonic mass is
constant. The exterior is exactly GR. In the unscreened small-field limit, the
fixed row is Newton plus an attractive massive scalar, with the identity
`d log(g)/d log(r) <= -2` for every positive strength and range.

Across `1e-8<=r/L<=1e8` and scalar strength through `1e6`, the shallowest
acceleration slope is `-2`, no flat-slope interval exists, and circular speed
falls to at most `0.316228` over a radial decade versus the required
`0.9--1.1`. See
[`SIGMA_V5C_EXTERIOR_LAW_RESULTS.md`](SIGMA_V5C_EXTERIOR_LAW_RESULTS.md).

This rejects the fixed canonical massive-scalar row, not every DHOST theory.
Together with the strict-causality failure of pure static `P(X)` derivative
screening, it removes the present local one-scalar route. The next action must
provide a constrained baryon-forced response that persists through vacuum
without introducing a freely assigned halo state.

## 2026-08-03 v5C degeneracy-first action selection

The successor lane is a fixed four-constant member of the published luminal
Class-Ia quadratic DHOST family. Its `A1=A2=0` tensor condition gives `c_T=c`,
while `A4` and `A5` are fixed algebraically by `F`, `F_X`, and `A3`; they are
not new fit functions. The provisional row uses a curvature-sourced canonical
massive scalar and one even Hessian activation
`X_hat^2/(1+X_hat^2)^(3/2)`. It is globally signed-safe and makes every
dependent coefficient bounded through the frozen high-field scan.

Ten thousand random coefficient tuples and the complete signed trial scan
give a maximum normalized degeneracy residual of `2.10e-16`. The row has four
universal constants and one physical metric. No data were opened. The action
class is prior art; only the fixed activation and proposed baryon-locked
lensing use are possible novelties. Full equations, FLRW/scalar health,
hyperbolicity, PPN, branch uniqueness, and a term-level prior-art audit remain
mandatory. See
[`SIGMA_V5C_DEGENERACY_FIRST_ACTION_SELECTION.md`](SIGMA_V5C_DEGENERACY_FIRST_ACTION_SELECTION.md).

The selection also closes a tempting shortcut. Any pure `P(X)` derivative
screen that grows with a static spacelike gradient has
`c_parallel^2=1+2X P_XX/P_X>1`; the executable representative reaches almost
three. It is rejected under the project's strict causal-characteristic gate.

## 2026-08-03 v5B nonlinear-degeneracy failure

The exact v5B action is retired before data. Its `sigma=0` FLRW branch remains
GR at linear order: the transition source begins at fourth metric-perturbation
order, its tree-level feedback begins at eighth order, and the free scalar has
positive subluminal kinetic coefficients. The nonlinear static background is
decisive instead.

In a local ADM reduction, STEGR plus a canonical scalar has a rank-two kinetic
Hessian with a null lapse direction. The v5B band-pass source alone makes it
rank three and changes the new lapse coefficient from negative below the
transition to positive above it, crossing zero at the source maximum. The
orientation transport alone is also rank three on all 5,000 frozen random
backgrounds. The combined representative has eigenvalues `-11.5842`,
`-0.178473`, and `2.49808`. The analytic Hessian matches finite differences to
`1.17e-8`. See
[`SIGMA_V5B_NONLINEAR_DEGENERACY_RESULTS.md`](SIGMA_V5B_NONLINEAR_DEGENERACY_RESULTS.md).

No observational array was opened. The next action must be selected from a
degenerate scalar/vector/tensor class before attaching a transition source or
orientation carrier; changing a v5B parameter cannot repair the rank identity.

## 2026-08-03 v5A cosmological failure and v5B selection

The exact v5A action is retired before data. On FLRW,
`tilde(Q)_a tilde(Q)^a=0`, but generic perturbations make that Lorentzian
invariant either sign. The inherited Sigma-v2 primitive rejects negative `Y`
and its positive-side derivative grows from `-9.51` to `-9999.5` over the
frozen near-zero probe. It has no open real differentiable background domain.
See
[`SIGMA_V5A_COSMOLOGICAL_BRANCH_RESULTS.md`](SIGMA_V5A_COSMOLOGICAL_BRANCH_RESULTS.md).

The already-screened polarization source depends on `Z=Y^2` and is exactly
real, even, and smooth through zero. Sigma v5B therefore places that causal
polarization directly on STEGR/GR, with the same four constants and no MOND
base. Its FLRW background has `sigma=0`, its background metric equations are
GR, and its quadratic TT action has `c_T=c`. Galaxy and cluster departures
must now arise from the same polarization field. See
[`SIGMA_V5B_STEGR_POLARIZATION_ACTION.md`](SIGMA_V5B_STEGR_POLARIZATION_ACTION.md).

## 2026-08-03 Sigma v5A complete static weak variation

The local v5A action now has compact exact metric, flat-connection, and scalar
Euler equations plus the complete leading static equations for `Psi`, `Phi`,
and the polarization. The derivation includes the metric dependence of both
the transition source and orientation-dependent kinetic tensor. Independent
finite differences pass at `1.09e-8` for the transport chain, `1.53e-9` for
the source derivative, and `4.24e-10` for the combined polarization variation.

Massive tracers respond to `-grad(Psi)` and photons to
`W=(Psi+Phi)/2` from the same metric; no photon multiplier is inserted. This
closes the static weak-variation gate, but not the nonlinear mode, FLRW tensor,
cosmological-branch, or PPN gates. See
[`SIGMA_V5A_WEAK_FIELD_DERIVATION.md`](SIGMA_V5A_WEAK_FIELD_DERIVATION.md).

## 2026-08-03 Sigma v5A causal-polarization action screen

The first local causal completion now has a concrete covariant action
candidate. A dimensionless polarization scalar is sourced by the fixed
transition band-pass `x^4/(1+x^4)^2` and propagates with a bounded disformal
inverse metric built from `W_a=Q_a-4 tilde(Q)_a`. The source is `1e-20` at
both `g/a_sigma=1e-5` and `1e5`, peaks at `1/4`, and the scanned local scalar
cone is healthy and no faster than light. Restricting `0<=alpha_sigma<=10`
keeps the minimum kinetic eigenvalue at least `1/11`; the theory uses four
universal constants and has a unique regular static decaying profile.

No observational data were accessed, and no fit is authorized. Complete
metric/connection and weak equations, nonlinear mode count, background
`c_T`, cosmological branch, PPN/Solar response, and prior-art audits remain
hard gates. See
[`SIGMA_V5A_CAUSAL_POLARIZATION_ACTION_AUDIT.md`](SIGMA_V5A_CAUSAL_POLARIZATION_ACTION_AUDIT.md).

## 2026-08-03 Sigma v5 postulates and action selection

The post-v4 rethink now has explicit physical postulates and an action-level
selection result. A plain `F(Sigma) R` coupling changes the two weak potentials
with opposite signs and cancels from their Weyl average, while an unconstrained
tensor--Weyl coupling risks both a hidden homogeneous state and a changed
tensor principal cone. The selected geometric direction instead uses the
nonmetricity trace `W_a=Q_a-4 tilde(Q)_a`, whose static weak square is exactly
`16 |grad((Psi+Phi)/2)|^2/c^4` and which vanishes for linear TT modes.

The resulting Sigma v5 envelope couples a uniquely baryon-forced anisotropic
trace state to that invariant with four provisional universal constants. It is
not yet a complete theory: a causal in-in or degenerate no-free-state action,
its complete functional variation, constraint count, and health proof are
mandatory before another map fit. See
[`SIGMA_V5_ORIENTATION_TRANSPORT_POSTULATES_AND_ACTION_SELECTION.md`](SIGMA_V5_ORIENTATION_TRANSPORT_POSTULATES_AND_ACTION_SELECTION.md).

## 2026-08-03 Sigma v4C and scalar-memory stop decision

The positive baryon-seeded coherence trace passes its uniqueness, positivity,
broadness, high-field, covariance, integral, and padding checks. It is the
strongest v4 projected source, reducing joint spent-map RMSE from `0.907582`
to `0.814737`. It improves PLCKG287 by `20.66%`, but AS295 by only `0.57%`,
worsens one AS295 shear channel, fails both transfer directions, and drives
the high-field scale to its upper bound. See
[`SIGMA_V4C_BARYON_SEEDED_COHERENCE_TRACE_RESULTS.md`](SIGMA_V4C_BARYON_SEEDED_COHERENCE_TRACE_RESULTS.md).

Together, v4A, v4B, and v4C activate the three-failure stop rule for one-scale
isotropic scalar-memory closures. The next lane must derive a baryon-sourced
trace plus orientation-preserving tensor transport from an action before
another map fit. See
[`SIGMA_V4_SCALAR_MEMORY_MECHANISM_FALSIFICATION.md`](SIGMA_V4_SCALAR_MEMORY_MECHANISM_FALSIFICATION.md).

## 2026-08-03 Sigma v4B vector-stress memory result

The lower-derivative projected action built a bounded interaction from the
quadratic total AQUAL field stress and one Helmholtz memory. Its analytic
variation, positive sign, conservation, signed support, numerical stability,
and broad-power gates pass. More than 80% of its correction power lies at
wavelengths of at least 50 kpc, resolving v4A's edge-localization problem.

The shared two-cluster fit nevertheless scores `0.882874` normalized Fourier
RMSE against the `0.907582` AQUAL baseline and the preregistered `0.500` gate.
It improves AS295 by `4.56%`, PLCKG287 by `0.98%`, worsens one of six map
channels, and transfers at `0.928731` and `0.868558` versus the `0.800` gate.
The exact mechanism is retired without opening an untouched observation. The
lesson is sharper: broad vector-stress redistribution is not enough when its
phase and shear geometry do not transfer. See
[`SIGMA_V4B_VECTOR_STRESS_MEMORY_RESULTS.md`](SIGMA_V4B_VECTOR_STRESS_MEMORY_RESULTS.md).

## 2026-08-03 Sigma v3C spent operator inference

The already-opened AS295 and PLCKG287 maps were used to infer the complete
AQUAL-to-halo transfer across convergence and both shear components.  A single
real isotropic transfer was fitted in 22 wavelength bins from 18 to 500 kpc and
then moved to the other cluster.  Same-cluster oracle errors remain
`0.708--0.773`; cross-cluster errors are `0.800--0.956`; and median radial phase
coherence is only `0.276--0.291`.  The two-parameter entire filter scores
`0.800`, and a post-failure lower-length sensitivity only improves this to
`0.787`.

This rejects wavelength-only real linear filtering of the registered source
maps as the missing Hessian mechanism.  The next action must respond to local
tidal eigenstructure, component overlap, or a larger baryonic environment and
carry that information through a uniquely baryon-forced retarded tensor
memory.  No new raw holdout was exposed, so the count of action-level raw
topology failures remains two.  See
[`SIGMA_V3C_SPENT_OPERATOR_INFERENCE.md`](SIGMA_V3C_SPENT_OPERATOR_INFERENCE.md).

## 2026-08-03 Sigma v3B linear nonlocal spectral audit

A scale-dependent one-metric transfer can implement the proposed separation
between locally measured and large-scale gravity.  The no-zero form

$$
T(k^2)=\exp[A\exp(-k^2L_\Sigma^2)]
$$

uses two provisional universal constants, changes the complete manufactured
shear map, retains the luminal massless pole, and with an illustrative
`L_sigma=100 kpc` gives only `3.07e-32` fractional force addition at 1 AU,
`1.000269` force ratio at 10 kpc, and `5.89565` at 500 kpc.

The action-health gate remains decisive.  A standard positive spectral
propagator normalized at high momentum cannot be stronger in the infrared.  A
rational filter achieving the spent `6.7268` cluster/AQUAL amplitude ratio has
a `-5.7268` massive residue.  The entire escape has no extra finite pole but
reverses standard spectral monotonicity and lacks a proved causal Lorentzian
completion.  It is retained as a mathematical clue but not frozen as Sigma v3.
The next lane is a nonlinear retarded tidal interaction whose quadratic
propagator remains Sigma-v1/GR.  This pre-fit result does not increment the two
raw-topology failures.  See
[`SIGMA_V3B_LINEAR_NONLOCAL_SPECTRAL_AUDIT.md`](SIGMA_V3B_LINEAR_NONLOCAL_SPECTRAL_AUDIT.md).

## 2026-08-03 Sigma v3A local DHOST edge audit

The first trace-free local framework after Sigma v2 was screened without
opening a new raw holdout.  A one-parameter `c_T=1`, `beta_1=0`
beyond-Horndeski envelope satisfies the quadratic-DHOST degeneracy identities
to `4.23e-16` relative error and derives the spherical photon correction

$$
\Delta {dW\over dr}=-\pi\alpha_H G r^2\rho_b'(r).
$$

The same derivation supplies a hard amplitude veto.  Positive matter response
in a uniform core requires $\alpha_H<1/3$, limiting any smooth power-law Weyl
enhancement to `18.75%`.  The physically source-scaled correction closes only
`1.53%` of the spent Sigma-v1 convergence gap; even an intentionally
unphysical halo-scaled upper bound closes at most `39.82%`, below the frozen
`75%` advancement threshold.  The local edge term is retired as the sole broad
cluster response.  It does not count as a third raw-topology failure.  The next
derivation target is the causal baryon-forced nonlocal tidal lane.  See
[`SIGMA_V3A_DHOST_EDGE_AUDIT.md`](SIGMA_V3A_DHOST_EDGE_AUDIT.md).

## 2026-08-03 Sigma v2 trace-geometry cycle

The second renewed action cycle added the independent squared second
nonmetricity trace.  This is the smallest geometry-only term that makes the
static time and spatial metric potentials obey different equations while
introducing no material or freely initialized halo state.  Its weak equations
reduce exactly to simple QUMOND matter dynamics with the physical photon
potential fixed to the half-QUMOND, half-Newtonian Weyl average.

The action passes the declared contraction, primitive, deep-limit, high-field,
parameter-count, and external dwarf-galaxy checks.  It scores `12.403 km/s` on
the 13 external dwarfs, exactly the best frozen MOND result.  The fresh raw
lensing calculation uses the repaired registered-map coordinate contract and
recovers only `0.333` of held-out roots in both AS295 and PLCKG287; all held-out
topologies are wrong.  No cluster parameter was fitted.  The action is retired.
See
[`SIGMA_V2_TRACE_NONMETRICITY_ACTION_RESULTS.md`](SIGMA_V2_TRACE_NONMETRICITY_ACTION_RESULTS.md).

Sigma v1 and v2 now independently show that the two minimal local scalar
nonmetricity routes collapse to AQUAL and QUMOND, respectively.  Sigma v3 must
carry a baryon-forced trace-free/tidal state capable of predicting shear
orientation.  A free vector/tensor concentration is disallowed because it
would function as a hidden halo.

## 2026-08-03 Sigma v1 pure-geometry cycle

The renewed action-first goal has now tested the smallest one-metric,
baryon-only symmetric-teleparallel action.  The nonlinear nonmetricity action
passes its invariant, deep-field, high-field, parameter-count, and external
dwarf-galaxy gates.  Its regular isolated weak-field equations prove
`Phi=Psi` and reduce exactly to standard-mu AQUAL.  It therefore inherits the
frozen AQUAL raw-lensing result: `0.333` root convergence in both ready
clusters and incorrect held-out topology.  The action is retired without a
fit.  See [`SIGMA_V1_NONMETRICITY_ACTION_RESULTS.md`](SIGMA_V1_NONMETRICITY_ACTION_RESULTS.md).

This closes the pure one-invariant geometric route.  Any next action must add
a baryon-predictable vector/tensor or causal nonlocal state that supplies
anisotropic stress; another scalar interpolation of the same invariant is not
a materially new cycle.

Status: active, updated 2026-07-29. The thresholds were recorded before H7a or
H7s was scored. Neither candidate is advanced by changing a bound after the
result.

The later unbounded curvature-running cycle is documented separately in
`docs/UNBOUNDED_CURVATURE_RUNNING_RESULTS.md`. Its best balanced setting passes
the Solar-System screening proxy and the broad BCG/cluster gates, and improves
on the fitted NFW galaxy reference, but it remains 39% worse than fixed RAR on
untouched SPARC outskirts and has unacceptable raw-lensing chi-square. It is
retained as a phenomenological control, not advanced as a theory survivor.
The subsequent locked multi-cluster raw-image transfer also fails: the two
controls score 18.2--18.6 arcsec equal-system held-out RMS on four raw coordinate
likelihoods, compared with 9.05 arcsec for an inadequate compact halo. A
post-failure per-cluster amplitude grid is non-universal and does not pass the
rescue gate. The first spatial-vector target is now complete as well. A frozen
mass-conserving redistribution of the same baryonic monopole into 63--120
observed member-light directions per cluster gives 18.2--18.7 arcsec and makes
every predictive score slightly worse. A post-failure all-root-converged oracle
over all 148 universal grid settings improves the best parent by only 1.6% and
still has more than twice the compact-halo error. Member light alone is not the
missing source variable. A common-200-kpc aperture correction also fails, with
18.210 arcsec for its best all-root setting versus 18.165 arcsec for the parent.
A further cluster test requires complete gas, BCG, ICL, and satellite
surface-density maps rather than a covariant completion of either failed scalar
closure.

## Current decision

The frozen measured/profile-constrained Stage 4 cycle supports the baryonic-
potential environment variable, but none of the attempted local closures survives.
H7s remains on its Stage 3 hard bound. The subsequent five-parameter EA-Q0
action passes its conservation, unit-vector, mode-speed, and quasistatic checks,
but fails before fitting: the reciprocal Aether source changes the dynamical
environment field by orders of magnitude more than the allowed 5%. EA-Q0 is
retired. The five-parameter EMOG-Q0 control then passes its local field-health,
conservation, short-distance, large-distance, and one-metric lensing derivations,
but its monotone chameleon response and one universal Yukawa range fail the
joint 5% structural target. EMOG-Q0 is also retired. The declared next stage is
a premise-level rethink, not another interpolation term.

## Gate scoreboard

| Stage/outcome | Concrete threshold | Result | Decision |
|---|---:|---:|---|
| SPARC inverse usable | at least 70% | 93.84% | pass |
| CLASH inverse usable | at least 70% | 100% | pass |
| Analytic inverse round trip | max relative error $\le10^{-10}$ | $4.45\times10^{-16}$ | pass |
| Central galaxy/cluster $\log_{10}\chi$ overlap | diagnostic | 0 dex; 0.462-dex central gap | warning |
| H7a SPARC $\chi^2/N$ | $\le9.41784$ | 9.138 | pass |
| H7a raw CLASH $\chi^2/N$ | $\le5.00$ | 4.809 | pass |
| H7a macro $\chi^2/N$ | $\le7.20$ | 6.974 | pass |
| H7a parameters off bounds | all five folds | width at lower bound in 4 folds | **fail** |
| H7s SPARC $\chi^2/N$ | $\le9.41784$ | 9.220 | pass |
| H7s raw CLASH $\chi^2/N$ | $\le5.00$ | 4.114 | pass |
| H7s CLASH with scatter | $\le2.50$ | 2.286 | pass |
| H7s CLASH RMS | $\le0.160$ dex | 0.156 dex | pass |
| H7s macro $\chi^2/N$ | $\le7.20$ | 6.667 | pass |
| H7s parameters off bounds | all five folds | $F=100$ in 3 folds | **fail** |
| Original 50 BCG eRASS gas-scale coverage | at least 30 systems and 80% | 1 system; 9.1% of public-footprint subset | **fail** |
| SPIDERS-MaNGA bridge systems | at least 30 | 34 unique hosts | pass |
| SPIDERS host-scale coverage | at least 80% | 100% | pass |
| Disjoint JAM proxy calibration | diagnostic RMS | 0.093 dex $g_{\rm obs}$; 0.098 dex $g_{\rm bar}$ | usable |
| Local-only H7s BCG $\chi^2/N$ | $\le5.0$ | 5.218 | **fail** |
| Local-only H7s BCG mean residual | $|\bar\Delta|\le0.15$ dex | -0.227 dex | **fail** |
| eRASS-median gas host $\chi^2/N$ | $\le5.0$ | 3.982 | pass |
| eRASS-median gas host mean residual | $|\bar\Delta|\le0.15$ dex | -0.184 dex | **fail** |
| Cosmic-baryon host $\chi^2/N$ | $\le5.0$ | 2.814 | pass |
| Cosmic-baryon host RMS | $\le0.17$ dex | 0.168 dex | pass |
| Cosmic-baryon host mean residual | science: $|\bar\Delta|\le0.10$ dex | -0.135 dex | **fail** |
| Frozen profile-constrained systems | at least 30 | 34/34 | pass |
| Independent satellite-catalog union | at least 30 | 30/34 | pass |
| Measured/profile host $\chi^2/N$ | science: $\le3.0$ | 1.658 | pass |
| Measured/profile host RMS | science: $\le0.17$ dex | 0.132 dex | pass |
| Measured/profile host mean residual | science: $|\bar\Delta|\le0.10$ dex | -0.083 dex | pass |
| Stage 4 uncertainty pass probability | continue $\ge0.80$; science $\ge0.50$ | 1.000; 1.000 | pass |
| EA-Q0 global parameters | at most 5 | 5 | pass |
| EA-Q0 tensor speed | $|c_T/c-1|\le10^{-15}$ | $c_{13}=0$, $c_T=c$ | pass |
| EA-Q0 deep/high-field limits | 5%; $10^{-5}$ | $2.50\times10^{-4}$; $5.00\times10^{-11}$ | pass |
| EA-Q0 BCG $Q$ reproduction | fractional change $\le0.05$ for all 34 | minimum lower bound 216 | **fail** |
| EA-Q0 allowed/required response | diagnostic | required $\eta$ is 84,469 times allowed | **fail** |
| EMOG-Q0 global parameters | at most 5 | 5 | pass |
| EMOG-Q0 local mode health | positive kinetic/gradient; $c_T=c$ | all principal speeds $c$ | pass |
| EMOG-Q0 universal-vector Solar bound | $|\gamma_{\rm eff}-1|\le2.3\times10^{-5}$ | $\alpha\le1.15\times10^{-5}$; favorable target envelope needs $\sim0.98$ | **fail** |
| EMOG-Q0 $1/r$ radial support | broad observed support | 0.055 dex for slope $-1\pm0.05$ | **fail** |
| EMOG-Q0 CLASH environment ordering | pointwise error $\le0.05$ | analytic lower bound 0.590 | **fail** |
| EMOG-Q0 joint force target | all SPARC/CLASH/BCG points within 5% | 0 points pass in favorable envelope | **fail** |

H7s numerically clears the stronger science-score targets, but the bound failure
has priority. Widening $F$ now would turn the test into post-hoc parameter
search. Its paired macro improvement over U0 also does not exclude zero: the
95% interval is -1.280 to +0.293 per point.

## What Stage 1 established

The pointwise reconstruction retained every row and marked
$g_{\rm obs}\le g_{\rm bar}$ points as unavailable for analytic inversion.
Across valid points, the median RAR-equivalent scales are

$$
\log_{10}a_{\rm eff}=-9.922\quad\text{(SPARC)},
$$

$$
\log_{10}a_{\rm eff}=-8.722\quad\text{(CLASH)}.
$$

The broad frozen U0 transition contains 31 SPARC systems and all 20 CLASH
systems, so the declared count gate passes. However, the central 10--90%
potential supports do not overlap. The CLASH 10th percentile lies 0.462 dex
above the SPARC 90th percentile. This makes the intermediate BCG regime a
required identification test rather than an optional external check.

Artifacts:

- `results/constitutive_target/report.json`
- `results/constitutive_target/constitutive_target.png`
- `scripts/reconstruct_constitutive_target.py`

## What the two action-derived cycles established

H7a and H7s use the same three global parameters, no per-object force term, and
no lensing-only multiplier. They differ only in the constitutive derivative of
the weak-field action:

$$
\mu_a(x)=\frac{x}{1+x},
\qquad
\mu_s(x)=\frac{x}{\sqrt{1+x^2}}.
$$

Both preserve SPARC while improving CLASH over U0. This is evidence that an
action-derived nonlinear Poisson limit can reproduce the useful part of U0; it
is not evidence that the potential transition or a void origin has been
established. The repeated boundary behavior and separated sample supports are
the structural warning.

Artifacts:

- `docs/H7A_WEAK_FIELD_DERIVATION.md`
- `results/h7a_cv/report.json`
- `results/h7s_cv/report.json`
- `scripts/cross_validate_h7a.py`

## Independent BCG bridge and host-scale result

The direct eRASS1 optical-BCG cross-match yields only 25 unique MaNGA BCGs, so
it cannot satisfy the frozen count gate. The declared sample rethink combines
three official products without using an acceleration value to select a system:

- GEMA-VAC identifies the brightest galaxy in each MaNGA group.
- SPIDERS supplies spectroscopically confirmed X-ray hosts, luminosities, and
  $R_{200}$ estimates.
- MaNGA DynPop supplies quality-assessed mass-follows-light JAM dynamics.

The frozen 200-kpc and $|\Delta z|\le0.01(1+z)$ match gives 34 unique hosts.
Eleven have direct Tian et al. accelerations. For the other 23, the DynPop/NSA
acceleration proxy is calibrated on 33 disjoint Tian systems; no test system
calibrates itself. The calibration RMS is 0.093 dex for $g_{\rm obs}$ and 0.098
dex for $g_{\rm bar}$, and those scatters enter the proxy uncertainties.

The frozen full-development H7s fit remains at $F=100$, so this cannot repair
its Stage 3 identifiability failure. It nevertheless supplies a useful external
diagnostic. With only BCG baryonic potential it gives $\chi^2/N=5.218$, RMS
0.253 dex, and mean residual -0.227 dex. The direct and proxy subsets have
nearly identical mean residuals, arguing against the proxy calibration being
the source of the missing acceleration.

The first host completion contains no fitted BCG parameter:

$$
M_{200}=\frac{4\pi}{3}200\rho_c(z)R_{200}^3,
\qquad
\chi_{\rm host}=\frac{Gf_bM_{200}}{R_{200}c^2},
$$

with $f_b=\Omega_b/\Omega_m$ from Planck18. This optimistic retained-cosmic-
baryon scale improves the score to 2.814, the RMS to 0.168 dex, and the mean
residual to -0.135 dex. It passes the continue gate but misses the 0.10-dex
scientific bias limit.

That upper-bound result is not enough. Across 10,440 eRASS1 systems the catalog
$f_{\rm gas,500}$ median is 0.064 and the 90th percentile is 0.098, versus a
larger cosmic baryon fraction. Applying those independent gas fractions to the
same SPIDERS potential scale gives:

| Host input | $\chi^2/N$ | RMS (dex) | Mean residual (dex) | Continue? |
|---|---:|---:|---:|---|
| none | 5.218 | 0.253 | -0.227 | no |
| eRASS median gas fraction | 3.982 | 0.210 | -0.184 | no |
| eRASS 90th-percentile gas fraction | 3.493 | 0.193 | -0.165 | no |
| retained cosmic baryon fraction | 2.814 | 0.168 | -0.135 | yes, not science success |

Thus measured hot gas at the catalog scale helps but is insufficient by itself.
The remaining physically allowed terms are the host's stellar/satellite baryons
and the central weighting of the gas profile. Neither may be normalized using
the BCG residual.

Artifacts:

- `configs/bcg_bridge_sample.json`
- `scripts/build_and_test_bcg_bridge.py`
- `data/derived/bcg_bridge_sample.csv`
- `results/bcg_bridge_sample/report.json`
- `results/bcg_bridge_sample/predictions.csv`

All source hashes and selection rules are recorded in the report. Large FITS
files remain local and reproducible through
`scripts/download_bcg_environment_catalogs.ps1`.

## Completed measured/profile-constrained host cycle

The archival audit found direct eRASS or pointed Chandra/XMM coverage for only
10 of the 34 frozen hosts. It would therefore be inaccurate to call this a
30-host directly measured X-ray-profile test. The preregistered alternative is
the profile-constrained route:

- each host retains its measured SPIDERS $R_{200}$ and derived halo scale;
- a 10,439-system eRASS calibration fixes the gas-mass relation and its scatter;
- 46 published Chandra density profiles fix the gas radial-shape population;
- each BCG's NSA Sersic profile supplies the stellar exterior-shell correction;
- a published 21-cluster satellite-mass relation and redMaPPer radial scale fix
  the satellite population; and
- SPIDERS/redMaPPer provide an independent member-catalog coverage check for
  30 of 34 systems, although those members are not normalized with BCG data.

For a spherical component truncated at $R$, the potential is integrated as

$$
\chi_b(r)=\frac{G}{c^2}\left[
\frac{M_b(<r)}{r}+\int_r^R\frac{dM_b(s)}{s}
\right].
$$

This corrects the point-scale approximation by including exterior shells and
the central weighting of extended gas and satellite profiles. It introduces no
host normalization. All 34 systems are scored with the unchanged H7s vector.

| Result | Point estimate | 5--95% Monte Carlo interval | Gate |
|---|---:|---:|---|
| $\chi^2/N$ | 1.658 | 1.493--1.818 | $\le3.0$ |
| RMS | 0.132 dex | 0.125--0.139 dex | $\le0.17$ dex |
| Mean residual | -0.083 dex | -0.087 to -0.070 dex | absolute value $\le0.10$ dex |

The direct Tian subset has $\chi^2/N=2.764$, RMS $=0.140$ dex, and mean
residual $=-0.100$ dex; the 23 calibrated proxy systems give 1.129, 0.128 dex,
and -0.074 dex. The agreement is not driven solely by the proxy subset. Every
one of the 5,000 uncertainty realizations passes both declared Stage 4 gates.

Artifacts:

- `configs/measured_host_profile_validation.json`
- `scripts/download_host_profile_catalog.py`
- `scripts/inventory_host_profile_coverage.py`
- `scripts/validate_measured_host_profiles.py`
- `data/derived/host_profile_coverage.csv`
- `data/derived/measured_host_profile_sample.csv`
- `results/measured_host_profiles/coverage_report.json`
- `results/measured_host_profiles/report.json`

## EA-Q0 derivation result

The selected local EA-Q0 action has one minimally coupled physical metric, a
unit timelike Aether, and a scalar-curvature environment field. Its five global
parameters are $\{\beta,L_Q,\eta,c_1,c_{14}\}$; $c_{13}=0$ exactly and no
per-object or lensing-only parameter is present. The scalar, Aether, constraint,
and metric equations were varied from the same action, and their diffeomorphism
Noether identity verifies on-shell stress-energy conservation.

The action produces the desired standard-$\mu$ static response and healthy
declared high-/low-field limits. It also necessarily produces the reciprocal
source

$$
(\Box-L_Q^{-2})Q=-\frac{R}{2}
-\frac{\eta a_Q^2}{2\beta}
\left(\mathcal H_s-Y\mathcal H_{s,Y}\right).
$$

The second term cannot be removed without abandoning the action. The frozen
spherical check maximizes $\beta$ under the PPN $\gamma$ gate, uses the shortest
field range consistent with 5% potential accuracy, omits all exterior baryons,
and continues only already enclosed mass. Even this lower bound changes $Q$ by
factors of 216--5,211 across the 34 BCGs. None passes the 5% gate. The required
environment response is 84,469 times stronger than the largest response that
keeps every BCG within the gate.

Artifacts:

- `configs/eaq0_derivation.json`
- `docs/EAQ0_DERIVATION.md`
- `src/voidscreen/eaq.py`
- `scripts/check_eaq0_derivation.py`
- `results/eaq0_derivation/report.json`
- `results/eaq0_derivation/feedback_points.csv`

## EMOG-Q0 result and premise-level rethink

The [environmental MOG control](ENVIRONMENTAL_MOG0_DERIVATION.md) freezes one
physical metric, a canonical chameleon scalar, a positive-energy Proca vector,
and one conserved composition-independent charge. Its five parameters are
$\{\beta,\Lambda_s,n,\mu,\alpha\}$. The full metric, scalar, vector, and matter
equations follow from the same action, and the diffeomorphism Noether identity
verifies their on-shell conservation. The regular $F>0$ field domain has
positive scalar and Proca kinetic terms, luminal principal characteristics, and
$c_T=c$. The same metric potential supplies the light-deflection prediction.

At a constant scalar background the massive-particle law is

$$
{g_{\rm dyn}\over g_{\rm bar}}={1\over F_0}
-\alpha(1+\mu r)e^{-\mu r}.
$$

The matched condition $F_0^{-1}=1+\alpha$ recovers Newtonian gravity at small
$r$ and enhanced attraction at large $r$. It does not solve the radial-shape
problem: the extra point-mass acceleration has slope $-1\pm0.05$ for only
$1.682<\mu r<1.908$, or 0.055 dex. Nor can the matching persist when the scalar
responds to environment, because $F(s)$ changes while the conserved vector
charge $\alpha$ is universal.

The action's adiabatic scalar response predicts $F^{-1}$ increasing as mean
baryonic density decreases. CLASH alone contains a lower-density point
requiring enhancement 4.217 and a higher-density point requiring 16.368. Any
response with the action's ordering must miss at least one by 59.0%, independent
of optimizer or power-law details. A deliberately favorable global envelope,
which lets the scalar follow its density minimum instantly, finds no point in
SPARC, CLASH, or the 34 BCGs within the 5% gate.

Artifacts:

- `configs/environmental_mog0_derivation.json`
- `docs/ENVIRONMENTAL_MOG0_DERIVATION.md`
- `src/voidscreen/mog.py`
- `scripts/check_environmental_mog0.py`
- `results/environmental_mog0/report.json`
- `results/environmental_mog0/feasibility_points.csv`

EMOG-Q0 is retired before fitting; no Stage 3 or Stage 4 configuration is
frozen. All three declared relativistic completion routes have now failed a
pre-fit structural or identifiability gate. Under the stopping rule, the next
cycle must revisit the one-field environmental-unification premise and the
meaning of the 5% pointwise target. Adding an interpolation term, making the
range object-dependent, or introducing a lensing-only amplitude is forbidden.
The concrete R0--R3 checkpoints are frozen in
[`PREMISE_LEVEL_RETHINK.md`](PREMISE_LEVEL_RETHINK.md); R0 begins with the raw
observable and covariance provenance behind the CLASH and BCG targets.

## Universal variable-exponent result

The curvature-running exponent was promoted from a constant to the bounded
universal function

$$
p(X)=p_0\exp[\beta\tanh(\ln(X/X_*))],
$$

using force-equivalent enclosed baryonic mass, local baryonic density, or the
local-to-mean density ratio as $X$. Five gravity constants were fit on the
system-held-out BCG+cluster bridge and transferred unchanged to 131 SPARC
galaxies. The mass version improved the bridge to 0.1158 dex but failed SPARC at
61.54 km/s; density scored 0.1241 dex and 38.87 km/s. The best transfer was the
distribution-shape version at 0.1396 dex and 15.26 km/s, but it saturated at
$p=5.0004$ throughout the bridge, effectively returning to a constant exponent.
The fixed $p=2$ control remains better at 0.1377 dex and 14.40 km/s and also
retains all RX J2129 image roots. No variable-exponent candidate advances. See
[`VARIABLE_EXPONENT_RESULTS.md`](VARIABLE_EXPONENT_RESULTS.md).

## Galaxy-locked metric-slip result

The photon/matter distinction was isolated without changing the galaxy force
law. Four smooth curvature matter laws were calibrated only on inner SPARC
radii and failed untouched outer radii at 82.81--88.38 km/s, versus 10.35 km/s
for fixed RAR. Fixed RAR was therefore locked as the matter potential before
any lensing data were used.

The weak-field split

$$
\Phi=\Phi_N+\phi,\qquad \Psi=\Phi_N+(1+s)\phi
$$

was then tested on raw cluster image positions. One shared, complete-root value
$s=5$ was selected on MACS0329 and MACS0429 and transferred unchanged to
MACS1115 and MACS1931. It lowers the unseen equal-system radial RMS from 25.67
to 18.43 arcsec, but the compact-halo control reaches 9.99 arcsec. The far-tail
control changes the result by only 0.51%, so the failure is not caused by the
RAR integration cutoff.

The same slip improves the secondary 20-system radial lensing score from 0.509
to 0.161 dex, showing that a light/matter amplitude difference is useful.
However, it cannot alter the fixed-RAR BCG dynamics, which remain at 0.299 dex
RMSE and only 55.3% of observed acceleration at the median. Raw-image failures
and inconsistent cluster preferences show that one scalar amplitude lacks the
required spatial structure. The candidate is retired; the next justified test
is a universal tidal-tensor slip evaluated with explicit member and gas maps.
See [`METRIC_SLIP_RESULTS.md`](METRIC_SLIP_RESULTS.md).

## Spherical spacetime and hard-cavity result

The proposed spherical-medium picture was separated into an exact closed
three-space Gauss law and an exact impermeable-sphere potential-flow analogy.
For a closed three-space the force enhancement is

$$
{g\over g_{\rm bar}}=\left[{r/L\over\sin(r/L)}\right]^2.
$$

Keeping the same geometry valid through the 3,000-kpc raw-lensing integral
forces $L$ above 1,005 kpc. The fit reaches its lower bound at 1,096 kpc and is
then too weak on galaxies: 72.39 km/s outer SPARC RMSE versus 10.35 km/s for
fixed RAR. A galaxy-only curvature radius fails at 88.74 km/s and reaches its
antipodal singularity before cluster scales. A screened local-curvature variant
improves BCG dynamics to 0.248 dex but catastrophically extrapolates to 177.97
km/s and becomes invalid on clusters.

For a hard spherical cavity the linear directional correction scales as
$(a/r)^3$, cancels in the isotropic average, and leaves an RMS correction of
order $(a/r)^6$. Even treating an entire disk scale as a perfectly hard cavity
gives only a 1.0027 median favorable-axis factor against a required 3.8476;
none of 960 outer points is reached. Real stellar covering fractions are below
$2.2\times10^{-11}$ in the generous upper-bound calculation.

The frozen post-failure raw transfer scores 25.15 arcsec on unseen cluster
images, indistinguishable from baryons at 25.20 and much worse than the compact
halo at 9.99. The result changes only 0.75% across 600--3,000-kpc cutoffs. The
literal global-sphere and hard-cavity candidates are retired. The remaining
direction is a sourced, conservative tensor constitutive law, not another
spherical amplitude. See
[`SPHERICAL_SPACETIME_CAVITY_RESULTS.md`](SPHERICAL_SPACETIME_CAVITY_RESULTS.md).

## Sigma v4A variational-source result

After three scalar tidal-memory scores failed the same synthetic morphology
gate, the strongest commutator interaction was varied to obtain its complete
signed Euler--Lagrange Weyl source. The projected source is conserved, has
both signs, passes its analytic derivatives at `6.27e-12` relative error, and
selects the physically allowed positive action coefficient. It improves all
six convergence/shear channels across spent AS295 and PLCKG287 maps.

The improvement is only `0.398%` in joint RMSE: `0.907582` becomes `0.903971`,
far above the frozen `0.500` gate. It explains `0.794%` of weighted missing
field power. Cross-cluster transfers score `0.913907` and `0.894538` versus
the required `0.800`, while changing the padding boundary alters the result by
only `1.42e-8` fraction. The exact source is retired without opening a
holdout. The result closes the possibility that the v3E scalar failed only
because its earlier volume score discarded the sign. See
[`SIGMA_V4A_PROJECTED_VARIATIONAL_SOURCE_RESULTS.md`](SIGMA_V4A_PROJECTED_VARIATIONAL_SOURCE_RESULTS.md).
