# Sigma Gravity galaxy--cluster research wrap-up

Date: 2026-08-06

Status: deliberate stopping point after the complete CCD7 background archive
repair and before the 494-region thermodynamic likelihood, I4/I5 source test, a
new gravity equation, or any untouched lensing/galaxy holdout is opened.

## Executive conclusion

We have not discovered a replacement for dark matter or MOND. We have built a
rigorous research program, identified one unusually strong empirical
galaxy--cluster bridge, ruled out many simpler mechanisms, isolated the main
raw-lensing failure, and prepared a target-blind cluster source-data pipeline
capable of testing the next physical idea.

The strongest empirical bridge remains RAR plus a squared coherence-gated
refracted-gravity response. It is only 1.94% worse than fixed MOND on 131 SPARC
outer rotation curves and is much closer than galaxy-scale MOND to 20
NFW-derived CLASH radial fields. It also gave an encouraging one-cluster raw
lensing result. It is not a fundamental theory: RAR and refracted gravity are
published ingredients, coherence is not yet one covariant observable, and the
formula has not transferred across raw multi-cluster lensing topology.

The deepest observational lesson is that the cluster problem is not mainly
"more gravity." A successful equation must put convergence and shear in the
right locations and orientations so that it produces the observed image roots,
parities, folds, cusps and critical curves. Scalar radial enhancement generally
changes deflection amplitude without generating that spatial Hessian.

The most promising physical clue is component-wise nonlinear baryonic response:

\[
\mathcal N\!\left(\sum_i \rho_i\right)
\ne
\sum_i \mathcal N(\rho_i).
\]

That distinction is small in a smooth, centrally dominated galaxy but can be
large in a cluster containing separated gas peaks, a BCG, intracluster light
and many member galaxies. It can create more lensing structure without an
object label, but it still needs a continuum-safe field equation and a measured
baryonic source.

The next measurement-first candidates are I4 thermodynamic-gradient stress,
which can supply a direction, and I5 baroclinicity, which can supply scalar
activation. The source pipeline is now ready to test them, but neither has been
constructed or scored. That is the recommended next research chapter.

## The non-negotiable theory goal

The target is one universal baryon-only field theory that:

1. derives massive-particle dynamics and photon lensing from one physical
   metric;
2. introduces no invisible matter;
3. uses no galaxy/cluster, dwarf/giant, relaxed/merger or morphology switch;
4. has no lensing-only multiplier and no per-object gravity parameter;
5. uses at most five universal physical constants;
6. freezes its equation and constants before validation and holdout data are
   opened;
7. matches diverse galaxy dynamics and raw multi-cluster lensing topology;
8. has a healthy covariant action, conservation law, stable degrees of freedom,
   causal propagation, numerical convergence and a GR/Solar-System limit.

"Replicating dark matter" means predicting the raw observations that motivate
dark matter. Reproducing an NFW acceleration or halo map inferred under GR is a
development diagnostic, not independent validation.

## Best empirical formula

The strongest broad bridge is

\[
\frac{g}{g_{\rm bar}}
=
E_{\rm RAR}(g_{\rm bar})
+
[1-w(C)]^2
\left[\frac{1}{\epsilon(\rho_b)}-1\right],
\]

with

\[
E_{\rm RAR}
=
\frac{1}{1-\exp[-\sqrt{g_{\rm bar}/g_\dagger}]},
\qquad
g_\dagger=1.2\times10^{-10}\ {\rm m\,s^{-2}},
\]

\[
w(C)=3C^2-2C^3,
\]

and

\[
\epsilon(\rho_b)
=
\epsilon_0+(1-\epsilon_0)
\left[1+\exp\left(-2Q\ln\frac{\rho_b}{\rho_c}\right)\right]^{-1}.
\]

The three fitted bridge constants are

\[
\epsilon_0=0.1295,
\qquad
\rho_c=10^{-23.785}\ {\rm g\,cm^{-3}},
\qquad
Q=0.4272.
\]

In plain language:

- `g_bar` is the Newtonian acceleration from observed stars and gas.
- `E_RAR` supplies the well-established low-acceleration galaxy behavior.
- `epsilon(rho_b)` makes an additional response available in diffuse regions.
- `C` represents coherent rotational organization.
- The squared gate suppresses the density response in orderly disks while
  leaving more of it available in disordered, diffuse systems.

This explains why the interpolation works. It also explains why it is not yet
an elegant root equation: it combines two successful phenomenologies and the
definition of coherence has not emerged from one covariant local state.

## Best numerical results

| Test | Candidate | Comparator | Honest interpretation |
|---|---:|---:|---|
| 131 SPARC galaxies, 968 held-out outer points | 10.586 km/s | fixed simple MOND: 10.385 km/s | candidate is only 1.94% worse |
| 20 CLASH NFW-derived radial fields | 0.1387 dex | fixed galaxy MOND: 0.5184 dex | strong derived-field bridge, not independent truth |
| Mean predicted/required CLASH field | 0.999 | ideal: 1.000 | average amplitude is essentially correct |
| CLASH points within factor 1.5 | 81.9% | fixed MOND: 1.4% | large amplitude improvement |
| RX J2129 seven held-out raw image positions | 1.064 arcsec | compact one-halo control: 2.536 arcsec | encouraging one-cluster result |
| RX J2129 all-image comparison | 0.618 arcsec | published 71-halo model: 0.29 arcsec | still about twice the published error |

A later frozen two-potential candidate reached 10.735 km/s on 13 new dwarf
galaxies versus 12.403 km/s for its best frozen full-field MOND comparator and
22.070 km/s for baryons-only Newtonian gravity. It did not transfer to cluster
lensing: it often predicted one image where the data contained two to nine.

Component-wise nonlinear experiments increased root completeness to 85.1% in
one cluster and 55.6% in another, but image positions remained roughly 13--21
times worse than halo comparators. This is a mechanism clue, not a passing
formula.

No tested universal formula has yet matched both trusted galaxy controls and
raw multi-cluster dark-matter lens models with one setting.

## What we explored

### Void and negative-gravity ideas

We tested inward pressure from surrounding voids, void distance and midpoint
proxies, smooth void-wall fields, low-acceleration activation, external tides
and potential-boundary layers. Uniform external acceleration cancels from
internal relative motion, while the leading smooth internal term is tidal or
harmonic. Measured ordinary large-scale tides were far too small in the tested
systems. These experiments did not yield a universal galaxy--cluster law.

The useful residual idea is not a literal `-9.8 m/s^2 / R` void force. It is
that boundary conditions or a nonlocal baryon-sourced field could redistribute
gravitational response. Such a field must carry conserved flux/energy and
predict its own scale rather than infer it from halo residuals.

### Variable G, distance laws and exponents

We varied effective gravitational strength, distance falloff, mass-, density-
and concentration-dependent exponents, Solar-screened isothermal tails and
wave/catch-up scales. Some variants improved either rotation curves or cluster
amplitudes, but universal transfer failed. Allowing the exponent to depend on
object properties usually recreated a MOND-like acceleration law or became an
implicit object classifier.

The causal catch-up completion retained the empirical bridge's static scores,
but its time term is exactly invisible in static galaxy and lensing tests. A
long gravity wavelength is therefore not enough by itself; it needs a
time-dependent prediction that can be measured independently.

### MOND, RAR and refracted gravity

MOND/RAR remains the strongest simple galaxy benchmark. Refracted-gravity-like
density response can provide cluster-sized amplitude in diffuse environments.
The empirical bridge combines them smoothly, but until one field equation
causes both limits, it remains "MOND/RAR here and refracted response there."

Potential-dependent AQUAL variants were competitive phenomenologically but
largely recreated MOND/AQUAL and used a noncovariant environment definition.
The unique part of a future Sigma theory cannot merely be a rearranged MOND
interpolation function.

### Scalar, tensor, nonlocal and diffusion routes

We tested scalar curvature responses, metric slip, member-light vectors,
member-tidal tensors, conservative radial diffusion, one-sided memory,
nonlocal density morphology, porosity, self-coupling and component routing.

The consistent lesson is:

- scalar amplitude is not enough for raw lens topology;
- simply adding member-light vectors does not transfer;
- strong tensor couplings may improve a local cost while losing exact image
  roots;
- diffusion trades galaxy accuracy against lensing and can destroy roots;
- a nonlinearity must be independent of arbitrary source segmentation.

### Spherical-spacetime and cavity analogies

Closed-sphere amplification and hard cavity-flow analogies were tested. The
global spherical laws badly damaged galaxy fits. The hard-cavity far-field
effect is too small and its analytic net force vanishes. These literal
geometric analogies are retired.

### Photon/QED-only explanations

We considered whether apparent rotation or lensing could arise because photon
paths add differently from massive-particle gravity. Existing galaxy rotation
measurements use electromagnetic signals, so a photon-only effect can in
principle bias inferred motion. But it must also preserve achromatic lensing,
spectral lines, time delays, polarization and observations made with different
messengers.

No tested photon-only formula reproduced the full galaxy and cluster evidence.
The academically defensible route is still one physical metric unless a new
photon coupling makes a distinctive, independently testable prediction. A
double-slit or "non-space" interpretation is speculative and was not treated
as empirical evidence.

### Covariant action attempts

We explored conformal/disformal scalars, AQUAL-like actions, Aether/khronon
completions, Proca/vector sectors, convex reduced carriers, gauge-like tensor
carriers and reciprocal Sigma actions. Individual pieces sometimes supplied a
healthy weak-field response, but complete candidates failed through wrong
radial shape, invisible dust-like charge, ghosts, unbounded energy, superluminal
or ill-posed characteristics, loss of the GR limit, or failure to lens from one
metric.

No healthy covariant action currently produces the empirical bridge and raw
lensing topology. This remains a central unsolved requirement.

## What we learned about galaxies versus clusters

Galaxies are often close to one dominant, approximately ordered rotating
configuration. Their rotation curves constrain the radial first derivative of
the dynamical potential. MOND/RAR is exceptionally effective because a nearly
universal acceleration transition captures that behavior.

Clusters are multi-centered, pressure-supported and dynamically complex. Raw
strong lensing constrains second spatial derivatives of the Weyl potential.
The equation must create the correct convergence and directional shear around
gas peaks, the BCG, intracluster light and member galaxies. Matching a radial
mass profile or average deflection is insufficient.

This does not justify assigning light and mass unrelated laws. It says a single
metric must contain more spatial structure than a scalar radial enhancement.
The candidate source and propagator probably need a tensor or nonlocal
component.

## Current status by theory requirement

| Requirement | Status at wrap | Evidence |
|---|---|---|
| Competitive galaxy amplitude | promising | empirical bridge is 1.0194 times fixed MOND RMSE on held-out SPARC outer points |
| Morphology-diverse full galaxy holdout | incomplete | many strata tested, but final full-curve WALLABY/morphology gate not passed |
| Cluster radial amplitude | promising diagnostic | 0.1387 dex on NFW-derived CLASH fields |
| Raw multi-cluster image topology | failed so far | no universal candidate passes roots, topology and halo-relative positions |
| Halo size/shape derived from baryons | incomplete | component-wise nonlinearity is a clue, not a derived mapping |
| Independent baryonic source variable | ready to test | I4/I5 inputs now have a corrected response archive |
| One metric for matter and light | missing | no surviving derivation of both potentials from one action |
| Healthy covariant action | missing | explored candidates retired or incomplete |
| At most five universal constants | empirical bridge passes counting | three bridge constants plus fixed RAR scale; coherence definition remains noncovariant |
| Per-object gravity parameters | zero in the bridge | measurement nuisances remain separate |
| Solar/relativistic consistency | partial diagnostics only | no final theory to test completely |
| Numerical infrastructure | strong but formula-dependent | solver, convergence, injection and observation adapters exist; final equation must rerun them |
| Cosmology and structure formation | largely untested | intentionally deferred until low-redshift galaxy/lensing transfer is credible |

## Infrastructure and data completed

The project now contains:

- a formula and prior-art registry;
- a machine-readable formula scorecard and master validation matrix;
- frozen development/validation/holdout conventions;
- SPARC galaxy and CLASH-derived comparison pipelines;
- raw strong-lensing root, topology and image-position tests;
- morphology, dwarf/giant, gas fraction and surface-density diagnostics;
- a simulator/API that can evaluate supplied formulas, compare synthetic or
  observation-matched galaxies and produce reports;
- Chandra source-map, astrometry, response, background and regional spectral
  machinery for the Bullet Cluster and Abell 2146;
- deterministic hashes, checkpointed products and failure-preserving reports.

The simulator is useful for forward-model comparisons and sensitivity studies.
It is not yet a generative physical universe that can infer a fundamental law
from dark-matter maps, and synthetic agreement cannot replace held-out raw
observations.

## Latest source-data chapter: V19DP--V19DR

V19DP jointly fitted one registered region per cluster while retaining every
observation's PHA, background, ARF and RMF. It used one shared temperature,
abundance and normalization with no per-observation free parameter.

- Bullet bin 169 passed at reduced statistic 0.763083.
- Abell 2146 bin 62 initially failed at 1.949652.
- Omitting ObsID 10464 CCD7 lowered Abell to 1.327141, localizing the failure.

The provenance audit found that the real blank-sky files contained more than
1.35 million CCD7 events per observation, while a later common-grid
`reproject_events` product contained zero. The earlier response pipeline had
therefore built zero-background PHAs from a coordinate-processing loss, not a
physically empty detector background.

V19DQ restored the real pre-reprojection background for the two registered
cells without changing source events, region masks, response settings,
particle scales, model, statistic, free parameters or thresholds. Abell then
passed at reduced statistic 1.031211 and Bullet remained unchanged.

V19DR extended the same frozen repair to the entire unified archive:

| Audit | Terminal result |
|---|---:|
| Selected CCD7 products | 256 |
| Passed products | 256 |
| Failed/retried products | 0 / 0 |
| Source-band event range | 22--532 |
| Recovered background-band range | 20--1,219 |
| Total recovered 0.5--7 keV background events | 45,252 |
| Unified product index | 5,082 unique cells |
| Exact CCD7 replacements | 256 |

Every event-to-PHA channel audit, particle scale, response link and finite
response gate passed. This closes the CCD7 calibration problem and authorizes
the 494-region unmerged likelihood. It does not make any claim about I4, I5 or
gravity.

## Why this is a good stopping point

The current chapter had a concrete question: can the observed baryonic gas
state be measured reliably enough to test a directional or thermodynamic Sigma
source? The response/background archive is now complete and its registered
joint-spectrum boundary passes.

Starting the 494-region likelihood would begin a new, substantially larger
scientific chapter. Stopping here preserves a clean decision point:

- the best empirical formula and its limitations are recorded;
- failed mechanisms are catalogued;
- raw-lensing topology is identified as the decisive observational problem;
- the next source candidates are preregistered conceptually;
- the source archive is repaired and audited;
- no I4/I5 or new gravity result has been previewed.

## Recommended next steps

### Stage 1: complete the source-only thermodynamic test

1. Run the 494-region unmerged joint spectral likelihood using the V19DR
   5,082-cell product index.
2. Propagate response, background, temperature, abundance, gas-density,
   projection and smoothing uncertainty into registered draws.
3. Construct I4 thermodynamic-gradient stress and I5 baroclinicity using only
   baryonic observations. Do not use lensing maps, halo residuals or galaxy
   velocities in their definition or selection.
4. Score both candidates across the Bullet Cluster and Abell 2146 under every
   frozen smoothing, aperture and gas-correlation branch.

Advance only if I4 supplies a stable transferable direction and either I4 or I5
supplies independently measurable amplitude/activation without an object
label. If both fail, issue a source-mechanism falsification rather than tuning a
new threshold.

### Stage 2: choose one root field equation

If a source survives, derive one continuum-safe tensor/nonlocal field equation
with:

- a RAR-like deep-field limit derived rather than inserted;
- nonlinear response to overlapping baryonic components;
- conserved flux/energy and stated boundary conditions;
- a high-acceleration GR limit;
- one physical metric for matter and photons;
- at most five universal constants and no system-specific force parameter.

The equation must be invariant to arbitrary splitting or merging of the same
continuous baryonic density.

### Stage 3: freeze and test cross-domain transfer

Before opening new outcomes, freeze the equation, constants, source maps,
solver, target split, masks, covariance, nuisance priors and success thresholds.
Then test in this order:

1. development clusters for raw roots, multiplicity, parity, critical curves
   and positions;
2. morphology-diverse galaxy validation with full curves and resolved fields;
3. untouched multi-cluster strong and weak lensing;
4. joint dynamics-plus-lensing systems;
5. only after low-redshift success, Solar/relativistic completion and
   cosmological consequences.

Do not abandon a broadly competitive candidate after one failed stratum. Keep
the failure visible, complete the registered matrix, diagnose it, and apply the
three-materially-distinct-closures stopping rule before changing mechanism.

## Decision tree

| Next result | Decision |
|---|---|
| 494-region likelihood fails because of response/background inconsistency | repair measurement model only; do not judge I4/I5 or gravity |
| Likelihood passes, but I4 has no stable cross-cluster direction and I5 no transferable activation | retire thermodynamic-source route and return to a different independently measured baryonic source |
| I4/I5 survives source-only gates but cannot improve raw lens topology under one equation | retain source result; reject or redesign the propagator/action |
| New equation passes galaxies but repeatedly fails raw cluster topology | stop scalar/radial closure changes; require a different tensor/nonlocal mechanism |
| New equation passes clusters but damages morphology-diverse galaxies | reject object switching; revisit the universal limiting law |
| One frozen equation passes galaxies and raw multi-cluster lensing | advance to full mathematical, Solar, relativistic and cosmological validation |

## Files to read first when resuming

1. `docs/PROJECT_WRAP_UP_AND_NEXT_STEPS_2026-08-06.md` -- this handoff.
2. `docs/MASTER_FORMULA_VALIDATION_MATRIX.md` -- complete acceptance contract.
3. `docs/FORMULA_SCORECARD.md` -- numerical formula history.
4. `docs/FORMULA_AND_PRIOR_ART_REGISTRY.md` -- published versus project ideas.
5. `docs/SIGMA_V19DP_V19DQ_UNMERGED_JOINT_AND_CCD7_RECOVERY.md` -- registered
   spectral diagnosis.
6. `results/sigma_v19dr_full_ccd7_background_archive_recovery/report.json` --
   terminal archive audit.
7. `results/sigma_v19dr_full_ccd7_background_archive_recovery/unified_product_index.csv`
   -- required parent for the next 494-region likelihood.

## Final assessment

The project has produced a serious falsification framework and a useful
empirical bridge, not a completed theory. The bridge says that low acceleration,
diffuse baryonic state and dynamical organization matter. Raw lensing says the
missing physics must also carry spatial direction and multi-center curvature.
The cleanest next experiment is to determine whether measured thermodynamic
structure supplies that missing directional source. If it does not, the project
should change source mechanism rather than add another interpolation term.
