# Initial preregistration

Status: project-start specification, 2026-07-25. This file should be versioned
before long production runs or the addition of environmental measurements.

## Research question

Can a single smoothly unscreened acceleration law predict unseen outer SPARC
rotation-curve points across heterogeneous galaxies? If so, does its amplitude
increase with an independently measured surrounding underdensity?

These are two separate questions. Passing the first is evidence only for a
useful low-acceleration phenomenology. The specific void hypothesis requires the
second test as well.

## Primary model and estimands

For each radial point, the baryonic acceleration is constructed from the SPARC
gas, disk, and bulge components. The additional effective acceleration is

$$
g_{\rm add}=A_0 e^{\beta\mathcal V}a_t
\left(g_{\rm bar}/a_t\right)^p S(g_{\rm bar}),
$$

where $a_t$ is the activation acceleration, $w$ is the transition width in
decimal log acceleration, $p$ controls the outer radial behavior, $A_0$ is a
universal amplitude, and $\beta$ is the environmental coupling. The primary
estimands are $p$, outer-holdout predictive error, and—only after a frozen
external environment catalog exists—$\beta$.

For the initial implementation, constrained transforms bound the search to
$10^{-13}\le a_t\le10^{-8}$ m/s², $10^{-4}\le A_0\le10^2$,
$0.03\le w\le1.5$ dex, $0.05\le p\le1.5$, and, when enabled,
$-5\le\beta\le5$. A fit accumulating near any boundary triggers a wider-bound
sensitivity run; it is not interpreted as a measurement.

## Locked validation design

1. Apply galaxy-level cuts without inspecting model residuals: SPARC quality at
   most 2, inclination at least 30 degrees, and at least 8 valid radial points.
2. Within each retained galaxy, sort by radius. Fit the inner 70% and hold out
   the outer 30%, retaining at least five training and two holdout points.
3. Fit global parameters jointly. Treat disk and bulge mass-to-light ratios,
   distance, and inclination as nuisance parameters with the published
   uncertainties or declared priors.
4. Report train and outer-holdout chi-squared per point, RMSE, MAE, parameter
   count, and approximate AIC/BIC. Because nuisance parameters are
   MAP-regularized, information criteria are secondary diagnostics; outer-holdout
   performance is the primary Phase 1 score.
5. Use identical cuts, splits, error floors, and nuisance treatment for the
   Newtonian, RAR, NFW, and screened-void models.
6. After the analysis code is frozen, add galaxy-level cross-validation and
   repeated seeds/bootstraps. No result from a held-out galaxy may tune a global
   parameter.

## Comparators

- Newtonian baryons only.
- Empirical RAR with fixed characteristic acceleration $1.2\times10^{-10}$
  m/s².
- Baryons plus an NFW halo with two parameters per galaxy.
- Smooth screened model with free $p$ and no environment term.
- Smooth screened model with $p=1/2$ and no environment term.
- Smooth screened model with a preregistered external void score.

## Evidence thresholds

The phenomenology advances only if a common parameter set improves untouched
outer predictions over Newtonian baryons, remains competitive with RAR and NFW
after parameter penalties, and is stable across galaxy folds, quality cuts,
initialization seeds, and nuisance-prior variations. The free fit must put $p$
near $1/2$ without that value being imposed for a flat-curve claim.

The void interpretation advances only if an environment table built without
rotation-curve information gives a reproducible positive $\beta$, improves
held-out-galaxy predictions beyond the environment-free model, and survives
checks for distance method, inclination, surface brightness, angular resolution,
and survey-selection confounding.

## Outcomes that count against the hypothesis

- The free-$p$ fit is unstable or prefers a materially different exponent.
- Improvements disappear on outer radial or whole-galaxy holdouts.
- RAR or NFW predicts held-out data better once model complexity is counted.
- The environmental coefficient is null, negative, catalog-dependent, or
  explained by observational covariates.
- Per-galaxy amplitudes are required with no independently predictive structure.
- The inferred effect requires ordinary smooth-void tides many orders of
  magnitude above their standard gravitational scale.

## Causal guardrails

An environment score may use sky position, distance, reconstructed density,
void-center distance, void radius, density contrast, and a prespecified spatial
kernel. It may not use observed rotation speed, baryonic residual, fitted model
amplitude, or any transformation of those quantities. Environmental smoothing
scales must be declared before looking at their association with residuals, or
corrected as a multiple-testing family.

## Project phases

1. **Infrastructure and recovery tests:** ingestion, units, signed gas, synthetic
   parameter recovery, deterministic radial splits, and CPU/CUDA agreement.
2. **SPARC phenomenology:** free-$p$, fixed-$p$, comparators, radial holdout,
   galaxy folds, bootstraps, and sensitivity analyses.
3. **Environment construction:** cross-match SPARC coordinates to one frozen
   3-D density reconstruction and one independent void catalog.
4. **Void-specific inference:** estimate $\beta$, repeat galaxy-level holdouts,
   and run confounder/placebo tests.
5. **Three-dimensional predictions:** only after Phases 1–4, test disk thickness,
   orientation, and lopsidedness predictions with separate observations.

## Frozen Phase 3 environment construction

Frozen on 2026-07-26 before inspecting any environment-enabled fit or SPARC
residual association:

- Primary reconstruction: grouped Cosmicflows-4 64^3 density contrast grid.
- Primary score: `void_score = -delta_grouped_64`; a larger score means a lower
  reconstructed density. There is no learned transform or density threshold.
- Coordinates: SPARC ICRS sky position and published distance transformed to
  observer-centered Supergalactic Cartesian coordinates with Astropy.
- Distance convention: physical Mpc multiplied by `h100 = 0.746`, matching the
  CF4 catalog value `H0 = 74.6 km/s/Mpc`.
- Grid geometry: centered 500 h100^-1 Mpc box, 64 cells per axis, 7.8125
  h100^-1 Mpc voxels; trilinear interpolation at voxel centers.
- FITS convention: the official `(SGZ, SGY, SGX)` FITS-axis description becomes
  NumPy array indexing `(SGX, SGY, SGZ)` because FITS axes are reversed on read.
- Sensitivity reconstructions: the ungrouped 64^3 grid and the official 2026
  ungrouped 128^3, 1000 h100^-1 Mpc release.
- The published files named `delta_error` are 2-D and lack enough metadata to
  interpret as voxel-wise uncertainties, so they are preserved but not used as
  3-D errors.

The construction is deterministic and uses no SPARC velocity, model residual,
or fitted parameter. The exact inputs and checks are recorded in
`data/derived/cf4_environment_report.json`.
