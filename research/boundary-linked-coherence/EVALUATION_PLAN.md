# Evaluation Plan

## Objective

Decide whether boundary-linked coherence is:

1. mathematically inconsistent;
2. a descriptive reinterpretation of the canonical radial window;
3. a viable physical interpretation with no new predictive content; or
4. a predictive extension supported by held-out observations.

The work is ordered so that inexpensive no-go tests happen before large data fits.

## Phase 0 — Freeze the comparison

### Tasks

1. Record the current commit, Python environment, canonical parameters, and output hashes.
2. Run `python scripts/run_regression_extended.py --core` and save its report as the locked `H0`
   baseline.
3. Create deterministic galaxy-level train/validation/test splits. Radial points from one galaxy
   may not cross splits.
4. Freeze primary metrics and gates in `configs/preregistration.yaml` before looking at BLC test
   residuals.
5. Label all existing cluster data used to calibrate `n=0.27` as calibration, not independent
   validation.

### Deliverables

- `configs/preregistration.yaml`
- `configs/splits.json`
- `outputs/baseline/manifest.json`
- `outputs/baseline/core_report.json`

### Gate G0

The baseline must reproduce the repository metrics within documented numerical tolerance. If it
does not, stop and resolve environment or data drift.

## Phase 1 — Theory and consistency checks

### T1. Dimensions and limiting cases

Use analytic disks, rings, spheres, compact binaries, and homogeneous point clouds to verify every
limit in `MODEL_SPEC.md`. Fail on hidden dimensional constants, divergent source counts, or target
mass dependence.

### T2. Reciprocity and momentum

For isolated point sets and continuous meshes, compute total internal force and torque. Test both
the symmetric static kernel and deliberately asymmetric negative controls.

**Gate:** normalized residual force and torque below `10⁻¹⁰` at converged resolution, with the
failure control detected.

### T3. Field-energy accounting

Integrate the effective source and field energy over successively larger volumes. Identify the term
that pays for any additional binding. A literal luminosity model must use actual radiative energy,
not stellar rest mass under a luminosity label.

### T4. Causal extension sketch

Specify light-cone support, conserved quantities, and degrees of freedom. This is a paper-and-unit-
test deliverable, not a full relativistic solver.

**Gate G1:** all static checks pass and there is a credible route to a conserved causal extension.
Failure ends the “physical mechanism” track, though a phenomenological kernel may still be studied
under that label.

## Phase 2 — Synthetic discriminators

Build source distributions for which the canonical radial window and BLC make different
predictions:

1. Same radial density, different azimuthal gaps.
2. Same galaxy, with and without an exterior companion.
3. Same mass profile, rotated relative to a filament-like external distribution.
4. Smooth disk versus a disk with a disruption annulus.
5. Equal baryonic mass and geometry, different emissivity profiles.
6. Source-count refinement at fixed continuous density.

### Primary synthetic metrics

- outer log-slope of the additional acceleration;
- recovery of injected global kernel parameters;
- false-positive rate under `H0`;
- force/torque closure;
- stability to mesh and source-count refinement;
- ability to distinguish `HM` from `HL` in blinded injections.

### Gate G2

- Recover a `g_BLC ∝ r⁻¹` injection with slope between `-1.1` and `-0.9` over the preregistered
  outer interval.
- Include zero in the nominal confidence interval for at least 95% of null simulations, with a
  false-positive rate no greater than 5%.
- Infer `HM` versus `HL` correctly in at least 90% of powered synthetic catalogs.
- No fitted quantity drifts materially with particle count or grid resolution.

## Phase 3 — SPARC mechanism and luminosity ablation

The repository already contains SPARC mass models for 175 disk galaxies and uses 171 in the
canonical analysis. Reuse the existing signed-gas and fixed mass-to-light preprocessing.

### Luminosity data augmentation

SPARC's 3.6 μm photometry is primarily used as a stellar-mass tracer, so it cannot by itself cleanly
separate mass-normalized from current-radiation hypotheses. Before testing `HL`, preregister a
crossmatch for independent luminosity-state tracers such as UV, Hα, mid/far-infrared, color, and
star-formation rate. Construct both dust-corrected bolometric luminosity and recent-star-formation
proxies, propagate nondetections, and retain a no-crossmatch control sample. No luminosity proxy is
chosen by its correlation with rotation-curve residuals.

### Models compared

1. Canonical Σ-Gravity (`H0`).
2. Existing exponential radial kernel from SI §13b.
3. Legacy one-dimensional survival kernel.
4. Mass-normalized BLC (`HM`).
5. Luminosity phase-carrier BLC (`HL-phase`).
6. Literal radiative-energy BLC (`HL-energy`).
7. MOND and a conventional halo benchmark using the repository's existing comparison methods.

### Leakage controls

- Compute BLC features only from baryonic maps, independent environment catalogs, predicted
  kinematics, or velocity-dispersion measurements.
- Do not use `V_obs`, observed residuals, or a fitted halo property in `B`, `C`, `K_R`, or split
  construction.
- Shuffle BLC features within bins of baryonic mass, scale length, surface brightness, gas
  fraction, and distance.

### Primary metrics

- held-out Gaussian log likelihood with velocity uncertainties;
- galaxy-balanced weighted RMS in km/s;
- RAR scatter in dex;
- AICc/BIC or an equivalent complexity-aware score;
- calibration slope and coverage of predictive intervals;
- worst-decile galaxy error, not only the mean.

### Interpretation gate G3a

`HM` may replace the radial window as a physical interpretation if it uses no more than two new
global parameters, passes G1–G2, and its held-out galaxy-balanced RMS is no more than 5% worse than
canonical Σ-Gravity.

### Predictive gate G3b

BLC advances as a predictive extension only if a non-radial BLC feature improves the preregistered
held-out likelihood after the complexity penalty, the improvement survives all stratified shuffles,
and the effect direction repeats in the untouched test split.

### Luminosity decision

- Prefer `HL` only if luminosity or stellar-population information improves held-out prediction at
  fixed baryonic structure and the result survives dust, distance, gas-fraction, and mass-to-light
  uncertainty tests.
- Reject literal luminosity scaling if its inferred amplitude requires more gravitational source
  energy than the radiative-energy budget or if old/dim and young/bright systems contradict its
  preregistered residual pattern.
- Prefer `HM` if boundary information matters but present luminosity does not.

## Phase 4 — Environment and orientation test

This is the distinctive BLC phase. Crossmatch the SPARC test galaxies, or a larger compatible
rotation-curve sample, to an external baryonic density reconstruction without using rotation-curve
residuals during matching.

### Precomputed predictors

- scalar external baryonic field;
- distance and direction to the dominant external baryonic concentration;
- tidal tensor eigenvalues and eigenvectors;
- disk-axis alignment with the external field or filament;
- forward/backward openness asymmetry;
- local neighbor density and an isolation flag.

### Tests

1. Does BLC residual improvement correlate with preregistered openness after controlling for
   baryonic mass, `g_N`, surface brightness, scale length, inclination, gas fraction, and distance?
2. Are receding/approaching or azimuthal residuals aligned with the external-field direction?
3. Do matched isolated and environmentally connected galaxies differ in the predicted direction?
4. Does the effect survive sky-position, survey-depth, and distance negative controls?

The analysis should reuse the independent external-field estimates and caveats developed in
published SPARC environment studies rather than infer “environment” from the target residuals.

### Gate G4

The sign and angular dependence must match the preregistration, survive multiple-testing correction,
appear in the untouched test sample, and remain when the dominant external object is removed one at
a time. A scalar environment correlation without the predicted orientation is insufficient evidence
for a boundary link.

## Phase 5 — Cross-scale and gravity constraints

Only models passing G3a may enter this phase.

### Galaxies and counter-rotation

Re-run the repository's MaNGA counter-rotation test without retuning. Determine whether pairwise
`C_xy` predicts the reported direction and magnitude more directly than the canonical scalar.

### Clusters and lensing

- Keep the Fox et al. sample labeled as calibration for the current amplitude exponent.
- Use an untouched cluster sample or spatially resolved weak/strong-lensing maps for validation.
- Predict both dynamics and lensing from the same potential unless a separately constrained slip
  is explicitly introduced.
- Test centroid and angular structure, not only one aperture mass ratio.

### Compact and external-field systems

- Cassini and planetary-ephemeris limits.
- Wide binaries in the Milky Way external field.
- Dwarf satellites and ultra-diffuse galaxies across environments.
- Binary-pulsar and gravitational-wave propagation constraints for any new dynamical mode.

### Gate G5

No cross-scale validation may be rescued with a system-specific amplitude, screening threshold, or
kernel family. A failure is reported against the exact preregistered variant.

## Phase 6 — Relativistic theory decision

Proceed only if G1–G5 pass.

Required deliverables:

1. a covariant action or explicitly conserved effective field equations;
2. a well-posed initial-value formulation;
3. a causal kernel or emergent-retardation derivation;
4. weak-field and post-Newtonian limits;
5. lensing and gravitational-wave propagation from the same theory;
6. cosmological background and linear-perturbation equations.

At this point the mechanism can be compared with nonlocal gravity as a theory rather than as a
phenomenological analogy.

## Decision table

| Outcome | Classification | Next action |
|---|---|---|
| Fails G1 | Inconsistent mechanism | Stop; retain only as rejected hypothesis |
| Passes G1–G2, fails G3a | Mathematically viable but empirically poor | Archive with null results |
| Passes G3a, fails G3b/G4 | Interpretation of canonical `W(r)` | Document as non-unique physical picture |
| Passes G3b or G4, fails G5 | Galaxy-scale phenomenology | Publish bounded result; do not claim universal gravity theory |
| Passes G1–G5 | Candidate predictive mechanism | Begin Phase 6 relativistic construction |

## Initial execution order

1. `BLC-000` through `BLC-005`: baseline and theory gates.
2. `BLC-010` through `BLC-014`: synthetic suite.
3. `BLC-020` through `BLC-024`: SPARC ablation and luminosity test.
4. `BLC-030` through `BLC-033`: independent environment/orientation test.
5. `BLC-040` onward: cross-scale validation only after prior gates pass.

The IDs and their exact pass conditions are in `EXPERIMENT_MATRIX.csv`.
