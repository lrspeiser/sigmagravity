# Theory-development progress against the stage gates

## 2026-08-06 Sigma V19CZ official-state regional NXB result

The independently specified correction to the V19CY background parameter
state has now been executed on all seven A2319 regions without loading a source
spectrum.  The official recipe retains the delivered photon-index and Au-line
width freedoms during the second fit in addition to the twelve recommended
normalizations.  That distinction does not rescue the public regional model:
six reduced chi-square values remain 10.97--17.96, and free parameters reach
hard bounds in every region.  Only region A is near acceptable at 1.138.

No new A2319 source fit is authorized.  The public Resolve-v1 small-subarray
NXB route is retired pending an independently released region-aware model;
A3667 validation and A754 holdout remain sealed.  This is an observational
likelihood failure, not a gravity-formula result.  See
[`SIGMA_V19CZ_A2319_OFFICIAL_NXB_PREFIT_RESULTS.md`](SIGMA_V19CZ_A2319_OFFICIAL_NXB_PREFIT_RESULTS.md).

## 2026-08-06 Sigma V19CY A2319 response-aware spectral result

All seven NXB-only prefits and 21 source-model fits completed with the frozen
responses and mixed likelihood. All seven primary velocities converged, all
met the 200-km/s precision threshold, and six regions passed both robustness
branches. The terminal gate nevertheless failed because recommended NXB
normalizations hit the public empirical model's hard bounds in every fit. Six
regional NXB prefits had reduced chi-square 10.7--17.5; only region A was
acceptable at 1.12. The velocity pattern is therefore not promoted to a signed
Sigma source. A3667 validation and A754 holdout remain sealed. See
[`SIGMA_V19CY_A2319_RESPONSE_AWARE_SPECTRAL_RESULTS.md`](SIGMA_V19CY_A2319_RESPONSE_AWARE_SPECTRAL_RESULTS.md).

## 2026-08-06 Sigma V19CY Chandra response-image gate

The frozen A2319 extended-source image needed for response-aware XRISM ARFs is
now complete. CIAO combined Chandra ObsIDs 3231 and 15187 over 0.5--7.0 keV and
returned exit code zero. The 1463 by 1463, 11.9966-arcmin crop contains 486,683
positive pixels and passed its celestial-center, finite-pixel, positivity,
width, and SHA-256 gates. Its SHA-256 is
`4db29ad4f17fb05cbf264865709ed604a2e3c9976fac9bd749923210c2c99bc9`.

Two infrastructure attempts are recorded rather than hidden: an unset CIAO
`PFILES` startup and a too-short outer host timeout. Neither generated or read
an XRISM spectrum, response, background, velocity, validation result, or
holdout result. The clean run moved temporary projection work to WSL-native
storage without changing any scientific input or frozen setting. Response and
background generation is the next active development step; A3667 validation
and A754 holdout remain sealed. See
[`SIGMA_V19CY_DIRECT_ICM_VELOCITY_EVIDENCE_PLAN.md`](SIGMA_V19CY_DIRECT_ICM_VELOCITY_EVIDENCE_PLAN.md).

## 2026-08-05 Sigma V19CY direct-velocity evidence plan

The post-V19CX observational pivot is now frozen before validation or holdout
velocity outcomes are opened. Public XRISM/Resolve data provide a direct
time-odd observable that the thermodynamic source chain lacked: signed
line-of-sight gas current. Abell 2319 is development-only, Abell 3667 is sealed
validation, and Abell 754 is the untouched holdout.

The protocol requires three spectral-model branches, sign-topology stability,
a spatially nonconstant velocity field, density-independent signed-current
variance and leave-one-region-out robustness in every system. It separately
tests whether I4 aligns with the observed velocity-gradient axis and whether I5
tracks kinetic stress. Lensing, halo maps, action selection and gravity fitting
remain forbidden. See
[`SIGMA_V19CY_DIRECT_ICM_VELOCITY_EVIDENCE_PLAN.md`](SIGMA_V19CY_DIRECT_ICM_VELOCITY_EVIDENCE_PLAN.md).

## 2026-08-05 Sigma V19CX terminal source-reconstruction result

The observation-hierarchical Bullet recovery succeeded mechanically: all 3,812
Bullet integrated cells were combined exactly once, full-PHA source counts were
conserved at 707,569, all response links passed, and the ten-group hierarchy
avoided the direct-stack CIAO failure. Both selected-region fits also passed.

The unchanged scientific commissioning nevertheless failed. The Bullet
integrated one-temperature absorbed-plasma fit had reduced statistic 2.7937,
above the frozen 1.5 limit; Abell 2146 integrated passed at 1.2232. V19CX is
therefore a terminal scientific gate failure, not an execution failure. Full
494-region production, V19X4B/V19BMB, V19BQ/V19BS and action derivation are not
authorized, and no fit rule may be changed to rescue this closure. See
[`SIGMA_V19CX_BULLET_HIERARCHICAL_RECOVERY_RESULTS.md`](SIGMA_V19CX_BULLET_HIERARCHICAL_RECOVERY_RESULTS.md).

## 2026-08-05 Sigma V19BS source disposition

The post-source decision is now frozen before terminal I4/I5 evidence exists.
A valid source failure excludes every action-placement class for this route and
requires new direct temporal gas evidence or a preregistered independent merger
sample; it cannot be repaired with lensing, halo maps or threshold changes. A
source pass authorizes mathematical comparison of the time-even P1 constrained
composite and P3 degenerate pure-metric placements, but does not select either.

The P2 causal-memory route is excluded unless independent time-odd or clocked
evidence is later obtained. P1 versus P3 must be chosen only by constraints,
degrees of freedom, conservation, boundedness, hyperbolicity, stability and the
one-metric weak-field limit. The preflight and pass/fail logic tests pass with
all gravity targets sealed. See
[`SIGMA_V19BS_SOURCE_DISPOSITION.md`](SIGMA_V19BS_SOURCE_DISPOSITION.md).

## 2026-08-05 Sigma V19BR target-sealed terminal chain

The final execution sequence from the live response archive to the I4/I5 source
decision is now frozen as one fail-closed eleven-stage driver. It requires the
protected V19W process to exit before V19W5 can start, hashes every stage
executable, accepts only the separately frozen successor configurations and
stops on any failed or corrupt terminal artifact. This removes the risk of
manually invoking a historical V19W4/V19X3/V19X4/V19BP path.

A valid negative V19BQ scientific result is terminal evidence, not an execution
error: it records source-mechanism falsification and forbids action derivation.
Only a passing source result can authorize the later V19BJ action route. The
driver contains no lensing, halo, galaxy-rotation, action, gravity-fit or
holdout stage. Its preflight and fail-closed tests pass while all terminal
stages remain pending behind the active base process. See
[`SIGMA_V19BR_TARGET_SEALED_TERMINAL_CHAIN.md`](SIGMA_V19BR_TARGET_SEALED_TERMINAL_CHAIN.md).

## 2026-08-05 Sigma V19BQ observed-source successor preflight

The final target-sealed handoff from the V19W5-authorized gas chain to the
source-physics decision is now prepared. V19BQ requires terminal V19X4B and
V19BMB products and copies the complete V19BP registered inputs, six spatial
variants, thresholds, I4-before-I5 decision rule and streaming execution
contract without modification. It cannot be frozen until both terminal parents
pass every gate and product hash.

The preflight passes. The eventual run evaluates two merging clusters, three
temperature--density dependence branches and six spatial variants for each of
I4 and I5: 36 conditions per candidate and 72 candidate evaluations. I4 must
carry a stable direction everywhere; scalar I5 can rescue amplitude only. No
terminal source result, lensing, halo, galaxy rotation, action, gravity
parameter or holdout was opened. See
[`SIGMA_V19BQ_V19X4B_OBSERVED_SOURCE_SUCCESSOR_PREFLIGHT.md`](SIGMA_V19BQ_V19X4B_OBSERVED_SOURCE_SUCCESSOR_PREFLIGHT.md).

## 2026-08-05 Sigma V19BMB V19X4B stellar-successor preflight

The stellar nuisance control is now safely routed to the future V19X4B common
grids without editing the already hash-bound V19BM/V19BP chain. V19BMB retains
all 4,096 member-posterior draws, exact physical grid, cloud-in-cell deposition,
unit-light conservation, 50/100-kpc smoothing and within-draw percentile ranks.
Cross-filter amplitudes and stellar-mass inference remain forbidden.

The preflight and manufactured fail-closed tests pass. V19BMB cannot be frozen
or executed until V19X4B supplies all 12 passing, hash-bound products. No
terminal gas or stellar value, source score, lensing, halo, gravity parameter or
holdout was opened. See
[`SIGMA_V19BMB_V19X4B_STELLAR_SUCCESSOR_PREFLIGHT.md`](SIGMA_V19BMB_V19X4B_STELLAR_SUCCESSOR_PREFLIGHT.md).

## 2026-08-05 Sigma V19X4B V19X3B gas-successor preflight

The full gas-posterior stage is now prepared behind V19X3B without rewriting
the original V19X4/V19BP chain. A mechanical freezer copies and canonically
hashes all eight V19X4 science sections: the corrected APEC algebra,
composition, geometry, 4,096-draw uncertainty model, three dependence branches,
depth/profile fallbacks, common-grid reconstruction, smoothing and runtime
gates. Only the terminal regional-data authority changes from V19X3 to V19X3B.

The preflight and manufactured tests pass. Terminal V19X3B must still contain
all 494 regions with every production gate and the target seal intact before a
V19X4B configuration can exist. No observed regional temperature, gas
posterior, source score, lensing, halo, gravity or holdout result was opened.
See
[`SIGMA_V19X4B_V19X3B_GAS_SUCCESSOR_PREFLIGHT.md`](SIGMA_V19X4B_V19X3B_GAS_SUCCESSOR_PREFLIGHT.md).

## 2026-08-05 Sigma V19X3B V19W5 regional-successor preflight

The 494-region production handoff is now prepared without mutating the
hash-bound V19X3/V19X4/V19BP preregistration chain. V19X3B accepts only a
passing V19X2 report whose configuration carries the explicit V19W5 terminal
status and `v19w5_recovery` archive. It then reuses the existing V19X3 regional
engine byte-for-byte, preserving every combination, checkpoint, plasma-fit,
uncertainty, retention and quality rule.

The preflight passes and synthetic tests prove both directions: a V19W5
authority is propagated into response validation, while a V19W4 authority is
rejected. The unchanged original V19X3 runner/freezer still match the hashes in
V19X4. V19X3B remains runtime-closed until V19W5 and V19X2 pass. No terminal
temperature, gas source, lensing, halo, gravity or holdout result was opened.
See
[`SIGMA_V19X3B_V19W5_REGIONAL_SUCCESSOR_PREFLIGHT.md`](SIGMA_V19X3B_V19W5_REGIONAL_SUCCESSOR_PREFLIGHT.md).

## 2026-08-05 Sigma V19BP observed source-invariant executor preflight

The terminal source-only integration is now implemented and frozen before any
observed source score or target field is available. Passing V19X4 gas
posteriors and V19BM stellar ranks will be streamed through the commissioned
V19BO map mathematics and V19BN decision engine. The executor preserves full
draw-level activation, I4 direction and PRESS novelty evidence plus regional
posterior summaries and independently verifies every terminal product hash.

Each I4 and I5 candidate must survive 2 clusters, 3 gas-correlation branches
and 6 smoothing/aperture variants: 36 conditions per candidate and 72 candidate
evaluations in total. The 50-kpc/350-kpc setting is primary, but all five
alternatives are mandatory. Branches cannot be averaged. I4 direction must
pass everywhere; only then may either I4 amplitude or scalar I5 satisfy the
strength requirement. Thus a scalar signal cannot manufacture the direction a
single-metric lensing theory will need.

The preflight passes, with lensing, halos, galaxy rotation, action selection,
gravity parameters and holdouts sealed. Observed execution remains blocked on
terminal V19X4 and V19BM. Solar/PPN remains a later hard veto rather than an
optimization input. See
[`SIGMA_V19BP_OBSERVED_SOURCE_INVARIANT_EXECUTOR_PREFLIGHT.md`](SIGMA_V19BP_OBSERVED_SOURCE_INVARIANT_EXECUTOR_PREFLIGHT.md).

## 2026-08-05 Sigma V19W2C/V19W5 CCD7 response boundary

The live response build exposed a detector boundary not present in the earlier
commissioning set: 254 Abell 2146 CCD7 cells exhausted both base attempts
because CCD7 is absent from the matching blank-sky geometry. A new read-only
snapshot independently validated 3,706 completed cells and 14,824 products
before freezing six outcome-blind CCD7 cases: the minimum, lower-median and
maximum source-count ranks in each of the two affected observations.

All six cases passed. Their exact materialized background subsets contained
zero events at all energies, allowing the already frozen V19W2 method to build
the source response, create and link a zero-count background PHA, and pass every
mask, histogram, detector-medoid, ARF/RMF, link, scaling, size and hash gate.
V19W5 now supersedes the unexecuted V19W4 terminal launcher and makes this CCD7
pass mandatory. It remains blocked until the unchanged base process exits and
its full-interval report passes. The response adapter has an explicit,
fail-closed V19W5 status and archive-label mode while retaining V19W4 only as
its low-level historical default. The V19X2 production freezer/runner now
requires V19W5 and cannot silently fall back to V19W4. The hash-bound V19X3/X4
preflight chain is preserved pending a separately named successor. No
spectrum was combined, no gas state was fit and no lensing or gravity result
was opened. See
[`SIGMA_V19W2C_CCD7_RESPONSE_COMMISSIONING_RESULTS.md`](SIGMA_V19W2C_CCD7_RESPONSE_COMMISSIONING_RESULTS.md)
and
[`SIGMA_V19W5_CCD7_HARDENED_RESPONSE_RECOVERY_PLAN.md`](SIGMA_V19W5_CCD7_HARDENED_RESPONSE_RECOVERY_PLAN.md).

## 2026-08-05 Sigma V19X3 full regional spectral preflight

The executable gap after V19X2 commissioning is now closed in advance.  A
checkpointed successor groups all 5,082 unified response cells into the 366
Bullet and 128 Abell 2146 regions, combines each region once, fixes abundance
to the passing cluster-integrated V19X2 value and applies the identical plasma
fit and uncertainty rules.  Separate combination and fit checkpoints are
content-bound to cell identities, PHA hashes, count totals and abundance, so a
long run can resume without outcome-selective recombination or refitting.

The mechanical freezer still refuses to emit a V19X3 configuration until
V19X2 has passed.  Seven synthetic tests pass.  No production region was
combined, no temperature or gas state was learned and no lensing or gravity
payload was opened.  See
[`SIGMA_V19X3_FULL_REGIONAL_SPECTRAL_PRODUCTION_PREFLIGHT.md`](SIGMA_V19X3_FULL_REGIONAL_SPECTRAL_PRODUCTION_PREFLIGHT.md).

## 2026-08-05 Sigma V19BJ source-invariant/action preselection

The post-gas decision is now frozen before any V19X temperature, lensing or
halo result is available.  Total density is an ineligible control; five
covariant source-state candidates are registered: component overlap, relative
gas--collisionless current, anisotropic stress, thermodynamic-gradient stress
and baroclinicity.  They must use one definition in Bullet and Abell 2146 and
pass fixed detection, projection, resolution, leave-one-region-out and
density-nonredundancy gates.

At least one scalar activation and one measured vector or tensor direction
must pass in both clusters before an action may be derived.  If none passes,
the response is not rescued with lensing or a tuned combination; direct gas
velocity information or another independent merger is required.  Three
action-placement classes are registered for later mathematical derivation,
but none is selected.  This advances the physical-postulate/source-definition
work while preserving V19W/V19W5, the V19X successor and all 494 regional fit
attempts as mandatory prerequisites.  Every region needs a finite best fit and at least
12 regions per cluster must pass the individual quality gate; 494/494 quality
passes are not required.  A reusable projected-map library now passes six manufactured
tests covering component limits, common-boost invariance, rotation covariance,
trace removal, known-axis recovery, baroclinic limiting cases and invalid-input
failure.  This commissions the math but is not an astronomical result.  See
[`SIGMA_V19BJ_SOURCE_INVARIANT_ACTION_PRESELECTION.md`](SIGMA_V19BJ_SOURCE_INVARIANT_ACTION_PRESELECTION.md).

## 2026-08-05 Sigma V19BI blind-galaxy admission protocol

The future galaxy test is now protected from reusing the 131 SPARC systems,
13 resolved LITTLE THINGS systems or 34 SPIDERS-MaNGA BCGs as fresh evidence.
The primary independent candidate universe is the 109-system WALLABY PDR1
kinematic release; PHANGS contributes 67 high-resolution inner CO curves and
DiskMass contributes 30 published radial-plus-vertical systems, subject to
identity and data-availability audits.

No new velocity target or galaxy is opened or selected.  The final holdout
must contain at least 48 galaxies, six per frozen mass/gas/surface-brightness/
bulge stratum, with raw-field, high-resolution inner and radial-plus-vertical
subsets.  WALLABY's few-beam resolution, fixed 10-km/s dispersion and
inclination/beam systematics are explicit forward-cube controls rather than
hidden caveats.  Measurement nuisances use the same external priors for Sigma,
fixed MOND/RAR and halo comparators and cannot become per-galaxy gravity
constants.  See
[`SIGMA_V19BI_BLIND_GALAXY_ADMISSION.md`](SIGMA_V19BI_BLIND_GALAXY_ADMISSION.md).

## 2026-08-05 Sigma V19BH blind-cluster admission protocol

The future cluster holdout is now protected by an outcome-blind, state-
stratified admission protocol.  The public starting universe combines 37 SGAS
lenses, the 28-system Chandra Strong Lens Sample, and 41 RELICS systems as a
reserve.  Eight SGAS systems absent from tracked analysis-bearing files at the
pre-protocol commit form a metadata-only shortlist: four relaxed-side and four
disturbed systems.  None is admitted and no raw image coordinate, lens map,
topology or Sigma residual was opened.

The final six must include cool-core and non-cool-core relaxed lenses,
plane-of-sky and projection-challenging mergers, and both mass halves.  Every
system still needs three secure families, one spectroscopic family, eight
images, per-image uncertainties, complete stars/gas/BCG/ICL/member baryons and
a same-catalog halo comparator.  PLCK G004.5-19.5 is recorded as a concrete
availability-without-eligibility example because its published one-family,
three-image constraint fails the frozen raw-lensing minimum.

The same checkpoint orders the non-Solar predictions.  Weak lensing and
merger offsets come first because they use the quasistatic metric already
needed for strong lensing; satellite, stream and friction tests follow a
three-dimensional time-dependent action; growth and the CMB follow the
covariant background and perturbation equations.  Solar/PPN remains a later
hard veto rather than the present optimization target.  See
[`SIGMA_V19BH_BLIND_CLUSTER_ADMISSION.md`](SIGMA_V19BH_BLIND_CLUSTER_ADMISSION.md).

## 2026-08-05 Sigma v19X2 unified-response adapter preflight

The future spectral combination step no longer assumes that all 5,082 response
cells occupy the original V19W archive. A target-blind adapter preflight consumes
an explicit `cell_directory` in a unified index and accepts the explicit
`base_v19w` and `v19w5_recovery` production schema. It independently rechecks the terminal
authority, index hash and size, allowed archive roots, task identity, event
counts, cell-report hash, all four product names/sizes/hashes, and the source
PHA channel-count audit.

Synthetic tests prove that mixed base/recovery cells are usable and that an
out-of-root path, changed report, or mutated product fails closed. This is only
an implementation preflight. An unfrozen orchestration scaffold now preserves
the original two integrated fits followed by the two abundance-fixed selected-
region fits and blocks all 494 production regions if any gate fails. It also
preserves each validated cell's base/recovery provenance. A mechanical freezer
now refuses absent/failed terminal evidence and, after a pass, will copy the
registered workload, combination, fit sequence, and gates exactly from V19X
while hashing every parent. Its config hash is recorded in the execution report
rather than circularly embedded in the config itself. V19X2 is not frozen until
V19W terminates and V19W5 produces its terminal report and unified index. The
freezer and commissioning runner now propagate that exact V19W5 authority,
status, archive label and hashes before any combination is allowed. V19X3 must
be superseded under a new name after V19X2 passes because V19X4 hashes the
current preregistered V19X3 files. No
response was combined, no spectrum or temperature was fit, and no lensing or
gravity payload was opened. See
[`SIGMA_V19X2_UNIFIED_RESPONSE_ADAPTER_PREFLIGHT.md`](SIGMA_V19X2_UNIFIED_RESPONSE_ADAPTER_PREFLIGHT.md).

## 2026-08-05 Sigma v19BG broad-phenomenology contract

The long-wave source-state lane is now explicitly prevented from optimizing
around the Solar System, one galaxy class, or one cluster. Detailed local
parameter work is deferred while Solar/PPN/propagation/health constraints
remain a mandatory final exclusion gate. The near-term priority is one frozen
metric across eight galaxy strata and six cluster states.

The audit also closes an apparent shortcut. The point-source activation
`1+A[1-(1+r/L)exp(-r/L)]` is exactly the published STVG/MOG radial shape. Its
existing action-level control had already covered 131 SPARC galaxies (3,034
points), 20 CLASH systems (84 model-derived radial points), and 34 BCG systems
and failed before an observational fit. Changing only its amplitude, range, or
Fourier exponent is not a new long-wave theory.

The surviving class remains a nonlinear, baryon-forced,
source-state-sensitive one-metric response. The future blind strong-lensing
sample is raised to at least six systems, including at least two relaxed and
two disturbed/merging clusters, with complete baryons and raw positional
uncertainties. Seven additional forward-prediction obligations are registered:
weak lensing, colliding-cluster offsets, dwarf/satellite dynamics, streams and
compact substructure, dynamical friction, cosmic growth, and the primary/lensed
CMB. The last three cannot be scored honestly until a covariant action supplies
field energy, background evolution, and perturbation equations. See
[`SIGMA_V19BG_BROAD_PHENOMENOLOGY_CONTRACT.md`](SIGMA_V19BG_BROAD_PHENOMENOLOGY_CONTRACT.md).

## 2026-08-04 Sigma v19B-v19F causal-source readiness checkpoint

The causal-assembly branch has a new target-blind mechanism-development pair:
the Bullet Cluster and Abell 2146. Both have greater-than-six-sigma primary
merger shocks, published projection/clock intervals, and local member tables
that retain individual velocity uncertainties. The parsed catalogs contain 78
and 63 members, respectively.

Twenty matched-depth Chandra observations have now been downloaded, hash
verified, and processed with the previously audited CIAO 4.18.0/CALDB 4.12.4
environment. All 20 pass reprocessing, flare, compact-source, and blank-sky
gates. The cleaned exposures are 561.128 ks for Bullet and 418.013 ks for
Abell 2146; the minimum retained fraction is 0.855990 against a frozen 0.50
gate. No event image was visually inspected and no replacement-cluster lensing
target was opened.

This advances data readiness, not the theory. Registration, automated shock
geometry, resolved thermodynamic uncertainties, and the projection/clock
ensemble must be frozen before source construction. Only a source that then
transfers without cluster-specific amplitude, scale, clock, or orientation may
be compared with the sealed lensing fields. See
[`SIGMA_V19B_V19F_CAUSAL_SOURCE_READINESS.md`](SIGMA_V19B_V19F_CAUSAL_SOURCE_READINESS.md).

## 2026-08-04 Sigma v19A assembly-history readiness result

The first causal-history audit did not authorize a source. Deterministic
member-catalog controls found a stable projected line-of-sight gradient in
MACS J0416 (`848.2 km/s/Mpc`, permutation-equivalent `3.353 sigma`) but no
corresponding signal in PLCK G287 (`441.0 km/s/Mpc`, `-0.796 sigma`). The
Dressler--Shectman controls are `2.636 sigma` and `0.096 sigma`, respectively.
These are instantaneous phase-space diagnostics rather than time coordinates.

Published evidence does not provide one common primary-event clock. MACS J0416
has mutually incompatible pre-merger versus prior-passage/second-approach
interpretations, and its detected outer discontinuity belongs to a smaller
interaction. PLCK G287 has a measured `389 +/- 6 kpc` shock and approximate
`180 Myr` propagation upper bound, but merger and AGN origins remain unresolved.
The derived catalogs also lack member-redshift uncertainties, the matched
resolved-temperature uncertainty maps are absent, and transverse velocity and
line-of-sight depth ensembles are unavailable.

This closes source construction on the current spent pair as **data
insufficiency and causal non-identifiability**, not as a physics falsification.
No formula, gravity parameter, lensing map, new target, or holdout was used.
History is therefore not allowed into the root equation as a fitted merger-age
or response-length parameter. See
[`SIGMA_V19A_ASSEMBLY_HISTORY_READINESS.md`](SIGMA_V19A_ASSEMBLY_HISTORY_READINESS.md).

## 2026-08-04 Sigma v18B-v18D collisionless-stress result and v19 gate

The AS295 public redshift releases remain 20 secure members short of the
frozen 50-member collisionless-stress gate, but an already-spent replacement
pair passed without weakening it: MACS J0416 retained 231 members and PLCK
G287 retained 129. One target-blind adaptive member-stress map was frozen for
each cluster before opening either spent GLAFIC target.

The cross-transfer result is decisive. The member-stress model has symmetric
full-field NRMSE 1.75737 versus 0.99689 for baryons-only GR, a 76.29%
worsening. Its two directional coefficients differ by 1.15455 dex, or a
factor of about 14.3. Residual power and shear fail in both directions; the
maximum R50/R80 error is 29.56%; only the 1.485% resolution gate passes.

Like thermal stress, the member source carries some halo-extent information:
three of four residual-field radii are within 10%. But present-day projected
member stress does not supply universal amplitude, phase, or shear. It is
retired as the direct root source, and no holdout was opened. See
[`SIGMA_V18B_V18D_COLLISIONLESS_STRESS_RESULTS.md`](SIGMA_V18B_V18D_COLLISIONLESS_STRESS_RESULTS.md).

An audit of v3-v14 and P0646-P0677 shows that a generic new memory, diffusion,
tensor AQUAL, material carrier, preferred clock, or gauge-tidal field would
replay an already-tested family. V19 is therefore frozen as a measurement-first
causal-assembly observability gate. It must identify time information in
baryonic data before selecting an equation or seeing a lensing target. See
[`SIGMA_V18_MECHANISM_EXHAUSTION_AND_V19_GATE.md`](SIGMA_V18_MECHANISM_EXHAUSTION_AND_V19_GATE.md).

## 2026-08-04 Sigma v17C-v17E thermal-stress result

The complete target-blind Chandra chain has now been executed on the spent
AS295/PLCK G287 pair. It produced 298 of 299 planned detector-region spectra,
with one exact CIAO empty-response-domain cell quarantined under a frozen
runtime signature, then fit all 29 AS295 and 21 PLCK G287 temperature regions.
Both regional quality gates passed. The integrated temperatures are 9.6036 and
14.3962 keV.

The best frozen thermal source improves the static symmetric full-field NRMSE
from 0.817941 to 0.785960, only 3.910% versus the required 10%, and remains far
above the absolute 0.500 gate. It fails transferred residual power in one
direction and shear alignment in both. It nevertheless predicts all four
amplitude-independent R50/R80 residual-field radii within 5.35--9.53% and is
stable to 1.009% under doubled resolution.

Projected gas density and temperature therefore contain a real clue about the
*extent* of the apparent halo response, but they do not determine its strength,
centroid, or tensor direction. V17F is skipped, and pressure/temperature alone
cannot enter the root constitutive law. The next authorized source is matched
collisionless baryonic stress or a genuinely distinct causal state. See
[`SIGMA_V17C_V17E_THERMAL_STRESS_RESULTS.md`](SIGMA_V17C_V17E_THERMAL_STRESS_RESULTS.md).

## 2026-08-04 Sigma v18 post-pressure gravitational-flux selection

The post-v17Q root variables are now frozen without selecting a convenient
new formula. An outward displacement field obeys
`nabla dot D=4 pi G rho_b`; one physical weak-field potential obeys
`nabla W=delta H/delta D`; and the conventional effective halo is an output,
`rho_Sigma,eff=-(4 pi G)^-1 nabla dot(g+D)`. Newtonian gravity is
`H=D^2/2`, while a spatially local `H_0(abs(D))` is only the published
Legendre-dual form of AQUAL.

This formulation makes the halo-size question explicit. In an isolated sphere,
`abs(D)=GM/r^2` yields the MOND radius `sqrt(GM/a_Sigma)`. In a multi-source
system, the transition surfaces must instead come from the full vector
boundary-value solution. A minimal fixed elastic length is not automatically a
universal halo radius: at the MOND radius its spherical correction scales as
`L_Sigma^2/M_b`, preferentially changing dwarfs. V17F may retain one length
only if the frozen cross-cluster extent and full-field gates select it.

The spatial-state term remains target-blind and conditional. Both v17B cluster
region gates have passed (29 AS295 and 21 PLCK G287 regions), and the frozen
v17C spectral pipeline is currently processing them. If thermal stress fails
v17E, it is prohibited from the constitutive law. If it passes, v17F decides
between a source-local state and exactly one universal correlation length. A
covariant, healthy one-metric action is still required before any holdout. See
[`SIGMA_V18_POST_PRESSURE_FLUX_SELECTION.md`](SIGMA_V18_POST_PRESSURE_FLUX_SELECTION.md).

## 2026-08-04 Sigma v17Q pressure-symmetron no-go and mechanism reset

The standard non-derivative symmetry-restoring completion cannot rescue the
direct pressure-only reciprocal metric. The full Model S Solar pressure
profile gives `Pi_sun=3.42083e-6`. Every one of the 84 spent CLASH points that
requires an order-unity extra Weyl field has a larger pressure-screening column
than the Sun; their minimum, median, and maximum column ratios are 1.288,
23.422, and 78.123. Because standard symmetron charge suppression is monotone
in that column, cluster normalization imposes the coupling-independent bound
`abs(gamma-1)>=0.0767803`, or 3,338.27 Cassini limits.

An independent quartic boundary-value solve through all 2,402 Model S radii
confirms the ordering. Even the most favorable declared 12.5 kpc range leaves
93.716% of the Solar scalar charge, while Cassini permits only 0.02996%. Its
slip proxy is `0.0719553`, 3,128.49 Cassini limits; the boundary residual is
`4.08e-13` and the resolution change is `4.73e-11`. No holdout was opened and
no target was fitted.

V17G's unscreened propagation, v17P's conserved kinetic-flux screen, and
v17Q's symmetry-restoring charge screen are three materially distinct failures
at the same one-coupling Solar/cluster gate. The stopping rule is triggered:
the direct pressure-only reciprocal metric is retired, and a fourth pressure
screen is prohibited. The next root action must use a different baryon-forced
source or propagation mechanism. See
[`SIGMA_V17Q_PRESSURE_SYMMETRON_NO_GO.md`](SIGMA_V17Q_PRESSURE_SYMMETRON_NO_GO.md).

## 2026-08-04 Sigma v17P pressure flux-screen no-go

Moving the v17 pressure susceptibility out of the derivative-dependent matter
metric and into a local AQUAL/K-mouflage-like kinetic flux avoids the v17N
matter-Hessian theorem, but it fails a different necessary Solar gate. A
monotone kinetic screen suppresses the local force without changing the
integrated pressure scalar charge. Its exterior field therefore becomes linear
again and leaves an unavoidable outer potential.

For every positive monotone flux with `mu(0)=1`, supplying an order-unity
cluster Weyl field at the conservative spent-data floor `g_bar/a_sigma=0.1`
implies `abs(gamma-1)>=9.8185e-5` at 10 AU, or 4.269 Cassini limits. Twelve
polynomial witnesses verify the bound; their best full-potential proxy is
`6.7274e-4`, 29.25 Cassini limits, even though steep examples can make the
local-force proxy pass. No holdout or new target was opened.

The complete local monotone shift-symmetric pressure flux-screen class is
retired. A successor must change the source-integrated Solar scalar charge with
a healthy potential-dependent mechanism, or reset the direct pressure channel;
another kinetic curve is not authorized. See
[`SIGMA_V17P_PRESSURE_FLUX_SCREEN_NO_GO.md`](SIGMA_V17P_PRESSURE_FLUX_SCREEN_NO_GO.md).

## 2026-08-04 Sigma v17F root-scale propagator freeze

Before any regional temperature, thermal source map, v17E target score, or
v17F result existed, the conditional reduction from source discovery to one
scale equation was frozen. V17F can run only if every v17E gate passes. It
replaces the flexible three-scale thermal interpolation with

$$
(1-L_\Sigma^2\nabla_\perp^2)s_\Sigma=q_b,
\qquad
\Delta\kappa_\Sigma=\beta_\Sigma s_\Sigma,
$$

and derives both shear components from the same E-mode potential. One common
thermal source family and one common $L_\Sigma$ are selected by symmetric
cross-transfer; the amplitude is trained on one cluster and applied unchanged
to the other. $L_\Sigma=0$ is the explicit source-only limit. The upper grid
endpoint fails identifiability rather than authorizing a longer post-result
search.

The candidate must retain every full-field and $R_{50}/R_{80}$ gate, stay
within 5% of the flexible v17E error, yield positive directional amplitudes
that agree within 0.15 dex, and remain stable when resolution doubles. A pass
would select either a source-derived or one-length propagation mechanism for
covariant action derivation; it would not validate a theory. Helmholtz/Yukawa
propagation is explicitly registered as prior art. See
[`SIGMA_V17F_ROOT_SCALE_PROPAGATOR.md`](SIGMA_V17F_ROOT_SCALE_PROPAGATOR.md).

## 2026-08-04 Sigma v17E halo-scale identifiability gate

Before either spent v17 lensing target was opened, the thermal-stress transfer
protocol added an amplitude-independent spatial-extent test. After the
unchanged static response is transferred, the required and thermal-predicted
one-metric residual triplets are each reduced to the field energy
`delta_kappa^2+delta_gamma_1^2+delta_gamma_2^2`. Their independently centered
`R50` and `R80` radii must agree within 25% in both cluster-transfer directions,
and the predicted radii must change by at most 2% when map resolution doubles.

This separates a halo-strength correlation from a halo-size explanation. A
single coefficient cannot alter the radii, and a cluster-specific propagation
length remains prohibited. The source/propagator distinction, provisional
constant roles, and action consequences are derived in
[`SIGMA_HALO_SCALE_IDENTIFIABILITY.md`](SIGMA_HALO_SCALE_IDENTIFIABILITY.md).
No target value or fitted inverse coefficient existed when this gate was
frozen.

## 2026-08-04 Sigma v15 spent covariant-invariant inference

After the v14 gauge-carrier reset, v15 asks which baryonic information the
missing cluster Weyl Hessian actually depends on before selecting another
field. Every scalar or spin-two baryonic feature is projected into a
convergence/shear triplet from one lens potential; no coefficient can tune the
two shear components independently. One shared operator is transferred in both
directions between the spent AS295 and PLCK G287 maps.

The primary scale-only family scores `0.771407` cross-cluster NRMSE, above the
frozen `0.500` gate. Total-baryon tidal invariants worsen it to `0.797320`, and
gas--star overlap/orientation reaches `0.788787`. A post-failure sensitivity
adding 5 and 10 kpc structure improves the winner only to `0.749848`, or
`2.795%`, below the required ten-percent rescue. The compact-scale winner also
uses alternating unregularized scale coefficients and still fails transferred
power and shear gates into AS295.

Static local density, gradient, Hessian, and component-overlap information is
therefore insufficient over the tested 5--150 kpc scales. No holdout was opened
and the inverse coefficients are not theory constants. The next spent stage
must separate internal E modes from harmonic boundary modes and enlarge the
baryonic context before a genuinely new dynamical-state postulate is allowed.
See
[`SIGMA_V15_SPENT_INVARIANT_INFERENCE.md`](SIGMA_V15_SPENT_INVARIANT_INFERENCE.md).

## 2026-08-04 Sigma v14 local covariant gauge-carrier falsification

The gauge-reduced tidal question selected by the v14 reset fails its first
action gate in three materially different local covariant forms. Minimal
covariantization has the exact gauge residual
`R_mnr{}^s nabla_s(alpha)`. The partially-massless metric correction cancels
constant curvature but leaves `C_mnr{}^s nabla_s(alpha)`, so it fails on the
Weyl curvature that carries real galaxy/cluster tides. A conserved neutral
Bach current exists, but its local conformal fourth-order completion has
opposite propagator residues, `+1/m^2` and `-1/m^2`.

The full stress tensor is conserved but would restore the forbidden direct
rank-two mass charge; its trace-free projection and the flat neutral
improvement are not conserved on generic curved backgrounds. V14A, v14B, and
v14C therefore trigger the three-formulation reset for the local covariant
scalar-gauge rank-two carrier. No observational data were opened, and no
viable theory is claimed. See
[`SIGMA_V14_GAUGE_CARRIER_FALSIFICATION.md`](SIGMA_V14_GAUGE_CARRIER_FALSIFICATION.md).

## 2026-08-04 Sigma v14 mechanism reset and gauge-tidal postulates

The post-v13C mechanism reset is now evidence-complete. Nine carrier classes
cover the local nonmetricity, DHOST, retarded-memory, positive-spin-2,
higher-derivative clock, local first-gradient, ordinary spacetime-tensor,
material-memory, and preferred-clock/ADM-trace sequences. Four explicit
three-formulation resets are recorded. Ordinary four-dimensional p-forms add
only scalar/vector dual classes, and a direct scalar-charge rank-two gauge
field has a fourth-order static point-source equation, giving a constant force
rather than Newton's inverse-square force.

V14A therefore advances only a new action question: pair the v13B convex
AQUAL-like monopole target with a gauge-reduced, zero-monopole tidal response.
Direct baryonic mass charge, an ordinary spacetime-tensor component kinetic
term, a material triad, a localized retarded multiplier pair, and any
khronon/ADM-trace placement are forbidden. A four-constant budget is frozen.
The first kill gate is to derive a covariantly conserved neutral tidal source
and complete gauge-invariant action; no such action or viable theory is yet
claimed, and observations remain closed. See
[`SIGMA_V14_MECHANISM_RESET_AND_POSTULATES.md`](SIGMA_V14_MECHANISM_RESET_AND_POSTULATES.md).

## 2026-08-04 Sigma v13C khronon trace-completion falsification

The minimal one-metric covariant placement defines a khronon normal, uses its
expansion `Theta=nabla_mu u^mu` and acceleration, and adds the first-order
Legendre pair `L_C=p Theta-H_13B(p,a)`. Subtracting the canonical reference
preserves the exact v13B static AQUAL response. This static khronon lane and
its equal weak-field metric potentials are established Blanchet--Marsat prior
art; the tested distinction is only the v13B temporal completion.

At a static acceleration background, Legendre duality fixes the extra trace
curvature to `delta=w(1-epsilon)/(epsilon+a/a_sigma)>0`. The ADM kinetic term is
then `K_ij K^ij-lambda K^2` with `lambda=1+c_trace-delta/2`. Eliminating the
scalar shift gives the exact coefficient
`2(1-3lambda)/(1-lambda)`, which is negative for `1/3<lambda<1`. On the frozen
minimal row the ghost begins at `a/a_sigma=0.74999825`. At the `1e5` high-field
sentinel, the static force correction passes at `9.99999e-6`, while the reduced
scalar kinetic coefficient is `-799994.8`.

Every finite positive completion weight approaches the ghost interval from
below at high acceleration. A constant trace counterterm avoids the continuous
crossing only for `c_trace<=-2/3` or `c_trace>=499999.5`; both fail the GR
high-field limit. The trace modifier leaves both TT tensor polarizations
unchanged, but that does not repair the scalar ghost. v13C is rejected before
data. This is the third materially distinct post-v12 bounded-Hamiltonian
failure, so the preferred-foliation clock/ADM-trace mechanism reset is
triggered. See
[`SIGMA_V13C_KHRONON_COMPLETION_FALSIFICATION.md`](SIGMA_V13C_KHRONON_COMPLETION_FALSIFICATION.md).

## 2026-08-04 Sigma v13B convex reduced-carrier selection

v13B replaces the failed AeST clock/Jeans repair with the Hamiltonian-first
reduced carrier

$$
\mathcal H={a_\Sigma^2\over2}F_\epsilon
\left({\Pi^2+|\boldsymbol\nabla\sigma|^2\over a_\Sigma^2}\right),
$$

whose static flux is
`mu(t)=epsilon+(1-epsilon)t/(1+t)`. The exact phase-space Hessian has transverse
eigenvalue `mu` and radial eigenvalue
`epsilon+(1-epsilon)[1-1/(1+t)^2]`. Both remain in `[epsilon,1)`, proving a
positive strictly convex Hamiltonian and a globally unique Legendre map. The
reduced system has no energy linear in a signed dust charge.

For arbitrary momentum/gradient backgrounds and propagation directions, the
principal speeds obey `(c+b)^2=A C`. The relevant matrix `[[A,b],[b,C]]` is a
compression of the full Hessian, so both speeds are real and bounded by its
largest eigenvalue, which is below the preferred-frame unit cone. The frozen
scan reaches maximum `|c|=0.9999999999999899`; the independent numerical
flux-Jacobian residual is `4.6734e-10`. All reduced-carrier advancement gates
pass with two of five allowed physical constants.

This selects v13B for covariantization, not observational testing. The source
of the preferred foliation, joint metric constraint count, baryonic source,
single-metric photon equations, luminal tensor cone, PPN limits, and cosmology
remain open. `theory_viable=false`, no observations were opened, and the
post-v12 failure count remains `2`. See
[`SIGMA_V13B_CONVEX_CARRIER_SELECTION.md`](SIGMA_V13B_CONVEX_CARRIER_SELECTION.md).

## 2026-08-04 Sigma v12A constraint-solved modal-energy falsification

Exact v12A is retired before data under the project's strict bounded-energy
gate. For every finite oscillatory generalized eigenvector, the canonical
energy `u^dagger(omega^2 K-B)u/4` now agrees with the independent Krein
derivative `omega u^dagger(2 omega K-iC)u/4`; the worst normalized identity
residual is `4.27e-9`. The positive flat finite-frequency control passes and
the descriptor system retains `24` finite plus `16` constraint roots.

On the on-shell tilt-`0.5` sentinel, none of 19 common-time boosts and 13 wave
directions has positive energy for every physical mode. The best maximin time
still has normalized minimum `-0.783010`. At a resolved near-rest time, the
negative branch persists from `k=100` through `2000`; its raw canonical energy
moves from `-0.1617` to `-0.3175`, with four signed/real-phase negative roots
at every wave number.

The failure is inherited rather than caused by the v12A DHOST term. Signed
`lambda_D` values from `-8` through `+4` all retain a negative branch;
`lambda_D=0` has best sampled normalized energy `-0.829917`, while `+8` loses
the scanned hyperbolic time. A 43-row healthy-flat AeST `KB,K2` rest-frame
screen also has no positive-energy row. Approaching `KB=2` only drives the
mode toward the zero-speed strong-coupling endpoint.

Published AeST work describes the aligned low-momentum ancestor as possibly
Jeans-like. The project applies its stricter bounded-Hamiltonian rule on every
claimed background, so this is a falsification. It is failure `1/3` after the
v12 mechanism reset; no reset is triggered. The next formulation must remove
or positively constrain the inherited zero sector at action level rather than
tune v12A. See
[`SIGMA_V12A_REDUCED_ENERGY_FALSIFICATION.md`](SIGMA_V12A_REDUCED_ENERGY_FALSIFICATION.md).

## 2026-08-04 Sigma v12A on-shell constant-background common cone

The arbitrary-covector action now reproduces the established unitary ADM
Hessian to `1.10e-17` and preserves `X,U^2,Q,Y` under a general Lorentz boost
to `4.44e-16`. More importantly, the tilted constant backgrounds now satisfy
their own projected aether equation
`sqrt(Y) dL/dQ=0`. The prior roughly `1.89c` off-shell rows remain a global
warning but are not used as a physical-branch falsification.

The original frozen `KB=1,K2=2` row nevertheless fails on shell: at tilt
`0.5`, its fastest stable characteristic extrapolates to `1.002576c`. The sign
of `k^2-omega^2` is Lorentz invariant, so this excess cannot be removed by a
different slicing.

A frozen 15-pair theory-side screen of the already-present `KB,K2` constants
has one survivor: `KB=1,K2=4`, with flat scalar `c_s^2=1/4`. No parameter or
equation was added and no observation was opened. On on-shell tilts
`0.1,0.5,1,2,5,8`, both DHOST signs admit one sampled time direction common to
five wave orientations at the declared finite-`k` thresholds. The negative
sign is provisionally retained for its larger moderate-tilt hyperbolicity
margin. Five negative-sign convergence sentinels preserve `24` finite plus
`16` constraint roots at all `k=300,600,1000`; the worst absolute extrapolated
frequency excess is `5.03e-6` and maximum normalized growth is `0.00808`.
All eight constant-background gates pass.

This is not a viability result. Some best-frame coordinate-energy diagnostics
remain negative, reaching about `-34.4`, and no Dirac-reduced physical energy
has been constructed on the common time. Nonzero scalar Hessian, aether
gradient, extrinsic curvature, and spacetime curvature also remain open. The
next kill gate is the reduced physical Hamiltonian, followed by those
nonconstant backgrounds. See
[`SIGMA_V12A_ON_SHELL_CONES.md`](SIGMA_V12A_ON_SHELL_CONES.md).

## 2026-08-04 Sigma v12A finite-tilt scalar-unitary characteristic warning

The frozen finite-tilt characteristic grid keeps the homogeneous generalized
eigenpairs `(alpha,beta)` until constraint roots are classified, avoiding
spurious huge frequencies from dividing an exact Class-Ia `beta=0` root. All
72 backgrounds retain 24 finite roots and 16 roots at infinity, and none fail
the `1e-7` Euler-polynomial residual gate.

The physical scalar-unitary slicing gates nevertheless fail. Across both
coupling signs, clock ratios `0.5,1,2`, tilt magnitudes `0.1,0.5,1,2`, and
relative angles `0,45,90` degrees, 22 rows have growth proportional to wave
number, 31 exceed the 1% metric-frequency tolerance, and 38 contain a negative
oscillatory coordinate-time energy. The worst normalized growth is `0.90355`,
the largest frequency/light ratio is `1.36698`, and the minimum energy is
`-48.7730`.

Wave-number sentinels confirm three persistent effects: growth approaches
`0.45162 k` at preferred clock/perpendicular tilt 2; an off-clock branch
approaches `1.19319` times the metric light frequency; and a parallel branch's
energy approaches `-3.7808`. These are not lower-order finite-wave artifacts.

This proves that scalar-unitary metric time is not a universal healthy slicing,
but it does not yet prove that no other common metric-timelike Cauchy covector
exists. Exact v12A is held before data while the general-time invariant cone
and reduced-energy calculation is derived. See
[`SIGMA_V12A_TILTED_CHARACTERISTICS.md`](SIGMA_V12A_TILTED_CHARACTERISTICS.md).

## 2026-08-04 Sigma v12A direct flat characteristic regression

The full local quadratic Euler system is now converted to the generalized
characteristic pencil `P(s)=s^2K+s(A-A^T)-B`. All 26 lapse, shift, metric, and
aether real-phase amplitudes are retained until the Euler matrices are formed;
only then is `h13=h23=h33=0` imposed in each phase. The resulting 40-dimensional
first-order pencil has 24 finite roots and 16 constraint roots at infinity.
Because each physical configuration mode contributes four real-phase/time
roots, this directly reproduces the six local linear AeST modes.

Using the frozen differentiable tilt sentinel `1e-5` and extrapolating three
wave numbers in `1/k^2`, the scalar principal squared speed is `0.5000024999`,
while the tensor/vector interval is `0.9999999998--1.0000000002`. Four roots
remain in the zero-frequency sector. All twenty finite-frequency roots have
positive quadratic energy; the minimum normalized energy is `0.4000021`.
The maximum Euler-polynomial residual is `1.06e-11`, and the two `lambda_D`
signs give identical spectra because the v12A interaction vanishes to first
order on the flat clock.

All nine flat gates pass. The documented Jeans-like zero sector, finite-tilt
common-Cauchy cone and energy problem, nonlinear degree count, and nonconstant
backgrounds remain open. No data were opened. See
[`SIGMA_V12A_FLAT_CHARACTERISTICS.md`](SIGMA_V12A_FLAT_CHARACTERISTICS.md).

## 2026-08-04 Sigma v12A constant-background tilted Dirac block

The complete local quadratic action now retains all six spatial-metric
components, all three shifts, all three aether components, and the lapse before
constructing the Class-Ia nullspace. In a real sine/cosine Fourier basis, its
26-dimensional kinetic matrix has exactly six shift primaries plus two DHOST
primaries. After normalizing the latter to a unit physical-clock perturbation,
their conformal metric component agrees with
`delta zeta=-(r^3 A3_bar/4)delta r` to `1.67e-16`.

The exact reduced block is
`Z^T[-B-(A^T-A)K^+(A^T-A)]Z`. The antisymmetric mixing term includes the
dynamical longitudinal aether. It cancels the apparent Maxwell lapse-gradient
term, leaving a wave-number-independent constant-background bracket. A frozen
48-background signed scan per coupling sign spans clock and tilt magnitudes
from `1e-2` to `1e2` and arbitrary relative angles. All nine gates pass. The
closest eigenvalues to zero are `-1.2975747367` for `lambda_D=+1` and
`-4.0032986306` for `lambda_D=-1`; there are no rank or sign failures, and the
maximum wave-number residual is `3.15e-13`.

Both signs survive; neither is selected. This proves only the constant-
background Dirac pair. Physical characteristic cones and energy signs, then
background scalar Hessian, aether gradient, extrinsic curvature, and curvature
remain open. No data were opened. See
[`SIGMA_V12A_TILTED_PRINCIPAL.md`](SIGMA_V12A_TILTED_PRINCIPAL.md).

## 2026-08-04 Sigma v12A corrected aligned finite-wave-vector gate

The earlier positive-sign falsification is withdrawn. It projected only the
`A4 q^2|Dq|^2` clock block and omitted the conformal metric component of the
Class-Ia primary null direction. With
`delta zeta=-(r^3 A3_bar/4)delta r`, the direct DHOST, Einstein cross, and
Einstein metric coefficients obey
`r^2 A4_bar-4 eta/r+2 eta^2=0` exactly. That correction still held the
longitudinal aether fixed. In the full Maxwell square
`K_B|dot A_L+i k delta N|^2`, the aether momentum contributes a Schur term
`-K_B/r^2` that cancels the apparent direct `+K_B/r^2` lapse gradient. The
complete aligned symbol is therefore `Delta/F0=-4K2`, which is nonzero for
`K2>0` and does not constrain the sign of `lambda_D`.

The corrected 50,000-row scan reproduces the old incomplete zero as a
regression fixture, verifies both cancellations, and keeps the exact positive
core at `4K2=8` for both signs and all wave numbers. No data were opened. The
constant tilted Dirac block is now also complete, but physical characteristics
and nonconstant backgrounds remain open; the theory is still not viable. See
[`SIGMA_V12A_ALIGNED_FINITE_K.md`](SIGMA_V12A_ALIGNED_FINITE_K.md).

## 2026-08-04 Sigma v12A tilted reduced-AeST clock susceptibility

The exact lower-derivative AeST contribution to the v12A constraint bracket is
strictly nonzero for arbitrary finite aether tilt and scalar-gradient
orientation. With `Q=chi q+A.s` and `Y=-q^2+s^2+Q^2`, Cauchy gives
`|A|^2Y-Y_q^2/4=|A|^2|s|^2-(A.s)^2>=0`. Applied to the fixed simple
interpolation, this yields

`d2L_AeST/dq2 >= 4K2+[4K2-(9/2)(2-KB)]|A|^2`.

Thus `K2>=9(2-KB)/8` is a sufficient all-tilt condition. The selected
`KB=1,K2=2` row has the strictly positive bound `8+(7/2)|A|^2`. A 50,000-row
signed logarithmic scan, including 10,000 exact `Y=0` projected-axis fixtures,
checks the direct susceptibility and analytic inequalities. All six gates
pass: the minimum susceptibility is `8.00000000000402`, the maximum normalized
Cauchy-identity error is `5.95e-16`, and the lower-bound violation is zero.

This computes the reduced AeST zeroth-order part of `Delta_eff`, not the DHOST
spatial differential operator. A finite-wave-vector cancellation or principal
rank change remains possible and is the next kill gate. No observation was
opened. See [`SIGMA_V12A_TILTED_CLOCK.md`](SIGMA_V12A_TILTED_CLOCK.md).

## 2026-08-04 Sigma v12A homogeneous aligned Dirac branch

The homogeneous aether-aligned branch passes its primary-secondary test. With
`q=nabla_n phi`, normal scalar Hessian `V_*`, and metric trace `K`, the v12A
invariants give `kappa=-2F0/3`, `b=-q^3 A3/2`, and
`a=-3q^6 A3^2/(8F0)=b^2/kappa`. The canonical primary is
`Psi=p_q-(b/kappa)pi_K`, and the reduced Hamiltonian loses all dependence on
`V_*` and `b(q)`.

The secondary is `Omega=-p_phi+dL_AeST/dq`. At the intended clock `q=Q0`, the
v12A activation and `b` vanish exactly, but
`{Psi,Omega}=-d^2L_AeST/dq^2=-4K2`; it is `-8` on the selected `K2=2` row.
Thus the new auxiliary Hessian coordinate is not strongly coupled at the flat
vacuum. A 4,001-point signed clock scan verifies the exact degeneracy and
Legendre identities.

This is not the arbitrary-gradient result. Spatial scalar gradients, aether
tilt, anisotropic metric velocity, and the full differential `Delta_eff` remain
the next kill gate. No observation was opened. See
[`SIGMA_V12A_HOMOGENEOUS_DIRAC.md`](SIGMA_V12A_HOMOGENEOUS_DIRAC.md).

## 2026-08-04 Sigma v12A canonical primary and conditional Dirac chain

The published reduced AeST metric momentum is exactly the GR momentum, so the
AeST base does not shift the standard Class-Ia primary constraint. Its explicit
canonical form is
`Psi=p_*-2(K^-1 B).pi+2 sqrt(h)[(K^-1 B).C-C0]`. Two thousand arbitrary
velocity/coefficient trials satisfy the identity to the frozen tolerance.

`Psi` commutes with the AeST `mu,nu` auxiliary primaries, so its preservation
necessarily generates `Omega={Psi,H0}=p_phi+Omega_rest`. After including the
AeST auxiliary secondary pairs, the exact remaining regularity condition is
the Schur bracket `Delta_eff=Delta-E C^-1 D`; the full Dirac determinant is
`det(C)^2 Delta_eff^2`. Random matrices verify the identity, and an exact
`Delta_eff=0` fixture makes the chain singular.

This does not yet prove the v12A chain regular. The explicit secondary density
and model-specific differential operator `Delta_eff` must be derived and shown
invertible on arbitrary scalar-gradient/aether backgrounds, including the flat
clock where the new activation vanishes. No observation was opened. See
[`SIGMA_V12A_PRIMARY_DIRAC.md`](SIGMA_V12A_PRIMARY_DIRAC.md).

## 2026-08-04 Sigma v12A unreduced joint ADM kinetic-rank subgate

The exact highest-velocity AeST--DHOST block preserves the Class-Ia primary
null direction. After introducing `B_mu=nabla_mu phi`, all `V_*` dependence is
inside the standard DHOST block. The AeST Maxwell field strength is connection-
free and adds a positive `K_B I_3` aether-velocity block. `Y`, `Q`, and
`F(Y,Q)` are configuration functions, while `J^mu B_mu` is affine in metric and
aether velocities and therefore shifts momenta without changing the Hessian.

Thus `H_total=diag(H_DHOST,K_B G_E)` with positive electric metric `G_E`. Two
thousand random Schur-degenerate DHOST blocks and random positive spatial-metric
congruences retain one exact null direction; the inertia changes only from
`(1,1,5)` to `(1,1,8)`, adding three positive modes. A finite-difference test
confirms arbitrary linear AeST momentum shifts leave the Hessian unchanged.

This is an unreduced kinetic statement, not a full Dirac or arbitrary-tilt
health pass. The primary constraint must be expressed in canonical variables
and preserved in time against the AeST auxiliary constraint chain to prove a
regular secondary and unchanged physical degree count. No observation was
opened. See
[`SIGMA_V12A_JOINT_ADM_RANK.md`](SIGMA_V12A_JOINT_ADM_RANK.md).

## 2026-08-04 Sigma v12A same-clock DHOST selection

The post-v11 reset selects a materially new theory-only lane. V12A adds no
memory field or object state. It retains the one-metric AeST galaxy clock and
uses its scalar Hessian in the published luminal Class-Ia DHOST basis with
`A1=A2=0`, `A4=-A3-X^2 A3^2/(8F0)`, and `A5=X A3^2/(2F0)`.

The provisional background-zero `A3` shape and its first derivative vanish on
the AeST flat clock. The frozen simple interpolation has asymptotic coefficient
one but local tangent `f_y(0)=0`, so the corrected finite-frequency squared
speeds are tensor `1`, vector `1`, and scalar `1/2`; the earlier `3/4` entry used
the wrong tangent. The signed scan keeps normalized degeneracy
residuals below `1e-12` and all normalized coefficients finite. Static `L3-L5`
invariants distinguish equal-trace isotropic and rank-one Hessians and remain
rotation covariant. Solar-boundary activation is below `1e-5`, five universal
constants are retained, and no observation was opened.

This is not yet a theory pass. The decisive next gate is the complete combined
AeST--DHOST ADM Hessian and constraint chain on arbitrary tilted backgrounds;
separate degeneracy of the two sectors does not prove joint degeneracy. See
[`SIGMA_V12A_SAME_CLOCK_DHOST_SELECTION.md`](SIGMA_V12A_SAME_CLOCK_DHOST_SELECTION.md).

## 2026-08-04 Sigma v11C Biot-stretch falsification and mechanism reset

Exact v11C is retired before data. It replaces v11B's Green strain by the Biot
strain `S=sqrt(D^T D)-I`. This repairs the exact one-dimensional quartic: the
vacuum longitudinal tilted Hessian is `gamma^2(1-3v^2/4)>0`. The global rank
gate nevertheless fails on an orientation-preserving anisotropic stretch.

For `D=diag(e,e,M)`, the rank-one `e1 tensor e2` Biot curvature is
`K=s[2+{-2+2(b-1/3)(M+2e-3)}/(2e)]`, and a slice tilted by `v` has physical
coordinate-velocity Hessian `H=gamma^2(1-v^2 K)`. With `v=1/2`, `e=1/10`,
`s=3/11`, and `b=17/24`, the finite rank surface is `M*=398/45`. Material flow
is exactly comoving with the timelike aether and `det D=e^2 M>0`. At `M=10`,
`K=57/11` and `H=-13/33`; an independent finite difference agrees.

V11A's bounded scalar alignment, v11B's Green-strain triad, and v11C's
Biot-stretch completion are three materially distinct post-v10-reset closures
failing the same nonlinear kinetic-rank gate. The stopping rule triggers:
reset the material-memory mechanism and do not add v11D. No observation was
opened. See
[`SIGMA_V11C_BIOT_STRETCH_FALSIFICATION.md`](SIGMA_V11C_BIOT_STRETCH_FALSIFICATION.md).

## 2026-08-04 Sigma v11B tilted-flow kinetic falsification

Exact v11B is retired before data. On a tilted Minkowski slice, perturbing one
material coordinate gives `E_11=2aw+a^2w^2`. Its positive spatial strain energy
enters the Lorentzian Lagrangian with a minus sign, producing a negative
quartic in the coordinate velocity. The exact Hessian is
`H=gamma^2-s(2/3+b)(2a^2+6a^3w+3a^4w^2)`.

At `v=1/2`, `s=3/11`, and `b=17/24`, the finite zero is
`w=1.68359945`, where material flow remains timelike at `-0.958040` and the
Lagrangian is finite. The Hessian crosses `0.0143410`, `0`, `-0.0144119` at
`0.99`, `1`, `1.01` times that velocity; all three material velocities remain
subluminal. A negative physical-coordinate Rayleigh direction cannot be fixed
by omitted mixing. This is the second distinct post-reset failure at nonlinear
kinetic rank. No observation was opened. See
[`SIGMA_V11B_TILTED_RANK_FALSIFICATION.md`](SIGMA_V11B_TILTED_RANK_FALSIFICATION.md).

## 2026-08-04 Sigma v11B stress-free elastic-triad selection

The second post-reset architecture passes its flat selection gate. V11B adds
three spacetime scalars `X^I` with internal Euclidean symmetry. The aether-time
velocity is `Q^I=A.nabla X^I`, while
`E^IJ=q^mn nabla_m X^I nabla_n X^J-delta^IJ` is the spatial strain. The action
is a positive `Q^2` time square minus positive trace/STF strain squares.

The unstrained reference `X^I=x^I` has `Q=E=0`, so the action, first variation,
and effective stress vanish. Linear displacements have two shear squared
speeds `3/11` and one longitudinal squared speed `3/4`; these fix the bulk
weight to `17/24`. Two thousand random directions reproduce that spectrum to
`8.88e-16`, with every mode positive and causal. Scalar derivatives add no
metric connection principal term, so the TT gravitational front remains
luminal. An algebraic graviton mass remains possible and is explicitly a
later gate.

Only `L_Sigma` is new, retaining five physical constants. V11B is not yet a
theory pass: nonlinear tilted rank, the complete constraint algebra, weak
metric sign/amplitude, source uniqueness, graviton mass, Solar/PPN, and
cosmology remain unresolved. No observation was opened. See
[`SIGMA_V11B_ELASTIC_TRIAD_SELECTION.md`](SIGMA_V11B_ELASTIC_TRIAD_SELECTION.md).

## 2026-08-04 Sigma v11A tilted nonlinear kinetic falsification

Exact v11A is retired before data. On a local Minkowski slice with finite
aether tilt `v=1/2`, choose `partial_t phi` and a finite orthogonal spatial
memory gradient `partial_x chi`. The supposedly spatial AeST gradient then has
`S:S=gamma^2 v^2 dot(phi)^2`, so the bounded alignment becomes
`z=c dot(phi)^2/(1+c dot(phi)^2)`. Its exact curvature is
`z''=2c(1-3c dot(phi)^2)/(1+c dot(phi)^2)^3` and is negative at finite
velocity.

The alignment energy is proportional to `(D_x chi)^2 z`. At
`c dot(phi)^2=1`, the total scalar velocity Hessian is
`H_phi-s alpha gamma^2 c (D_x chi)^2/4`. It crosses zero at a finite memory
gradient for every finite positive base Hessian. With the conservative
`H_phi=8`, the exact surface is `dot(phi)=sqrt(3)` and
`D_x chi=sqrt(1056)=32.4962`; the Lagrangian is finite. The Hessian is
`0.1592`, `0`, and `-0.1608` at `0.99`, `1`, and `1.01` times that gradient.

A negative one-coordinate Rayleigh direction cannot be repaired by omitted
off-diagonal velocity mixing. Zero anisotropy returns to the already-retired
v4 isotropic-memory lane, and a fitted gradient cutoff is forbidden. This is
the first failed closure after the v10 reset; it does not yet trigger another
three-closure reset. See
[`SIGMA_V11A_TILTED_RANK_FALSIFICATION.md`](SIGMA_V11A_TILTED_RANK_FALSIFICATION.md).

## 2026-08-04 Sigma v11A anisotropic scalar-memory selection

The first post-reset candidate passes its fixed-background selection gate.
V11A uses one massive scalar memory with bounded spatial kinetic tensor
`C^mn=s[q^mn-(1-u)S^mS^n/(a_Sigma^2+S:S)]` and source
`beta D_m chi J^m`. The coefficients remain derived: `u=3/4`, `s=3/11`,
`beta^2/K_B=2/11`, and anisotropy fraction `1-u=1/4`. Only the memory length
`L_chi` is new, leaving five physical constants total.

For every field magnitude and wave direction,
`9/44<=s_eff<=3/11`. The worst static Schur margin is `1/44`. All 80,601
fixed-background magnitude/angle cases have real positive mixed roots no
greater than one; the endpoints are `(9/44,1)` and
`(0.156573,0.979790)`. A scalar derivative contains no metric connection, so
the aether-rest TT metric principal symbol avoids the exact v10D failure.

This differs from v4's retired isotropic scalar memory because the propagation
operator is directionally aligned by the baryon-forced AeST field. It still
needs complete variation, nonlinear global rank, tilted/nonzero-gradient
cones, weak `Psi/Phi` and lensing equations, PPN/Solar limits, and numerics.
No observation was opened. See
[`SIGMA_V11A_ANISOTROPIC_SCALAR_MEMORY_SELECTION.md`](SIGMA_V11A_ANISOTROPIC_SCALAR_MEMORY_SELECTION.md).

## 2026-08-04 Sigma v10D tensor-cone falsification and mechanism reset

Exact v10D fails the decisive nonzero-background metric characteristic gate.
For an axisymmetric carrier
`P=diag(p_perp,p_perp,p_parallel)` and a wave along its symmetry axis, axial
spin makes the two TT metric polarizations an exact invariant sector. The
unit-determinant field redefinition that diagonalizes the carrier time
derivative leaves a spatial connection residual with
`R:R=(p_parallel-p_perp)^2(h:h)/2`. Hence

`c_TT^2=1+c_P^2(p_parallel-p_perp)^2`.

Every nonzero anisotropy is outside the one-metric null cone. At the frozen
`c_P^2=3/11`, an illustrative anisotropy of only `1e-6` already gives a
relative speed excess `1.3636e-13`, above the declared `1e-15` tolerance. The
stronger failure is analytic and amplitude-independent: the theory admits
arbitrary anisotropic carrier backgrounds, and all nonzero values are
superluminal. The aether exponential, lapse, scalar, and vector constraints do
not enter this spin-two sector.

V10B's instantaneous auxiliary tail, v10C's finite-amplitude vector ghost, and
v10D's widened TT cone are three materially distinct closures of the same
aether-tidal carrier mechanism failing Action 12's common
causality/stability gate. The preregistered three-closure rule is triggered:
do not patch a v10E; reset mechanism selection. No observational product or
holdout was opened. See
[`SIGMA_V10D_TENSOR_CONE_FALSIFICATION.md`](SIGMA_V10D_TENSOR_CONE_FALSIFICATION.md).

## 2026-08-04 Sigma v10D aether-rest ADM rank

The nonlinear local Legendre-rank subgate passes. In an aether-rest ADM frame,
the carrier velocity is the perfect square
`W_ij=dot(P)_ij-K_i^kP_kj-K_j^kP_ik`. The transformation from
`(dot(h),dot(P))` to `(dot(h),W)` is triangular with determinant one for every
carrier background. The metric--carrier Hessian is therefore congruent to the
Einstein-Hilbert DeWitt block plus six positive squares and has constant
inertia `(1,0,11)`.

Adding the globally positive completed aether block and selected positive AeST
clock gives full rest-frame inertia `(1,0,15)`. One thousand mixed-sign
carrier tensors confirm the analytic result. The generic constraint count
retains the published AeST four first-class/four second-class structure and
adds six regular carrier modes, for twelve physical degrees of freedom. The
arbitrary-foliation constraint rank and full anisotropic metric cones remain
the next kill gate. No observation was opened. See
[`SIGMA_V10D_ADM_RANK.md`](SIGMA_V10D_ADM_RANK.md).

## 2026-08-04 Sigma v10D anisotropic source-block characteristics

The arbitrary-orientation fixed-metric aether--carrier block passes. For every
carrier background, `F=exp(X)-X>=I`; for every wave direction,
`R=(I+n n^T)/2` lies between `I/2` and `I`. The static Schur complement is
therefore at least `I-(q/s)I=I/3`. The six sourced squared speeds solve
`M(y)=F y^2-(sF+uI+qR)y+suI`; their projected polynomial is positive at the
metric cone because `(1-s)(1-u)-q=0`, the original cone-saturation identity.

Two thousand random noncommuting carrier/direction cases give real, positive
roots no greater than one, and subluminal rest-frame speeds remain subluminal
under boosts through `|v|=0.999`. A nonzero background aether acceleration
does not change this derivative Hessian. The proof still excludes the
dynamical metric and complete AeST scalar/constraint sector, which is now the
decisive gate. No observation was opened. See
[`SIGMA_V10D_ANISOTROPIC_CHARACTERISTICS.md`](SIGMA_V10D_ANISOTROPIC_CHARACTERISTICS.md).

## 2026-08-04 Sigma v10D exponential kinetic selection

V10D is a parameter-free nonlinear successor to the retired v10C action. It
adds `K_B J.[exp(X)-I].J`, with `X=(beta/K_B)P`, so the constraint-reduced
physical vector kinetic matrix becomes `K_B[exp(X)-X]`. For every real carrier
eigenvalue, `exp(x)-x` has its unique global minimum one at zero. The v10C
finite-amplitude singularity and ghost region are therefore absent without a
carrier cutoff or sixth constant.

The completion begins at cubic perturbative order, so the v10C zero-background
static response and cone equations are unchanged. A scan of kinetic factors
from one through `1e8` keeps longitudinal and transverse mixed squared speeds
positive and no greater than one, and keeps the static block positive. This is
only a successor selection: projector/metric velocity mixing, nonzero-`J`
characteristics, full ADM constraints, PPN/Solar limits and numerics remain
unresolved. No observation was opened. See
[`SIGMA_V10D_EXPONENTIAL_KINETIC_SELECTION.md`](SIGMA_V10D_EXPONENTIAL_KINETIC_SELECTION.md).

## 2026-08-04 Sigma v10C nonlinear kinetic falsification

Exact v10C is retired before data. On a locally inertial aether-rest event,
differentiating `A^mP_mi=0` gives `dot(P)^0i=P_ij dot(A)^j`. The first-order
interaction therefore changes the physical aether-vector kinetic density to
`dot(A)^T[K_B I-beta P]dot(A)`. The full reduced nine-velocity Hessian agrees
with finite differences below `1e-9`.

For `P_ij=p delta_ij`, the vector coefficient crosses zero at
`p_star=K_B/beta=sqrt(11K_B/2)`, or `2.34521` at `K_B=1`. It is negative just
above that finite amplitude. The convex quartic potential remains finite at
the surface and the spatiality constraint imposes no amplitude bound, so the
hyperbolic carrier admits initial data at and beyond the kinetic singularity.
These are the published physical AeST vector modes, not lapse/shift gauge
directions. No observation was opened. See
[`SIGMA_V10C_NONLINEAR_KINETIC_FALSIFICATION.md`](SIGMA_V10C_NONLINEAR_KINETIC_FALSIFICATION.md).

## 2026-08-04 Sigma v10C covariant variation subgate

The v10C carrier has now been written with exact projected time and spatial
derivatives. Its derivative momentum is
`Pi^{r|mn}=A^r dot(P)^mn-(3/11)D^rP^mn`; a numerical directional derivative on
a tilted unit-aether background matches below `1e-9`. The four independent
conditions `A^mP_mn=0` reduce a symmetric four-tensor from ten to six spatial
components exactly.

Integration by parts changes `beta P^mn nabla_m J_n` into
`-beta(nabla_mP^mn)J_n`, so the action contains only first derivatives. The
carrier, aether, metric, scalar and multiplier Euler equations are at most
second order. The complete all-field diffeomorphism identity has been derived,
so the metric source is conserved on the nonmetric field equations while
ordinary matter remains conserved through its one minimal metric coupling.

This passes only the variation/order gate. The component-expanded metric
stress, nonlinear ADM Hessian and constraint chain, arbitrary-background
characteristics, PPN/Solar solution and numerics remain unresolved. No
observation was opened. See
[`SIGMA_V10C_COVARIANT_VARIATION.md`](SIGMA_V10C_COVARIANT_VARIATION.md).

## 2026-08-04 Sigma v10C covariant/PPN applicability precheck

The selected spatial-aether counterterm has now been reduced exactly on the
unit constraint. The identity `F^2=B^2-2J^2` maps the aether sector to
`-K_B u F^2/2+K_B(1-u)J^2`, or standard pure-aether coefficients
`(c1,c2,c3,c4)=(K_Bu,0,-K_Bu,K_B(1-u))`. Hence `c13=0`, `c14=K_B`, and the
pure-aether vector-speed proxy is `u=3/4` exactly.

The Foster--Jacobson pure Einstein-aether formula gives `alpha1=-4K_B` both
before and after the counterterm, so the counterterm does not create a new
proxy failure. The same substitution has `c123=0`, where the pure-aether
`alpha2` formula is singular. Because v10C contains the AeST scalar and the
new hyperbolic `P` carrier absent from that theory, importing the pure-aether
number as v10C's prediction would be invalid. The full moving-source
AeST-plus-`P` PPN parameters remain unresolved and the Solar/PPN gate remains
false. The interaction is boundary-equivalent to a first-derivative action,
so a complete second-order covariant variation remains viable. No observation
was opened. See
[`SIGMA_V10C_COVARIANT_PPN_PRECHECK.md`](SIGMA_V10C_COVARIANT_PPN_PRECHECK.md).

## 2026-08-04 Sigma v10C hyperbolic aether-tidal selection

V10C restores a positive time kinetic term to the v10B aether-acceleration
Hessian carrier. The coefficients are derived rather than independently
fitted. Requiring static mixing fraction `q/s=2/3` (capacity three) and an
upper mixed cone exactly equal to one at sourced base speed `u=3/4` gives
`c_P^2=s=3/11` and `beta^2/K_B=q=2/11`. A fixed spatial-aether magnetic
counterterm lowers the bare transverse vector speed from one to `3/4`, equal
to the frozen AeST scalar speed, without changing the physical TT cone.

The necessary theory-only gates pass. The longitudinal static determinant is
`1/11`, minimum eigenvalue `0.0759624`, and Schur complement `1/3`. Its mixed
squared speeds are `9/44` and `1`. Canonical transverse modes have squared
speeds `0.232009/0.881627`; unmixed P modes have `3/11`. A flat TT wave remains
unsourced with `c_T^2=1`. The time-kinetic carrier has a retarded finite front
rather than v10B's equal-time tail; strict static convexity plus zero boundary
data select one stationary profile. The same linear metric correction changes
dynamics and Weyl lensing, and trace/STF geometry remains nonzero.

This is selection only. The upper longitudinal cone has zero safety margin, so
complete covariant variation, nonlinear ADM constraints, and tilted,
inhomogeneous, nonzero-P, and FLRW characteristics are the immediate kill
gates. No observation was opened. See
[`SIGMA_V10C_HYPERBOLIC_AETHER_TIDAL_SELECTION.md`](SIGMA_V10C_HYPERBOLIC_AETHER_TIDAL_SELECTION.md).

## 2026-08-04 Sigma v10B auxiliary aether-tidal sequence

V10B keeps the six-component trace/STF geometry but sources it with
`D_(m J_n)`, where `J_m=A^n nabla_n A_m` is the aether acceleration, and makes
the tensor auxiliary rather than propagating. The fixed coefficient
`beta^2=2 K_B/3` leaves the worst static Schur complement `K_B/3`. At `K_B=1`,
the longitudinal static eigenvalues are `0.183503/1.816497` and the transverse
ones are `0.422650/1.577350`. The response capacities are `3` longitudinal and
`1.5` transverse; the first closes `93.26%` of the spent unit-to-`3.14465`
amplitude gap. The linear static interaction changes the physical lapse while
leaving the AeST traceless/no-slip and flat-TT equations unchanged, so dynamics
and Weyl lensing receive the same metric correction at this order.

The Dirac reduction succeeds. Six `pi_P=0` primaries generate six secondary
elliptic equations; their bracket operator is positive, so all twelve
constraints are second class and remove all six P configuration states. The
reduced Hamiltonian is positive and the flat vector squared speeds decrease
from one to `0.6` longitudinally and `0.75` transversely.

The exact causal-front gate fails. Eliminating a finite-range second-class P
produces a same-time Yukawa tail. In the physical transverse channel its local
coefficient is `3/4`, inverse range `sqrt(3/4)/L_P`, and nonlocal coefficient
`3/(16 L_P^2)`. It is nonzero at every radius on the same aether slice. This is
not a first-class lapse gauge constraint, so exact v10B is retired before data.
The positive aether-tidal static block survives as the starting point for a
hyperbolic causal carrier. See
[`SIGMA_V10B_AUXILIARY_AETHER_TIDAL_FALSIFICATION.md`](SIGMA_V10B_AUXILIARY_AETHER_TIDAL_FALSIFICATION.md).

## 2026-08-04 Sigma v10A spatial-polarization sequence

V10A is a six-component symmetric tensor constrained to the AeST aether's
spatial slice and sourced by the projected symmetric derivative of the AeST
scalar spatial gradient. Its trace supplies an isotropic response and its STF
components preserve tidal/shear orientation. The selected fixed coefficients
are `c_P^2=1-c_s^2=1/4` and `beta=c_s^2/2=3/8`, leaving the provisional physical
budget at five constants.

The theory-only selection identities pass. The normalized scalar--longitudinal
carrier block has squared speeds `0.0493061` and `0.950694`; the other five
carrier components have squared speed `0.25`. The carrier potential is strictly
convex, the local response is rotation covariant to `1.58e-16`, the frozen
two-source probe is `34.08%` nonadditive, and equal-`g_bar` systems are
distinguished because their Hessian source scales as `M/r^3`. The flat response
capacity is `4`, above the spent factor-`3.14465` target. No observation was
opened.

The exact next gate retires the action. On the simple-mu quasistatic branch,
`K_T=x/(1+x)` and `K_L=x(x+2)/(1+x)^2` both vanish as `x` tends to zero. The
constant mixing requires `K>beta^2/c_P^2=0.5625`. Transverse perturbations are
therefore non-elliptic below `x=9/7`, longitudinal perturbations below
`x=4/sqrt(7)-1`, and the zero-field principal matrix has eigenvalue
`-0.270285`. The mass and convex quartic are order-`k^0` and cannot repair an
order-`k^2` sign. Every nonzero constant beta fails; beta zero removes the
mechanism. Exact v10A is retired before data. See
[`SIGMA_V10A_SPATIAL_POLARIZATION_FALSIFICATION.md`](SIGMA_V10A_SPATIAL_POLARIZATION_FALSIFICATION.md).

## 2026-08-04 Sigma v9B local first-gradient mechanism closure

The v9A spherical null generalizes. For every regular, shift-symmetric local
quasistatic completion `F(Y,Z,U)`, spherical integration fixes each constitutive
flux by `G M_b(<r)/r^2=g_bar`. A single-valued inverse and universal boundary
condition therefore make the physical enhancement one universal function of
`g_bar`; changing the angular function changes that RAR but cannot produce two
answers at the same local acceleration.

The retrospective spent-data audit makes the conflict quantitative. All 72
cluster development points lie inside the 968-point SPARC outer acceleration
range. Their median nearest cross-domain separation is `0.001448 dex`, but the
median required enhancement gap is `0.50934 dex`, a factor `3.231`; all gaps are
positive and 70/72 exceed `0.2 dex`. A ten-neighbor comparison gives the same
`0.50650 dex` median. The declared local-state conflict gate passes. The CLASH
values are NFW-deprojected and this is not a raw-lensing likelihood, so the
closure is limited to the exact theorem assumptions and the current development
target.

The regular local first-gradient lane is closed. Object-specific branches or
integration charges are forbidden hidden halo states. The successor must add a
uniquely baryon-forced finite-environment/tidal variable with both a nonzero
spherical monopole and a traceless shear response. See
[`SIGMA_V9B_LOCAL_FIRST_GRADIENT_CLOSURE.md`](SIGMA_V9B_LOCAL_FIRST_GRADIENT_CLOSURE.md).

## 2026-08-04 Sigma v9A bounded first-derivative alignment sequence

V9A replaced v8B's higher-derivative clock completion with a first-derivative
Gram interaction between the projected scalar gradient and aether acceleration.
The direct term is quartic around flat space, keeps the published AeST quadratic
spectrum, uses five constants, and has an exact small-vector kinetic bound
`K_perp=K_B-4 eta y/(1+y)^2`.  Nevertheless, its full six-variable static
principal matrix changes inertia at finite aether acceleration for every tested
nonzero coupling. At the selected `K_B=1`, `eta=2/3`, the first surface is
`Y/a_sigma^2=1`, `Z/a_sigma^2=3.498466`, with a mixed `54.81%` aether and
`45.19%` scalar null direction.

The minimal double saturation bounds the complete interaction by
`eta a_sigma^2`. It preserves the AeST static inertia across 2,212 deterministic
and random points, with minimum singular value `1.369856`. It still fails the
mechanism gate exactly: for aligned gradients, `YZ-U^2=0` and both variation
fluxes vanish. Every spherical system is therefore unchanged from fixed
AeST/MOND for every eta. The existing mean cluster amplitude target is `3.14465`,
whereas the selected best-case perpendicular response is `3`; closing even 75%
of the gap requires at least `74.10 degrees` of misalignment, while spherical
fields have zero. No new observational data were opened.

Exact v9A is retired as the standalone unification completion. Another angular
gate is not authorized. The successor must generate a baryon-forced nonzero
spherical monopole and orientation/shear transport in the same healthy action.
See
[`SIGMA_V9A_BOUNDED_ALIGNMENT_FALSIFICATION.md`](SIGMA_V9A_BOUNDED_ALIGNMENT_FALSIFICATION.md).

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

## 2026-08-04 Sigma v8A one-metric Weyl-active selection

The successor is a one-metric AeST action plus one cubic Horndeski interaction.
Both ingredients are prior art; only their fixed combination and proposed use as
a cluster-geometry correction remain candidates for novelty.  Unlike v7C, the
AeST kinetic mixing puts the scalar directly into the physical metric with
`delta Psi=delta Phi=delta Weyl`.  The cubic equation distinguishes equal-trace
Hessians, giving responses `6` and `0` for isotropic and rank-one curvature.

At the frozen construction point, the published AeST flat-background tensor,
vector, and scalar squared speeds are `1`, `1`, and `0.75`.  The cubic term is
third order in flat perturbations, so it does not change that quadratic result.
These are the healthy finite-frequency propagating modes, not a proof of a
positive Hamiltonian at every momentum.  The published `omega=0` sector also
contains a constant zero-Hamiltonian mode and a linearly growing mode whose
Hamiltonian is negative below a scale of order `mu`; it is described as
Jeans-like but remains an explicit project warning.  The five-parameter row
passes selection only. Full combined variation, constraint counting, nonlinear
characteristics, PPN, Solar screening, and source uniqueness remain mandatory
before data. See
[`SIGMA_V8A_AEST_GALILEON_SELECTION.md`](SIGMA_V8A_AEST_GALILEON_SELECTION.md).

### v8A cubic nonlinear-characteristic result

The exact cubic interaction is now retired before observational use.  On its
positive spherical exterior branch, the radial scalar characteristic crosses
the physical light cone when the cubic supplies only `17.713%` of the conserved
scalar flux, and tends to `c_r^2=4/3` in the nonlinear limit.  Reversing the sign
makes the positive-source branch end where radial ellipticity vanishes.  Zero
coupling is healthy but removes the proposed geometry response.  The one-metric
AeST base is retained for a bounded replacement interaction.  See
[`SIGMA_V8A_CUBIC_CHARACTERISTIC_GATE.md`](SIGMA_V8A_CUBIC_CHARACTERISTIC_GATE.md).

### v8B preferred-time causal completion selection

AeST's existing unit timelike vector permits a causal partner for the cubic
static interaction. The added operator is
`(alpha-1) L_H^2 (Q-Q0)^2 D^2 phi`; it vanishes on the static background and
adds only to the perturbation time kinetic coefficient in the fixed-aether
scalar limit. The coefficient is derived rather than fit:
`alpha=1/[3 c_s^2(1-c_s^2)]=16/9`.

The `20,001`-point positive spherical scan is positive and causal. Its maximum
radial squared speed is `1`, with deep limits `c_r^2=0.75` and
`c_tangential^2=0.1875`. The result remains selection only: the full dynamical
vector/lapse constraint algebra and nonspherical characteristic cone can still
retire it. See
[`SIGMA_V8B_CAUSAL_COMPLETION_SELECTION.md`](SIGMA_V8B_CAUSAL_COMPLETION_SELECTION.md).

The scalar cone result now also has a geometry-independent static bound. The
nonnegative-source equation limits the most negative Hessian eigenvalue at each
trace; maximizing the resulting directional speed reproduces the spherical
extremizer and never exceeds `c^2`. This closes the arbitrary static scalar
Hessian subgate, but not the dynamical vector/lapse or time-dependent gates.

### v8B covariant variation and FLRW clock subgate

The completion's exact scalar variation has two apparent third-derivative
principal terms that cancel, while its vector variation is algebraic in the
aether and adds no vector velocity. On aligned FLRW, the reduced operator has no
lapse velocity and no metric-scalar velocity mixing at `Q=Q0`. It does reduce
the scalar clock kinetic coefficient to
`2 K_2-3(alpha-1)L_H^2 H Q0`. The selected row is stable only when
`L_H^2 H Q0<12/7`, equivalently `L_H^2 H mu_sigma<24/7`.

This is an open stable region, not a full constraint proof. The exact metric
stress tensor, Noether identity, nonlinear Hamiltonian count, off-`Q0` mixing,
and time-dependent characteristic determinant remain mandatory. See
[`SIGMA_V8B_COVARIANT_VARIATION_GATE.md`](SIGMA_V8B_COVARIANT_VARIATION_GATE.md).

### v8B metric Euler tensor and conservation identity

Varying the connection and integrating by parts gives the completion's exact
symmetric metric Euler tensor. Its algebraic constant-jet part matches an
independent centered metric finite difference below `1e-9` relative error. The
scalar, vector, and metric Euler derivatives satisfy the off-shell
diffeomorphism identity, reducing to completion-stress conservation on the
scalar/vector equations. This completes the new operator's variation subgate,
not the combined theory's nonlinear constraint or characteristic gates. See
[`SIGMA_V8B_METRIC_NOETHER_GATE.md`](SIGMA_V8B_METRIC_NOETHER_GATE.md).

### v8B inherited constraints and homogeneous source uniqueness

The published nonlinear AeST Hamiltonian has four first-class and four
second-class constraints. Its 24-dimensional phase space therefore contains
six physical degrees of freedom. That count does not cover v8B: the same paper
states that Horndeski and more general higher-derivative extensions require a
new canonical analysis.

The base homogeneous shift equation integrates to `a^3 K_Q=I0`. A nonzero
`I0` supplies an arbitrary leading density `8 pi G rho=Q0 I0/a^3`; the source
paper explicitly notes that this density is not classically predicted. The
project now freezes `I0=0` as a boundary condition, not a sixth parameter, and
forbids using this dust-like state as missing gravity.

For the full v8B homogeneous clock, the exact current is
`I0/a^3=(Q-Q0)[4K2-3(alpha-1)L_H^2 H(3Q-Q0)]`. At zero charge, `Q=Q0` is the
only positive-clock branch; the other algebraic root has negative current
slope. This closes homogeneous stable-branch selection, but not arbitrary
inhomogeneous uniqueness or the combined Hamiltonian count. V8B remains held
before data. See
[`SIGMA_V8B_SOURCE_CONSTRAINT_GATE.md`](SIGMA_V8B_SOURCE_CONSTRAINT_GATE.md).

### v8B tilted-ADM necessary kinetic subgate

Allowing the aether to tilt relative to an ADM slice exposes a scalar normal
acceleration in the v8B completion. An exact antiderivative
`F=x(Q-Q0)^3/(3 chi)` removes it by a boundary subtraction and produces a
first-order metric--aether--scalar density. The identity residual is
`2.71e-20`, and the resulting ten-velocity automatic-differentiation Hessian
matches an independent centered finite difference to `5.37e-9` relative error.

Across 385 deterministic and 1,024 frozen random points with `v_A<=0.9`,
`0.5<=Q/Q0<=1.5`, `L_H Q0<=1`, and six decades in `a_sigma/Q0`, the base and
combined Legendre maps have no rank failure or inertia change. The minimum
combined singular value is `1.180292` and the minimum combined/base determinant
ratio is `0.922509`. This supports a conditional six-degree-of-freedom local
patch if the full diffeomorphism constraints survive; it is not a constraint
count or Hamiltonian-positivity proof.

A finite rank-changing surface occurs outside the envelope at `v_A=0.97` and
`Q/Q0=2.8649430865`, where the raw inertia changes from `(1,0,9)` to `(2,0,8)`.
The full Hamiltonian gate therefore remains false. V8B advances only to a full
inhomogeneous Dirac and reachability analysis: if the field equations do not
make this surface dynamically inaccessible, the candidate must be retired
before Solar or observational tests. See
[`SIGMA_V8B_TILTED_ADM_GATE.md`](SIGMA_V8B_TILTED_ADM_GATE.md).

### v8B global Legendre-rank failure

The bounded tilted patch does not extend to a globally regular action. At zero
background curvature, the completion's large-clock kinetic mixing has Schur
coefficient
`[(32-20 K_B)x^3+(32-67 K_B)x^2+(8-78 K_B)x-27 K_B]/(4 K_B)`.
For the frozen `K_B=1`, it becomes positive at finite aether velocity
`0.9020884486`; every nonzero completion length then reaches a finite rank-zero
surface. At `v_A=0.97`, `L_H Q0=1`, the first root is
`Q/Q0=2.8649430865`, with finite canonical energy and a mixed metric--aether
null mode. The raw inertia changes from `(1,0,9)` to `(2,0,8)`.

The apparent high-`K_B` escape also fails. On isotropic extrinsic curvature,
the determinant is exactly affine in `K` because the completion is linear in
curvature and nonlinear in the scalar clock. At ordinary `v_A=0.5` and
`Q/Q0=1.2`, finite roots occur at `K/Q0=2.8993, 2.8041, 1.8488, 1.1641,
0.9337` for `K_B=1, 1.6, 1.7, 1.8, 1.95`; every crossing has finite momenta and
adds the extra raw negative direction. The affine identity residual is below
`2e-15`.

Exact v8B is retired before data. `L_H=0` removes the proposed cluster-geometry
interaction, while removing only the causal partner restores v8A's
superluminal nonlinear cone. The next completion must be degenerate by
construction on arbitrary tilted time-dependent backgrounds; another
coefficient change is not authorized. See
[`SIGMA_V8B_GLOBAL_RANK_FALSIFICATION.md`](SIGMA_V8B_GLOBAL_RANK_FALSIFICATION.md).

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

## 2026-08-04 Sigma v13A exact clock-constraint falsification

The first post-v12 repair attempted to remove the inherited AeST clock/Jeans
sector exactly with

$$
\Delta L=\Lambda(U^\mu\nabla_\mu\phi-Q_0).
$$

The multiplier equation fixes the clock, but the shift-symmetric scalar
equation integrates to `a^3(J_base+Lambda)=I`. On the aligned clock the reduced
sector has `H=Q0 I`, `rho=Q0 I/a^3`, and zero pressure. The signed integration
charge makes the unrestricted Hamiltonian unbounded; restricting it positive
still leaves a dust-like gravitating state that is not predicted by baryons.

The minimal finite regularization
`Lambda deltaQ-chi Lambda^2/2` integrates out exactly to
`deltaQ^2/(2 chi)`, or `K2 -> K2+1/(4 chi)`. It therefore restores a soft
clock susceptibility instead of an exact constraint and provides no new
escape from the v12A `K2` screen. The analytic identities, dust redshifting,
and independent finite-difference stationarity check pass. Exact v13A is
retired without observations. This is total post-v12 material failure `2` and
the second bounded-Hamiltonian failure; the three-failure reset is not yet
triggered. See
[`SIGMA_V13A_CLOCK_CONSTRAINT_FALSIFICATION.md`](SIGMA_V13A_CLOCK_CONSTRAINT_FALSIFICATION.md).

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

## 2026-08-04 Sigma v16 static-boundary falsification and v17 gate

Sigma v16 tested whether the missing cluster Weyl field is an ordinary
finite-window boundary field generated by measured baryons outside the scored
strong-lensing region. The original 275 kpc result is integrity-invalid because
the convergence taper created a bright shear ring exactly at the scored edge.
The preregistered v16B replacement scores only the inner 200 kpc and begins the
taper 100 kpc outside it; padding factors 2--4 are stable.

The corrected harmonic oracle closes 61.94% of boundary-shear power in AS295
but only 37.29% in PLCK G287. Expanding the analytic exterior-potential basis
from maximum order 6 to 12 changes the PLCK result by only `0.000155`. In the
fair v16D nested control, measured outer component baryons improve symmetric
cross-cluster NRMSE from `0.817941` to `0.813831`, only `0.5025%` versus the
frozen 10% gate. Shear alignment and power closure fail in both transfer
directions. The tested instantaneous static local-plus-boundary family is
retired on the spent pair.

The next measurement-first question is frozen in
[`SIGMA_V16_BOUNDARY_FALSIFICATION_AND_V17_DYNAMICAL_GATE.md`](SIGMA_V16_BOUNDARY_FALSIFICATION_AND_V17_DYNAMICAL_GATE.md).
V17 tests an object-label-free invariant of baryonic spatial stress in the
Landau frame. Public Chandra data can support matched thermal-stress maps for
both spent clusters; PLCK G287 also has a public 639-row spectroscopic catalog,
whereas matched AS295 member velocities are not yet verified. Thermal stress
therefore runs first, and collisionless member stress cannot be selected on
PLCK alone. Any passing source must still be generated by a healthy covariant
action before a holdout is opened.

The PLCK catalog is now downloaded and hash-verified under the frozen
acquisition manifest: 153 selected members include 129 spectroscopic members,
and the full file contains 639 redshifts. A derived 129-row table records only
measured coordinates, F160W magnitude, redshift, and the standard
cluster-rest-frame line-of-sight velocity about the sample median. It commissions data
ingestion only; no spatial kernel, mass weighting, or gravity score is selected.

The AS295 collisionless fallback has now been audited against both cited public
spectroscopic releases. Ruel (2014) contains 39 spectra and reports 30 members;
Bayliss (2016) contains 38 spectra and reports 29. All 38 Bayliss spectra match
Ruel within the frozen 1-arcsec radius, so the union contains 39 spectra and 30
fixed-window members rather than an additive 77 spectra or 59 members. The
frozen 50-member stage-B gate therefore remains closed by a 20-member shortfall;
the threshold is not lowered and PLCKG287 is not analyzed alone. Exact Sifon
2013 cluster/member queries and a 9-arcmin Sifon 2016 cone return no AS295 rows,
so the obvious independent ACT releases do not close the gap. The public MGCLS
crossmatch contains 4,995 AS295 objects but only photometric `zPhot`, so it adds
no measured velocities and is explicitly excluded from the member-stress
count. See
[`SIGMA_V18A_COLLISIONLESS_STRESS_READINESS.md`](SIGMA_V18A_COLLISIONLESS_STRESS_READINESS.md).

## 2026-08-05 V19X3 uncertainty completion and V19X4 gas-state correction

The full 494-region V19X3 executor now records an independent 68% Sherpa
profile-likelihood interval for APEC normalization in addition to temperature.
An ordered normalization interval is part of an individual region's quality
gate because normalization sets emission measure and gas mass. A finite
best-fit region that fails this subgate is still retained, consistent with the
frozen minimum of 12 complete quality passes per cluster.

The pre-result V19X4 gas-state audit found a material algebra error in the
hash-bound V19H prose. For $R=n_e/n_H$, the uniform-slab identities are

$$
n_e=\sqrt{R E_A/L},\qquad
\Sigma_{\rm gas}=\mu_e m_p\sqrt{R E_A L}.
$$

V19H had $R$ in the denominator. The historical file remains untouched; V19X4
freezes the correction prospectively. Corrected surface densities are exactly
1.2 times the historical expression. The executable audit admits all 366
Bullet and 128 Abell 2146 accepted region geometries, validates the official
APEC normalization and Rankine-Hugoniot shock identities, and leaves regional
spectra, lensing, halos, invariant selection and gravity fitting sealed.

The project priority is now explicit in
[`SIGMA_BROAD_PHENOMENOLOGY_ROADMAP.md`](SIGMA_BROAD_PHENOMENOLOGY_ROADMAP.md):
stratified galaxy morphology, raw cluster topology, measured component
geometry, merger offsets and joint lensing/dynamics come before expensive
Solar-System optimization. Solar and PPN consistency remain mandatory later
vetoes. Cosmological dark-matter observables require perturbing a final
covariant action and cannot be predicted uniquely from the current empirical
bridge.

## 2026-08-05 V19X4 executable posterior and common-grid admission

The corrected gas algebra is now implemented as a hash-frozen future executor,
not only a prose protocol. For each cluster it will generate 4,096
scrambled-Sobol draws for every accepted region under three predeclared
temperature-normalization dependence branches ($\rho=-0.9,0,+0.9$). Failed
ordered intervals use the complete pre-fit log bounds and remain flagged;
regions are never outcome-selected.

All source summaries are placed on identical 241-by-241, 10-kpc physical grids
and exposed at 50-kpc and 100-kpc FWHM. Surface-density smoothing is explicitly
mass conserving. A manufactured run against the real bin maps proves that all
366 Bullet and 128 Abell 2146 admitted region IDs survive resampling. The
executor verifies the future V19X3 config/report hashes and refuses to run
before all 494 finite fits and both cluster-level quality gates pass.

This closes a major pre-lensing implementation gap: the density, overlap,
thermodynamic-gradient and baroclinicity candidates can later be compared on
the same axes and physical resolutions. No regional spectral value, invariant
score, lensing/halo target, action or gravity constant was opened or selected.

## 2026-08-05 V19BK source-observability admission

The pre-result observability audit now restricts the V19BJ source library to
quantities the registered two-cluster data can identify. I4
thermodynamic-gradient stress remains eligible as a projected 2D tensor, and
I5 baroclinicity remains eligible as a scalar activation. I1 component
overlap, I2 relative current, I3 full anisotropic stress and I6 causal
relaxation are withheld: the member maps have only cluster-relative
single-filter light and line-of-sight velocity moments, the gas data have no
validated velocity vector, and there is only one thermodynamic snapshot.

The density null is correspondingly stricter. At the adaptive-region level, a
fixed quadratic nuisance model controls for physical gas surface density,
within-cluster normalized stellar-light morphology, gas-density gradient, and
gas-density Hessian invariants. Analytic leave-one-region-out PRESS must still
leave at least 20% of the candidate variance unexplained. This avoids calling
a differentiated density map new physics and forbids a spurious absolute
mass comparison between Bullet Bessel-I and Abell 2146 F814W light.

The automatic shock-front route remains paused after three pre-science
implementations failed their registered validation. No fourth detector will be
tuned while I4/I5 are scored. This audit is not a gravity result: a future
source pass would authorize a covariant derivation, after which raw lensing and
the broader galaxy/cluster gates can be opened under one frozen equation.

## 2026-08-05 V19BL invariant-scoring math freeze

The exact source-only scoring algebra is now executable and hash frozen. I4 is
represented by the two spin-two components of the dimensionless projected
thermodynamic-gradient tensor, while I5 is the bounded squared sine of the
projected density-pressure gradient angle. Manufactured polynomial,
parallel/perpendicular-gradient, axial-wrap and coordinate-rotation checks all
pass. The earlier V19BJ functions and tests remain intact.

A region enters only when both gradients required by its candidate have at
least three-sigma two-component Mahalanobis support. At least 32 regions must
survive. The fixed 21-term density nuisance basis is evaluated with analytic
leave-one-region-out PRESS; I4 uses a joint two-component residual fraction so
rotating the map cannot change the novelty verdict. At least 20% unexplained
variance must remain in 90% of posterior draws.

Both clusters, all three temperature-normalization dependence branches, 50-
and 100-kpc smoothing, and 250/350/500-kpc apertures must pass. Activation must
remain within 10%, I4 direction within 10 degrees, its 95% axial interval
within 30 degrees, and at least 90% of region omissions and projection draws
must remain stable. These are source-identifiability tests only; observed V19X4
gas maps, lensing, action selection and gravity parameters remain unopened.

## 2026-08-05 V19BM stellar-control executor

The filter-safe stellar morphology nuisance is now implemented as a future
hash-frozen executor. It consumes exactly member-ensemble sample IDs 0--4095,
normalizes the finite member light within each cluster and draw, inverts the
exact V19X4 native-pixel convention, deposits light on the 241-by-241 physical
grid with cloud-in-cell weights, and smooths independently at 50 and 100 kpc.
Every draw is rescaled after convolution to conserve unit light.

The resulting adaptive-region means are converted to within-draw percentile
ranks. Those ranks, not the relative luminosity amplitudes, enter the V19BL
density null. This prevents Bullet Bessel-I and Abell 2146 F814W photometry
from being treated as a common stellar-mass scale. The preflight passes, but
terminal execution correctly remains closed until the observed V19X4 report
provides three identical, hash-bound label grids per cluster. No gas posterior,
lensing, halo, action or gravity target was opened.

## 2026-08-05 V19BN source-score decision engine

The posterior decision engine is now commissioned independently of the still-
running response production. It enforces multi-gradient support, calculates
I4 amplitude/axis and I5 activation per draw, runs the fixed quadratic PRESS
novelty control, combines all resolution/aperture variants at draw level, and
tests each posterior-median region omission.

Manufactured data verify both sides of the decision: a response algebraically
constructed from density controls is rejected in every draw, independent
structure passes, a 2% transfer perturbation is admitted, a 45-degree I4-axis
rotation is rejected, and a uniform tensor survives every region omission.
This is still a preflight; the gas-map-to-region executor and observed score
remain gated on terminal V19X4/V19BM products, with lensing sealed.

## 2026-08-05 V19BO gas-source streaming layer

The remaining gas-map-to-region mathematics is now implemented and frozen.
Regional V19X4 draws are mapped and smoothed in bounded batches, differentiated
into all I4/I5 and gas-control quantities, reduced inside each registered
aperture, and concatenated only after their schemas agree. Full common-grid
draw stacks are never retained.

Manufactured execution produces all 14 quantities for every scale/aperture
variant, conserves gas surface density per draw to $10^{-12}$, keeps finite I5
values inside $[0,1]$, and leaves regions outside an aperture invalid so they
cannot enter gradient support. The remaining integration task is wiring the
terminal V19X4 and V19BM product manifests into this stream and the V19BN
decision engine; source outcomes and lensing remain unopened.

## 2026-08-05 V19BT blind-cluster source readiness

The balanced V19BH future-cluster shortlist now has an executable source-only
acquisition boundary. Six of eight systems have direct public SGAS HST F160W
images plus published Chandra observations with more than 1,000 counts inside
R500. They divide evenly into three relaxed-side and three disturbed-side
systems and span a factor 7.22 in nominal M500.

J1002+2031 remains a reserve because its published strong-lens evidence is not
well constrained and it lacks a direct source-only SGAS HLSP manifest.
J1226+2149 remains a reserve because its published BCG/ICL analysis uses only
F606W and because the projected J1226 pair needs separate component
deprojection. No final six are selected. Every system still lacks at least an
independent member-probability catalog, full stellar/ICL uncertainty and a gas
line-of-sight posterior, so the strict complete-baryon count remains zero.

Only HST `/images/v1/` products and published Chandra observation identifiers
are whitelisted. Lens maps, image coordinates, topology and residuals remain
sealed, and no formula or constant is selected. See
[`SIGMA_V19BT_BLIND_CLUSTER_SOURCE_READINESS.md`](SIGMA_V19BT_BLIND_CLUSTER_SOURCE_READINESS.md).

## 2026-08-05 V19BU WALLABY source-only candidate universe

The blind galaxy lane now has an actual, hash-bound WALLABY DR1 source catalog
rather than only a survey-level protocol. A strict CASDA ADQL projection
retrieved 711 source-finding rows representing the published 592 unique H I
detections. The 21 retained columns contain identity/provenance, sky position,
integrated H I source measurements, noise and quality metadata, moment-zero
geometry, distance and H I mass.

The separate kinematic table was inspected only at the schema-name level to
freeze a deny-list. No row containing systemic velocity, fitted inclination,
kinematic position angle, radial grid, rotation speed, velocity field,
residual or halo result was read. The spectral cubes also remain sealed because
they contain the target velocities. No final galaxy was selected, and no
action, constant or Solar-System tuning changed. See
[`SIGMA_V19BU_WALLABY_SOURCE_ONLY_METADATA.md`](SIGMA_V19BU_WALLABY_SOURCE_ONLY_METADATA.md).

## 2026-08-05 V19BV WALLABY release-row robustness

The 711-row WALLABY source catalog now has a deterministic 592-name canonical
view, but the release choice is not treated as certain. All 119 repeated names
are Hydra TR1/TR2 pairs. Five prespecified source-quality priority orders agree
for only 27 pairs; 92 select different rows under at least one reasonable
ordering. The first implementation's use of non-unique `catalogue_id` values
would have hidden this ambiguity, so the final audit identifies rows by the
archive primary `id`.

Every alternative remains in the immutable V19BU input. The 92 sensitive
pairs must propagate both source reconstructions or be removed by a target-
blind gate when the galaxy sample is frozen. No kinematic value, target
residual, gravity formula, constant or Solar result was opened. See
[`SIGMA_V19BV_WALLABY_CANONICAL_SOURCE_ROWS.md`](SIGMA_V19BV_WALLABY_CANONICAL_SOURCE_ROWS.md).

## 2026-08-05 V19BW WALLABY source-only variety frame

The blind galaxy lane now has an explicit broad-coverage frame before any
rotation speed or velocity field is opened. The 109 names with the published
successful-kinematic-product availability flag divide into 27 or 28 objects
in every quartile of H I mass, relative H I compactness, source axis ratio,
distance and source extent. They span Hydra, Norma and NGC 4636 with counts of
35, 31 and 43, and occupy 95 distinct five-axis cells; no cell contains more
than three names.

This is source-side coverage rather than a final evidence split. Optical
surface brightness, bulge/bar structure, stellar mass, environment and full
3D gas geometry remain missing. The release-row audit is propagated: 103
names keep kinematic availability under all five prespecified policies, six
under only some, and 78 names change at least one source-metric quartile.
No target value, formula score, action, constant, holdout label or Solar
optimization was opened. See
[`SIGMA_V19BW_WALLABY_SOURCE_ONLY_VARIETY_FRAME.md`](SIGMA_V19BW_WALLABY_SOURCE_ONLY_VARIETY_FRAME.md).

## 2026-08-05 V19BX SkyMapper optical-candidate contract

The next blind galaxy-source acquisition is frozen before retrieving any
SkyMapper row. Every one of the 592 V19BW H I centroids will receive the same
60-arcsecond DR4 cone query and exact 50-column source-only projection. All
returned objects are retained; nearest-neighbor or galaxy-like appearance is
not allowed to select an optical counterpart.

The uniform SkyMapper footprint covers Hydra, Norma and NGC 4636 and supplies
photometry, Petrosian radius, catalog quality and stellarity diagnostics. Its
published extended-source limitation is explicit, so these rows can measure
coverage and crowding but cannot yet define bulge fractions or stellar masses.
The query projection was checked against all 121 live `dr4.master` fields, and
all 50 declared columns exist. Five manufactured contract tests pass. No
WALLABY kinematic row, evidence split, force result, action, constant or Solar
calculation was opened. See
[`SIGMA_V19BX_SKYMAPPER_SOURCE_ONLY_CANDIDATES.md`](SIGMA_V19BX_SKYMAPPER_SOURCE_ONLY_CANDIDATES.md).

V19BX subsequently executed without changing its frozen query. All 592 cones
passed, yielding 17,094 candidate rows. Hydra contributes 3,906 candidates,
NGC 4636 contributes 1,417 and Norma contributes 11,771. The 109-name
kinematic-availability lane alone contains 3,616 candidates, including 2,459
around the 31 Norma sources. This is direct evidence that a nearest optical
neighbor cannot be treated as the counterpart: the next source-only step must
combine H I footprints, optical cutouts and star masks under a frozen
probabilistic association model. No counterpart or target was selected.

## 2026-08-05 V19BY WALLABY moment-zero map contract

The H I spatial-footprint acquisition is now frozen before any image download.
It requires one public two-dimensional moment-zero FITS map for all 711 V19BU
release rows, including every Hydra TR1/TR2 alternative. Only four declared
DR1 `source_data_*` planes and `_mom0.fits` artifacts are eligible.

Cubes, channel masks, moment-1/2 maps, spectra and kinematic planes are
explicitly rejected. Every admitted file must reproduce the CADC byte length
and MD5 and have `NAXIS=2` with no third coordinate axis. Five contract tests
pass. The maps will support later H I/optical spatial association, but this
checkpoint cannot choose a counterpart, evidence split, action or constant.
See
[`SIGMA_V19BY_WALLABY_MOMENT0_SOURCE_MAPS.md`](SIGMA_V19BY_WALLABY_MOMENT0_SOURCE_MAPS.md).

V19BY subsequently passed the unchanged contract. The archive supplied all
711 expected maps (10,200,960 bytes): 148 Hydra TR1, 272 Hydra TR2, 147 NGC
4636 TR1 and 144 Norma TR1. Every byte length and MD5 matched, every FITS file
was two-dimensional with no spectral axis, and no product was missing,
ambiguous or failed. The immutable manifest has SHA-256
`871df6aa9db724ad648a08762d619884f326d643c86ecd97414b79d4a2ae7aa7`.
All release alternatives remain available for later uncertainty propagation;
no counterpart, kinematic target, evidence split or force result was opened.

## 2026-08-05 V19BZ H I/optical spatial-information audit

The next blind-galaxy step is explicitly exploratory rather than retrospectively
called preregistered. A source-only inspection showed that moment-zero overlap
may not distinguish the dense SkyMapper candidate fields. V19BZ therefore
freezes a reproducible information audit across four beam-kernel widths while
retaining every candidate and every release alternative.

The audit uses no optical weight, counterpart prior, null posterior, hard
assignment or sample removal. It keeps every velocity, rotation, lensing,
halo, gravity, holdout and Solar-System payload sealed. A failure can authorize
better optical source information and uncertainty propagation, but cannot
reject or modify a gravity theory. See
[`SIGMA_V19BZ_HI_OPTICAL_SPATIAL_INFORMATION_AUDIT.md`](SIGMA_V19BZ_HI_OPTICAL_SPATIAL_INFORMATION_AUDIT.md).

The complete V19BZ audit passed all seven access and reproducibility gates but
found the spatial information insufficient for a hard counterpart. The
one-beam top-to-second margin is only 1.059 at the median. Twenty-four of 711
release maps reach 3:1 in that branch, but just three (0.42%) keep both the
same top object and a 3:1 margin across all four beam kernels. Hydra has zero
robust maps, NGC 4636 has three and Norma has zero. Only 82 of 119 duplicate
Hydra names retain one top identity across both releases and every kernel.

All 18,550 candidate/release pairs remain available. The result forbids a
convenient nearest or maximum-overlap assignment and requires uniform optical
images, foreground-star masks, deblending uncertainty and probabilistic
mixture propagation before this blind galaxy lane can support morphology
holdouts. No target or force result was opened.

## 2026-08-05 V19CA SkyMapper/Gaia foreground contract

The blind galaxy lane now has a frozen path to foreground-star evidence that
does not require mass retrieval from the SkyMapper image-cutout service. The
DR4 TAP database publishes exact Gaia DR3 identifiers for SkyMapper objects,
so V19CA will query all 17,034 unique candidates by object ID and retain all
17,094 candidate occurrences.

A one-arcsecond match plus five-sigma positive parallax or component proper
motion defines astrometric foreground evidence. A stricter diagnostic also
requires `RUWE <= 1.4` and a five- or six-parameter solution. Neither flag can
remove or weight a candidate. The schema and three source-only pilot rows were
inspected before freezing and are disclosed; the full population and all
kinematic/gravity targets remain unopened. See
[`SIGMA_V19CA_SKYMAPPER_GAIA_FOREGROUND_DIAGNOSTICS.md`](SIGMA_V19CA_SKYMAPPER_GAIA_FOREGROUND_DIAGNOSTICS.md).

V19CA subsequently returned all 17,034 unique SkyMapper objects in 43 exact-ID
batches with no missing or duplicate row. There are 13,958 exact Gaia matches,
12,801 objects with five-sigma foreground astrometry and 12,347 with the
stricter quality-controlled contamination flag. The fraction is sharply
field-dependent: 1,830/3,846 in Hydra, 304/1,417 in NGC 4636 and
10,213/11,771 in Norma.

This directly explains much of Norma's V19BZ crowding without pretending to
resolve it. All candidates remain represented because a moving foreground star
can overlap a background galaxy. Any future mask must be supported by optical
image and deblending evidence, and foreground-treatment uncertainty must be
propagated. No kinematic or force target was opened.

## 2026-08-05 V19CB foreground-treatment information audit

The complete Gaia source result has now been inspected, so V19CB is explicitly
post-source exploration. It measures whether retaining, softly suppressing or
counterfactually masking astrometric foreground objects makes the H I/optical
ranking robust across all four beam kernels. Zero-weight branches are not
authorized masks, and no treatment or counterpart is selected.

All 711 release rows appear in all four branches. The audit remains independent
of WALLABY velocities, lensing, halo maps, gravity residuals, evidence splits,
actions, constants and Solar-System results. See
[`SIGMA_V19CB_FOREGROUND_TREATMENT_INFORMATION_AUDIT.md`](SIGMA_V19CB_FOREGROUND_TREATMENT_INFORMATION_AUDIT.md).

The complete V19CB audit passed all six gates but did not resolve optical
association. Kernel-stable 3:1 cases rise from 3/711 when all candidates are
retained to 34/711 under a 0.1 foreground weight, 35/711 under a
quality-controlled diagnostic mask and 41/711 under the most aggressive
five-sigma diagnostic mask. That best branch is only 5.8% overall and 3/144 in
Norma; one Hydra release map is left with no positive candidate.

Gaia astrometry is therefore a useful foreground uncertainty layer, not a
counterpart solution. Optical pixels and deblending remain required, and none
of the exploratory treatments is selected for the future galaxy holdout.
