# Premise-level rethink after the three completion routes

Status: next active research stage after H7/EV-SVT, EA-Q0, and EMOG-Q0 failed
their declared identifiability or pre-fit structural gates.

## Why this is not another theory-variant cycle

The measured/profile-completed BCG result still supports full baryonic potential
as a useful empirical coordinate. It does not establish that a local scalar value
should control a universal interpolation law. Three different relativistic routes
have now exposed three different obstructions:

1. H7s improves the joint development score but needs its amplitude at the hard
   bound, so the proposed closure is not identified.
2. EA-Q0 makes the environment field dynamical, but the reciprocal action source
   overwhelms the independently measured field.
3. EMOG-Q0 keeps a consistent scalar source and healthy Proca vector, but one
   universal Yukawa range has the wrong radial breadth, the scalar response has
   the wrong cross-system ordering, and a long-range universal vector cannot make
   Solar-System dynamics and lensing agree.

Adding another activation, widening an amplitude, or assigning parameters by
object class would conceal rather than resolve these failures. The next stage
therefore audits the target and the one-field premise before selecting any action.

## Premises that must be separated

### P1. Raw observables versus derived accelerations

SPARC supplies rotation speeds, which are direct dynamical observables after the
usual distance and inclination calibration. The frozen CLASH table supplies a
lensing-derived total-acceleration reconstruction. That reconstruction is useful
for empirical comparison, but a relativistic alternative should ultimately
forward-model shear, convergence, or a published lensing mass likelihood rather
than treat a GR-interpreted acceleration as theory-independent data.

The first task is to trace every SPARC, CLASH, and BCG target back to its raw
observable and list the gravitational assumptions in the transformation. No new
theory is scored until this provenance matrix is complete.

### P2. One response field versus two metric potentials

Nonrelativistic dynamics primarily constrains the time potential $\Phi$ (plus any
direct nonmetric force). Lensing constrains the Weyl combination $\Phi+\Psi$.
Treating both as one scalar acceleration silently assumes a gravitational slip and
matter-coupling structure before deriving an action.

The rethink will reconstruct the required dynamical and lensing responses
separately. A future action must predict both from the same baryonic source and
boundary data; it will not be allowed to call the two reconstructions separate
normalizations.

### P3. Pointwise 5% structure versus measurement precision

The 5% pre-fit gate was intentionally severe and useful for rejecting structural
identities. It is not an uncertainty model. Some SPARC points, BCG proxies, and
lensing-derived accelerations have uncertainties larger than 5%, and neighboring
radial points have shared systematics. The next stage must preserve the 5% gate as
a deterministic shape diagnostic while adding a covariance-aware observable-level
gate. Passing either one cannot be reported as passing the other.

### P4. Local environment scalar versus nonlocal boundary-value problem

The successful BCG environment coordinate is an integrated baryonic potential,
including exterior shells. A local algebraic field value or local density minimum
does not automatically reproduce that quantity. The next theory class, if any, is
considered only after establishing whether the data require a nonlocal Green
function, a second metric potential, or merely better baryonic/profile inference.

## Concrete next checkpoints

### R0 - observable provenance matrix

Deliver one machine-readable row per scored column with source publication,
catalog/file, raw observable, transformations, assumed metric law, covariance
availability, and whether the value can be forward-modeled outside GR.

Advance only if raw or likelihood-level observables are available for all 20 CLASH
systems and for at least 30 of the frozen BCG hosts. Otherwise identify a public
replacement sample before theory work resumes.

### R1 — two-potential identifiability sample

Find systems with both resolved dynamics and resolved strong/weak lensing plus a
measured baryonic profile. The minimum useful pilot is 10 systems with at least
three independent radial constraints on each observable. Freeze the sample using
data quality and coverage only, never a residual.

Outcome: determine whether $\Phi$ and $\Phi+\Psi$ can be separately reconstructed
over overlapping radii. If not, the claimed dynamics-lensing unification target is
not yet empirically identifiable.

### R2 - theory-free response test

Using only baryonic profiles and boundary data as inputs, cross-validate the
smallest nonlocal/two-potential response representation. This is a diagnostic of
the target's dimensionality, not a proposed force law.

Concrete outcomes:

- retain the one-field premise only if one latent response predicts both raw
  dynamics and lensing without a class label and closes at least 50% of each
  domain's held-out benchmark gap;
- move to a two-potential premise if two independently required responses are
  identified on the same systems;
- stop the unification claim if neither response is identifiable with the
  available covariance and radial overlap.

### R3 - action selection only after R0-R2

Only a passed identifiability result may start another covariant action cycle. Its
field count must match the empirically required response dimension. The action must
again precede fitting and retain the same prohibitions on per-object, class-only,
and lensing-only parameters.

## Immediate research question

The next concrete task is R0: verify what raw CLASH lensing likelihoods or shear/
convergence profiles and what same-object BCG dynamics+lensing products are
publicly available. This determines whether the existing derived-acceleration table
can support the claimed relativistic test or must be replaced before more theory is
written.

## Current R0 checkpoint

The provenance matrix for all columns currently entering the SPARC, CLASH and BCG
scores is now frozen in `configs/r0_observable_audit.json` and generated as
`data/derived/r0_observable_provenance.csv`. The coverage audit finds 0/10 eligible
same-object systems. CLASH provides 3-5 GR+NFW-deprojected summary points per
cluster but no independent dynamics; the 34 frozen MaNGA/SPIDERS BCGs provide one
dynamical summary or proxy each and no resolved lensing. Therefore R1 and R2 are
not authorized on the current sample. See `docs/R0_OBSERVABLE_AUDIT.md`.

## Current replacement-sample checkpoint

The residual-blind replacement search now has a ten-system acquisition queue from
the Newman (2013) and Kaleidoscope (2025) cluster samples after deduplication and
after excluding A963 from the preliminary 3+3 count screen. The public article
sources provide 70 resolved BCG velocity-dispersion bins and many strong-lens
image constraints. However, the numerical lens-model chains/covariance, numerical
baryonic profiles, and exact dynamics-lensing radial overlap are not yet local and
verified. Consequently the published-count screen is 10/10, while strict R1
eligibility remains 0/10. See `docs/R1_REPLACEMENT_SAMPLE_INVENTORY.md`.

The subsequent RELICS audit recovered 100 Lenstool MCMC convergence maps for each
of A2537, MACS J0417, and MACS J0949. Radial covariance is now reconstructible for
those three projected lensing profiles. On the current FITS-reference-centered
grids, A2537 has three full lensing annuli inside the published stellar-kinematic
support; the two MACS ensembles have only one each at their archived range-map
resolution. Exact BCG-centered overlap remains unverified, so this is a
provisional radial count rather than a gate pass.

The Newman source also provides Chabrier SPS mass-to-light normalizations. Seven
BCG stellar components have now been reconstructed as numerical dPIE surface-mass
profiles on their dynamics grids, with a conditional covariance from the stated
photometric and cut-radius uncertainties. This is not a complete baryonic model:
M/L uncertainty, gas, extended ICL, satellites, and cross-probe covariance remain
missing. A2537 is therefore an engineering pilot only (and the source paper flags
it as the sample's most likely disturbed cluster), not a frozen science system.

Current machine-readable R1 state:

- published-count candidates: 10/10;
- local normalized BCG stellar components: 7/10;
- local standard-lensing MCMC map ensembles: 3/10;
- verified BCG-centered 3+3 overlap: 0/10;
- complete R1 systems: 0/10.

The next advancement condition is unchanged: no theory-free latent response is
fit until at least ten systems have complete baryonic inputs, observable-level or
re-runnable lens likelihoods with covariance, and three BCG-centered overlapping
radial constraints for both dynamics and lensing.
