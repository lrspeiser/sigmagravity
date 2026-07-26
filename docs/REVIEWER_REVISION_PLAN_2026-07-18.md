# Reviewer Revision Plan (Frozen Before the Derivation Sprint)

**Manuscript:** *Σ-Gravity: Coherence-Dependent Gravitational Enhancement in Galaxies and Clusters*

**Manuscript ID:** 1866133

**Frozen:** 2026-07-18

This document preserves the planned response to Reviewers 1 and 2 before any
manuscript revision. The derivation sprint may change which claims remain
defensible, but it must not silently rewrite this record. Sprint findings will
be reported separately in
[`DERIVATION_SPRINT_REPORT_2026-07-18.md`](DERIVATION_SPRINT_REPORT_2026-07-18.md).

## Revision stance

Recast Σ-Gravity as a tightly specified, nonrelativistic phenomenological
model unless and until an action-based derivation and independent coherence
test succeed. Separate calibration, validation, and exploratory evidence in
the abstract, results, tables, figures, and conclusions. Do not claim a
relativistic theory, derived lensing law, PPN constraint, or cosmological
viability without the corresponding calculation.

## Changes addressing both reviewers

1. **Define one canonical model.** State the exact equations used by the
   production analysis and distinguish them from exploratory alternatives in
   the repository. Reconcile the manuscript, README, and code.
2. **Clarify theoretical status.** Replace claims of a fundamental or
   covariant theory with an explicit statement that the submitted model is a
   nonrelativistic phenomenology. Explain which conservation properties follow
   from an action and which were previously assumed.
3. **Make coherence operational.** Define coherence from independently
   measured phase-space moments, never from the velocity being predicted.
   State the reference frame and the assumptions under which
   \(v_{\rm rot}^2/(v_{\rm rot}^2+\sigma^2)\) is meaningful.
4. **Make path length operational or remove it.** Replace system-class values
   such as \(L=600\) kpc with a reproducible functional of the baryonic
   distribution. If no such definition survives held-out tests, present the
   cluster amplitude as empirical and drop the path-length derivation.
5. **Separate cluster calibration from validation.** Label the Fox et al.
   sample as calibration. Validate without refitting on a separate cluster
   sample with measured radial baryonic and lensing profiles, propagating
   covariance and baryonic systematics.
6. **Replace the counterrotation comparison.** Construct matched controls and
   use observed velocity and dispersion maps as the primary outcome. Treat
   Newtonian/JAM-derived dark-matter fractions only as a secondary diagnostic.
7. **Upgrade statistical reporting.** Use galaxy- or cluster-grouped splits,
   uncertainty propagation, effect sizes, confidence or posterior intervals,
   covariate-balance diagnostics, multiplicity disclosure, and sensitivity
   analyses. If no external statistician is retained, state explicitly that
   the consultation request remains unfulfilled and provide complete code,
   grouped residuals, random seeds, and sensitivity outputs for independent
   audit rather than implying that reproducibility substitutes for consultation.
8. **Moderate all conclusions.** Replace “confirmed prediction,” “successfully
   predicts,” and “supports the framework's validity” with language that
   distinguishes calibration, consistency, correlation, and independent
   prediction.

## Reviewer 1-specific work

- Demonstrate the reduction of any proposed invariant coherence expression to
  the practical kinematic estimator, or withdraw the covariance claim.
- Quantify the error of the algebraic relation relative to an exact QUMOND
  field solve for nonspherical disks.
- Load the published SPARC photometric scale lengths rather than estimating a
  radius from the rotation-curve sampling. Remove the scale length from the
  canonical equation if the canonical code does not use it.
- Replace \(M_b(<200\,{\rm kpc})=0.4f_bM_{500}\) with measured gas, BCG, and
  satellite stellar profiles. Propagate their uncertainties and show the
  sensitivity to every remaining approximation.
- Give a unique three-dimensional prescription for any retained \(L\), with
  worked examples for disks, spheroids, and clusters.
- Reframe the Solar-System section as a nonrelativistic anomalous-acceleration
  check. Do not identify the enhancement factor with PPN \(\gamma-1\).
- State that zero gravitational slip and equality of lensing and dynamical mass
  are closure assumptions until derived from a relativistic action.
- Repair figures and tables: legible labels, uncertainty bands, calibration
  markers, sample counts, residual panels, and consistent values in text,
  captions, and tables.

## Reviewer 2-specific work

- Add a parameter-identifiability and sensitivity section for
  \(A_0,L_0,n,g^\dagger\), including their degeneracies and the fact that a
  single assigned cluster length identifies only one effective amplitude.
- Describe the 47% versus 53% SPARC comparison as comparable performance, not a
  win. Report uncertainty on the difference and use common baryonic inputs.
- Do not present the calibrated Σ-Gravity versus fixed MOND cluster comparison
  as two independent predictions.
- State clearly that the model is not a viable cosmology at present. Discuss
  CMB, primordial abundance, and structure-formation requirements as unsolved
  research questions rather than minor limitations.
- Explain why a dark-matter-model-derived quantity cannot by itself confirm a
  modified-gravity mechanism.

## Planned reviewer-response matrix

| Concern | Manuscript action | Required evidence | Decision rule |
|---|---|---|---|
| No theoretical foundation | Add action derivation only if the sprint verifies the Euler–Lagrange equations; otherwise reframe as phenomenology | Symbolic derivation, dimensional audit, conservation statement | No “theory” or covariance claim without a verified action |
| Circular cluster result | Relabel Fox as calibration and add a no-refit external test | Frozen parameters, disjoint cluster IDs, radial residuals, covariance | Independent result must be reported even if it fails |
| Simplified cluster baryons | Replace universal fraction/concentration estimate | Gas and stellar profiles with provenance | No main cluster claim from the 0.4 factor |
| Undefined path length | Supply an observable functional or remove \(A(L)\) | Cross-system and within-system held-out tests | No assigned morphology constants |
| Counterrotation confounding | Match controls and forward-model IFU maps | Balance table, direct kinematic likelihood, grouped CV | No “confirmation” from unmatched \(f_{\rm DM}\) |
| Algebraic field approximation | Compare with numerical QUMOND | Geometry-spanning convergence study | Use exact solver if error is material |
| Solar-System and lensing claims | Label closure assumptions and limits | Nonrelativistic acceleration calculation; relativistic work deferred | No PPN/slip assertion from a Poisson model |
| Overstated statistics | Rewrite results and conclusions | Effect sizes, intervals, sensitivity, preregistered outcomes | “Comparable,” “calibrated,” or “exploratory” as appropriate |
| Cosmological context | Add a dedicated limitations subsection | Requirements from CMB, BBN, growth, and GW propagation | Explicitly outside the model's validated domain |

## Deliverables before resubmission

- Revised manuscript and supplementary information with a single canonical
  equation set.
- Point-by-point replies to both reviewers, including negative results.
- Reproducible code and frozen data manifests for every revised figure/table.
- Independent cluster validation and direct-kinematics counterrotation result,
  or an explicit statement that these tests were not passed.
- Statistical review of the final analysis.

## Freeze rule

No manuscript or production-regression changes are authorized by this document
until the bounded derivation sprint records its go/no-go verdict. Research code,
tests, manifests, and audit reports may be added in an isolated namespace.
