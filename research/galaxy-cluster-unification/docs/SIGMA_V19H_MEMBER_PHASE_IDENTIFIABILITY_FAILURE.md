# Sigma v19H member-phase identifiability failure

## Outcome

The frozen heteroscedastic member phase-space gate failed for both causal
development clusters.  The target-blind BIC rule selected one component rather
than the required two in both the Bullet Cluster and Abell 2146.

| Cluster | BIC, one component | BIC, two components | BIC, three components | Selected |
|---|---:|---:|---:|---:|
| Bullet Cluster | 539.713 | 546.766 | 558.292 | 1 |
| Abell 2146 | 384.038 | 400.327 | 418.176 | 1 |

Lower BIC is preferred.  Adding a component was permitted only if it improved
BIC by at least 10 relative to the selected simpler model.  Neither catalog
even lowered BIC at two components.

This was not a numerical failure.  All six primary optimizations were finite,
the selected fits converged, and all 2,000 catalog/velocity-error bootstraps per
cluster converged with zero failures.  A synthetic two-component fixture is
correctly recovered by the same implementation.

Abell 2146's published A/B labels were retained only for a post-selection
check.  They were not used in fitting or model selection; because the selected
fit has one component, the adjusted Rand index is zero.  The Bullet catalog
has no comparable row labels.

## Meaning

The available positions and line-of-sight velocities do not independently
identify two discrete Gaussian merger components under the preregistered
model.  We therefore cannot use fitted component centroids as a model-free
launch point for the causal shock clock.  Forcing two components from the
published narrative would turn an external assumption into the answer.

This is a source-identifiability failure, not a test of a gravity equation and
not evidence against ordinary merger physics.  No registered image was viewed,
no lensing target was opened, and no gravity parameter was changed.

## Consequence and next defensible branch

V19H cannot satisfy its full advance gate as written.  The next source-side
branch must be frozen as a materially different measurement rather than a
threshold relaxation.  The defensible alternative is a continuous member
density/current field:

1. use all catalog rows without assigning discrete subclusters;
2. select one spatial bandwidth by source-only likelihood cross-validation;
3. estimate density, local mean line-of-sight velocity and velocity dispersion
   with quoted measurement errors propagated;
4. identify spatial modes by topological persistence under catalog bootstrap;
5. use the continuous field directly for causal-history and frame-dragging
   diagnostics; and
6. require transfer without a cluster label, forced component count or lensing
   information.

This alternative can fail too.  It must be preregistered before its bandwidth,
modes or current pattern are calculated.

Machine-readable records:

- `configs/sigma_v19h_causal_observable_protocol.json`
- `results/sigma_v19h_member_phase/report.json`
- `results/sigma_v19h_member_phase/bullet_bootstrap.json`
- `results/sigma_v19h_member_phase/abell2146_bootstrap.json`
