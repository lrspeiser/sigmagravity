# Massive-tracer observation adapter milestone

Date: 2026-08-02

## Outcome

The generic field and batch APIs can now make and score a real galaxy
prediction, rather than stopping at numerical convergence. A
`sigma-observation-target/1` document attaches a circular-speed curve to one
system and identifies a compatible model observable. The model still controls
the field equation; the target is evaluated only after the solve.

For a declared massive-tracer acceleration vector **a** on a Cartesian 2D or
3D grid, the adapter samples a ring at radius `R`, computes

```text
g_R(R) = mean_theta[-a(R,theta) dot e_R(theta)]
v_c(R) = sqrt(R g_R(R))
```

and reports the azimuthal coverage at every point. A prediction is invalid if
coverage is below the declared threshold or the mean inward acceleration is
not positive. It is not silently clipped into a plausible velocity.

Targets may provide independent velocity uncertainties or a full symmetric
positive-definite covariance matrix. Deterministic outputs include point
predictions and residuals, RMSE, inverse-variance-weighted RMSE, chi-square,
degrees of freedom, reduced chi-square, and Gaussian log likelihood.

The contract enforces all of the following before execution:

- the observable exists and was explicitly requested;
- it is a vector in `m/s^2` tagged for `massive_tracers`;
- dimensions, origin, center, plane axes, radii, and coverage are valid;
- target IDs are unique;
- observations and uncertainty dimensions agree;
- provenance and license metadata are present; and
- fitted nuisance parameters are explicitly counted.

A photon observable cannot be used for a rotation curve. Photon lensing will
use a separate adapter and scoring contract.

## Known-answer acceptance

Analytic 2D and 3D solid-body fields use
`a=(-omega^2 x,-omega^2 y,0)`, for which the exact curve is
`v_c=omega R`. The adapter recovers all four tested radii to below `2e-14`
relative/numerical tolerance. A separate fixture reproduces the direct
full-covariance chi-square calculation. Field-job tests then solve the
quadratic potential itself and reproduce the same curve through the complete
artifact path.

## Real DDO101 HTTP result

The strict local HTTP run performs the following chain:

1. upload the registered DDO101 gas and stellar density maps;
2. extract gravity-independent baryonic parameters;
3. regenerate a 25 x 25 x 9 baryonic volume;
4. solve one published-fixed Newtonian Poisson manifest;
5. derive the circular-speed curve from its massive-tracer acceleration;
6. score all ten published LITTLE THINGS circular-speed measurements; and
7. download and re-hash the batch and child prediction artifacts.

| Check | Result |
|---|---:|
| field/batch state | succeeded |
| valid observed points | 10 / 10 |
| per-object gravity parameters | 0 |
| Newtonian versus observed RMSE | `40.0443 km/s` |
| uncertainty-weighted RMSE | `42.7228 km/s` |
| reduced chi-square | `198.1283` |
| API versus frozen Newtonian curve RMSE | `0.49615 km/s` |
| API versus frozen Newtonian normalized RMSE | `0.05312` |
| worst field-equation residual | `2.7144e-10` |
| downloaded report artifacts | `10 / 10` hashes valid |
| batch ID | `batch_172db0645f53888f84a6b6b8` |
| batch scientific SHA-256 | `a4e1be6987606bd8a2aa744859fdf450833f2be07b4c5fa4abaefe017e441ab8` |

The poor observational result is expected for baryons-only Newtonian gravity
in this dwarf. It is evidence that the scorer is exposing the missing
acceleration rather than forcing agreement. The important implementation
cross-check is that the new generic worker/adapter follows the earlier frozen
Newtonian calculation to `0.496 km/s` RMS despite the coarse commissioning
grid.

## Scientific and operational boundary

This is one dwarf, one observable type, and one registered/deprojected baryonic
map. The vertical density is a declared prior, the box uses the current
finite-boundary approximation, and the 25 x 25 x 9 resolution is for API
commissioning rather than final inference. Distance, inclination,
mass-to-light, map, and covariance systematics are not marginalized here.

No photon deflection, shear, image root, resolved velocity field, or cluster is
scored. A theory cannot claim joint galaxy-and-lensing success from this result.
The public Vercel route remains disconnected from durable scientific workers.

## Next acceptance gate

Run the same adapter over the full resolved dwarf set with an untouched split,
resolution/boundary/vertical-prior sensitivity, and frozen Newtonian, AQUAL,
QUMOND, and Refracted Gravity manifests. Then implement a beam-aware
line-of-sight velocity-field adapter. Photon lensing remains a distinct later
gate with raw image positions and topology.

Reproduce while the development server is running:

```powershell
npm run smoke:batches
```
