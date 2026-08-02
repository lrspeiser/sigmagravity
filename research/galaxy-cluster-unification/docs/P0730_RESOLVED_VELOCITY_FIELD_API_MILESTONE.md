# Formula-neutral resolved velocity-field API milestone (P0730)

Date: 2026-08-02

## Outcome

The local simulator can now evaluate a resolved line-of-sight velocity map from
any compatible Cartesian 2D or 3D field model. The adapter contains no theory
name, galaxy name, or model parameter. It consumes only a declared
`massive_tracers` vector acceleration in `m/s^2`, observation geometry, and
content-hashed observation arrays.

This is an engineering acceptance result. It is not yet a scientific score on
the real LITTLE THINGS maps and it is not a photon-lensing result.

## Mapping

At each declared disk-plane coordinate `(x, y)`, the adapter interpolates the
solved acceleration field and computes

```text
r = sqrt(x^2 + y^2)
a_in = -(a_x x + a_y y) / r
v_c = sqrt(max(r a_in, 0))
v_los = handedness sin(inclination) v_c x / r
```

If a beam is declared, the predicted map is convolved as an
intensity-weighted velocity moment:

```text
v_beam = convolution(I v_los, K) / convolution(I, K)
```

The observed systemic zero point, mask, uncertainties, intensity weights, and
beam kernel are explicit target inputs. They are never inferred from a gravity
residual. A scored target produces ordinary, inverse-variance, and optionally
intensity-weighted RMSE; chi-square; degrees of freedom; reduced chi-square;
Gaussian log likelihood; and a deterministic per-pixel CSV.

## API and artifact behavior

- `sigma-observation-target/1` now accepts
  `kind=line_of_sight_velocity_field`.
- Every observation map and kernel is referenced by an array key whose shape,
  unit, and content hash are checked during preflight and upload.
- The field job writes `observation_velocity_field_predictions.csv`; circular
  curves remain in `observation_predictions.csv` so map pixels cannot be
  mislabeled as radial points.
- A multi-system batch aggregates the map prediction table and observation
  scores while preserving the one frozen model and parameter policy.
- The adapter rejects photon observables and cannot alter or re-solve the
  submitted field equation.

## Verification completed

- Analytic projected solid-body maps pass for both 2D and 3D acceleration
  fields.
- Intensity-weighted beam convolution and weighting gates pass.
- Shape, unit, missing-array, coordinate-system, tracer-type, and minimum-pixel
  failures are rejected.
- A full field job writes and indexes the scored map artifact.
- A multi-system batch validates and aggregates the resolved-map artifact.
- The complete local Node API suite and focused Python worker suite pass.

## Honest limits and next gates

The present mapping assumes circular equilibrium in a fixed declared plane.
It does not yet model asymmetric drift, pressure support, velocity dispersion,
warps, bars, radial flows, outflows, multiple gas components, channel response,
or a full spectral cube. Those effects must become separately declared
nuisance/forward-model terms rather than hidden gravity adjustments.

The next acceptance is to package the thirteen real LITTLE THINGS moment maps,
uncertainties, masks, intensity maps, beams, and projection geometry into
immutable bundles; reproduce the frozen P0712 results within a predeclared
tolerance; and run every compatible fixed model over the same systems. A
separate photon-typed adapter is required for cluster lensing.
