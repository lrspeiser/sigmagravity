# Formula-neutral resolved-galaxy comparator results (P0723)

Date: 2026-08-02

## Outcome

The local asynchronous API ran four independently declared field manifests
through all 13 registered LITTLE THINGS galaxies. All 52 formula--galaxy
solves converged, every system produced a massive-tracer circular-speed score,
all downloaded artifacts passed their recorded SHA-256 checks, and no model
used a per-galaxy gravity parameter.

This closes the first full-sample formula-neutral execution gate. Newtonian
Poisson, AQUAL, QUMOND, and Refracted Gravity entered through the same model,
data-upload, galaxy-generation, batch, and observation-adapter contracts. The
application did not select code by theory name.

## Frozen protocol

The configuration was committed in
`configs/p0723_formula_neutral_api_comparators.json` before the reported scores
were inspected. Its SHA-256 is
`a81c416e5b2a55d0356679f49f7f3e0ce2a5dfc56d5b68cb22f6c313c88127ac`.

- Sample: all 13 registered P0639 dwarf-galaxy maps.
- Reconstruction: gravity-independent radial/Fourier/sparse-residual
  parameter extraction followed by a `33 x 33 x 9` generated density volume.
- Observation: 161 published circular-speed points, with their tabulated
  uncertainties, sampled from each manifest's massive-tracer acceleration.
- Parameter policy: published fixed; zero fitted nuisance parameters and zero
  per-object gravity parameters.
- Numerical gates: 100% convergence, equation residual at most `1e-7`, at
  least 11 common scored systems, verified artifact hashes, and normalized
  RMSE at most `0.2` against the pre-existing P0708 curves where available.
- Sample status: previously unsealed and project-spent. This is engineering
  conformance, not a fresh blind scientific validation.

The input kinematics archive SHA-256 is
`967110269d59357ee3a94d1d6e46c2402aef38da3f674180d42044ceaf094173`.
The independent P0708 curve table SHA-256 is
`b9912258ed9cf223987817b153de97c09c49ce6a55ac6bb1e3cfd42ccc0adcbc`.

## Results

| Manifest | Universal gravity values | Per-galaxy gravity values | Equal-galaxy RMSE | Reduced chi-square | Frozen-curve normalized RMSE | Worst equation residual |
|---|---:|---:|---:|---:|---:|---:|
| Newtonian Poisson | 1 | 0 | `23.154 km/s` | `51.468` | `0.1491` | `5.93e-10` |
| AQUAL, simple mu | 2 | 0 | `13.131 km/s` | `12.848` | `0.0948` | `9.94e-9` |
| QUMOND, simple nu | 2 | 0 | `12.486 km/s` | `15.018` | `0.1743` | `5.53e-10` |
| Refracted Gravity published fixture | 4 | 0 | `14.439 km/s` | `20.899` | not previously frozen | `5.96e-10` |

Every declared engineering gate passed. QUMOND has the lowest unweighted
equal-galaxy speed RMSE in this coarse run, while AQUAL has the lowest aggregate
reduced chi-square and the closest numerical agreement with the earlier frozen
curves. Refracted Gravity outperforms baryons-only Newtonian gravity but does
not outperform these two MOND field formulations on this sample.

These are not good absolute fits under the tabulated statistical uncertainties:
even the lowest aggregate reduced chi-square is about `12.85`, far above one.
The comparison therefore demonstrates a working neutral test path and exposes
model/data/reconstruction mismatch; it does not establish any tested model as
an adequate description of the galaxies.

## What differs by galaxy

| Manifest | Three lowest-RMSE systems | Three highest-RMSE systems |
|---|---|---|
| Newtonian | CVnIdwA `5.19`, DDO216 `7.04`, DDO53 `8.20` | DDO101 `40.18`, DDO47 `35.33`, DDO87 `30.24` |
| AQUAL | DDO216 `2.80`, DDO126 `3.02`, DDO210 `3.37` | DDO101 `26.63`, NGC1569 `25.05`, DDO47 `15.75` |
| QUMOND | DDO53 `1.62`, DDO216 `2.15`, DDO126 `2.34` | DDO101 `29.08`, DDO47 `19.06`, DDO133 `14.19` |
| Refracted Gravity | DDO53 `3.35`, DDO216 `3.39`, DDO126 `3.67` | DDO101 `33.83`, DDO47 `21.36`, DDO133 `17.17` |

Values are circular-speed RMSE in km/s. DDO101 is the worst observational fit
for every modified-gravity fixture and also for Newtonian gravity. NGC1569 is
the largest coarse-grid numerical-conformance outlier for Newtonian and
QUMOND. That behavior is visible rather than excluded and motivates a declared
resolution/box/vertical-prior sensitivity study.

## Numerical changes required by the run

Two generic edge cases were fixed without adding a formula-specific route:

1. QUMOND's singular `nu(|grad Phi_N|)` has a finite physical flux limit when
   multiplied by an exactly zero gradient. The manifest now declares that
   convention with the general typed `multiply_zero_vector_limit` operator.
   Nonfinite scalar values paired with nonzero vectors still fail.
2. A compact component can collapse into one central pixel when a
   high-resolution parameter package is replayed on a coarse grid. Its depth
   is unresolved, not physically zero. The generator now uses one radial cell
   as a disclosed vertical-prior resolution floor while retaining the measured
   `r80` and exact projected-mass closure. This allowed NGC1569 to run rather
   than be dropped.

The local queue also demonstrated persisted restart recovery: one interrupted
QUMOND child was requeued from its immutable request and completed with the
same content identity.

## Scientific boundary and next gates

P0723 tests circular speeds of massive tracers only. It does not score a
resolved line-of-sight velocity field, beam smearing, photon deflection,
strong-lens image roots, shear, clusters, Solar-System behavior, or an
untouched galaxy holdout. The vertical structure is a prior realization and
the outer boundary is the current zero-Dirichlet commissioning approximation.

The next product gates are therefore:

1. Freeze and run grid, box, boundary, and vertical-prior sensitivity on this
   spent sample without changing P0723's result.
2. Add a beam-aware resolved line-of-sight velocity-field/spectral-cube
   adapter and test it on known-answer mock cubes before real data.
3. Build the photon-lensing adapter as a separate observable contract, then
   validate deflection, convergence, shear, and image-root recovery on analytic
   lenses before clusters.
4. Expand the real catalog beyond gas-rich dwarfs and reserve untouched whole
   galaxies for morphology and kinematic holdouts.
5. Add uncertainty-aware inverse/forward reconstruction with hidden-truth
   coverage tests for inclination, distance, bulges, thickness, warps, PSF,
   beam, noise, and mass-to-light ratio.
6. Connect the proven local contracts to durable object storage, a queue,
   isolated container workers, metadata storage, authentication, quotas, and
   monitoring before enabling public scientific execution.

## Reproduce

Start the local development service, then run:

```powershell
python scripts/run_p0723_formula_neutral_api_comparators.py `
  --base-url http://127.0.0.1:4173
```

The deterministic summary, point predictions, per-galaxy scores, and plots are
under `results/p0723_formula_neutral_api_comparators/`.
