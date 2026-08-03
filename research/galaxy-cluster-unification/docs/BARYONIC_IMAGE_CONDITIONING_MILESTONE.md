# Gravity-independent baryonic image conditioning

Date: 2026-08-02

## Outcome

Version 0.24 adds the first data-likelihood step between a resolved baryonic
map and a formula-independent field batch. The extractor can assign importance
weights to its generated surface-density draws by comparing only the generated
gas and stellar maps with declared gas and stellar surface-density
measurements. The same immutable weights then follow those draws into any
compatible 2D/3D gravity model.

This is deliberately not a gravity fit. Rotation speeds, velocity fields,
lensing constraints, inferred dark-matter maps, solved potentials, and
accelerations are forbidden as conditioning inputs. They remain downstream
targets used to test the frozen formula.

## Implemented likelihood

For component `c` (gas or stars), draw `j`, valid pixels `i`, measured surface
density `D`, and declared per-pixel uncertainty `sigma`, the worker evaluates

```text
log L_j = -1 / (2 A_corr) * sum_c sum_i ((S_j,c,i - D_c,i) / sigma_c,i)^2
```

`A_corr` is a disclosed correlation-area approximation in pixels. It prevents
the diagonal likelihood from pretending that every oversampled neighboring
pixel is independent; it is not a replacement for a real covariance matrix.
Given prior weights `pi_j`, the normalized surface weights are

```text
w_j = pi_j exp(log L_j - max(log L)) / sum_k pi_k exp(log L_k - max(log L))
```

and their effective sample size is

```text
ESS = 1 / sum_j w_j^2.
```

The worker records input-array hashes, mask use, valid-pixel counts,
log-likelihoods, prior and normalized weights, entropy, ESS, normalized ESS,
and a weight-quality status. It publishes `credibleIntervalReady: false` in
this commissioning version.

## Request and artifact contract

Conditioning is accepted only for `extract_roundtrip` jobs with a declared
uncertainty ensemble. The uploaded `sigma-array-bundle/1` must contain:

- `gas_surface_density_uncertainty`, scalar, `M_sun/kpc^2`;
- `stellar_surface_density_uncertainty`, scalar, `M_sun/kpc^2`; and
- optionally `baryonic_conditioning_mask`, scalar and dimensionless.

All arrays must share the observed map shape. New deterministic artifacts are:

- `baryonic_conditioning.json`;
- `baryonic_conditioning_weights.csv`;
- conditioned p16/p50/p84 gas, stellar, and total surface maps;
- the original complete prior ensemble; and
- `ensemble_prediction_quantiles.csv` after a batch propagates the selected
  draws through a field model and observation adapter.

The 3D materializer verifies the weight vector, its hash, its sum, ESS, and
quality classification before assigning each child a joint realization
weight. Surface weights are conditioned; the selected vertical profiles are
still equally weighted conditional priors.

## Real DDO101 HTTP acceptance

The end-to-end acceptance used the registered DDO101 baryonic map, explicitly
labeled commissioning fractional uncertainty maps, two surface draws, one
vertical draw per surface, a fixed Newtonian Poisson manifest, and ten
published LITTLE THINGS rotation-curve points.

| Check | Result |
|---|---:|
| parent systems | 4 |
| expanded field children | 5 |
| converged children | 5/5 |
| selected surface weights | `[1.0, 0.0]` |
| effective sample size | `1.0 of 2` |
| weight quality | `degenerate_importance_weights` |
| credible interval ready | no |
| weighted prediction rows | 10 |
| maximum equation residual | `2.62263e-10` |
| per-object gravity parameters | 0 |
| observation-added gravity parameters | 0 |
| deterministic artifacts | 17 |
| downloaded hashes | all valid |
| Newtonian fixture normalized RMSE | `0.0894727` |
| aggregate rotation RMSE | `40.174 km/s` |
| conditioned-weight rotation RMSE | `42.887 km/s` |

The collapsed weights are a useful negative commissioning result. With only
two proposal draws and approximate uncertainty maps, one draw receives
essentially all the probability. The software therefore refuses to describe
the weighted bands as credible intervals. It does not smooth over this failure
or use the measured rotation curve to repair it.

## What this establishes

- A baryonic observation can weight candidate source reconstructions before a
  gravity formula sees them.
- Those weights are immutable and model-independent, so Newtonian, MOND-like,
  Refracted Gravity, Sigma Gravity, and two-potential models can receive the
  same baryonic uncertainty distribution.
- The batch can report both unweighted and weighted score summaries plus a
  p16/p50/p84 predicted circular speed at each observed radius.
- Changing a velocity or lensing target cannot change the baryonic weights or
  field-job identity.
- Degenerate importance sampling is detectable and reported as a scientific
  limitation, not a narrow success band.

## What this does not establish

- The fractional commissioning uncertainties are not released survey error
  maps and do not represent a calibrated DDO101 posterior.
- A diagonal Gaussian likelihood plus `A_corr` does not encode PSF/beam
  convolution or spatial covariance.
- Two draws cannot resolve a posterior distribution.
- Surface maps do not identify scale height, bulge depth, warp, dust, or other
  line-of-sight structure.
- Weighted agreement with a rotation curve would not uniquely identify a
  gravity law, and this Newtonian commissioning score is physically poor.
- Public Vercel publishes the contract and guide but does not execute heavy
  galaxy, field, or batch jobs.

## Next build gates

1. Register released map uncertainty, PSF/beam, mask, and covariance products.
2. Evaluate the likelihood in observation space after PSF/beam convolution.
3. Replace a tiny fixed importance sample with adaptive sampling and require a
   preregistered minimum ESS before reporting calibrated intervals.
4. Add independent likelihood terms for bulge/depth/scale-height information;
   keep unavailable dimensions explicitly prior-only.
5. Calibrate interval coverage on synthetic hidden-truth galaxies, then freeze
   the pipeline before a morphologically varied whole-galaxy holdout.
6. Extend weighted prediction products from circular-speed rows to velocity
   fields, spectral channels, photon maps, and raw lensing observables.
7. Connect durable storage, queueing, isolated workers, authentication, quotas,
   cancellation, and signed audit records before enabling public execution.

## Reproduce

```powershell
python -m pytest tests/test_resolved_galaxy_job.py tests/test_galaxy_ensemble_materializer.py -q
node --test hosted-simulator/test/galaxy-job-preflight.test.mjs hosted-simulator/test/local-batch-service.test.mjs
npm --prefix hosted-simulator run smoke:batches
```
