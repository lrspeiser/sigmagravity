# Formula-independent baryonic ensemble propagation

Date: 2026-08-02

## Outcome

The local reference platform can now run one exact, researcher-confirmed
2D/3D field model across selected realizations from a resolved galaxy's
baryonic uncertainty ensemble. This closes the gap between saving plausible
density fields and measuring how those uncertainties change predicted
rotation or lensing observables.

This is a platform capability, not evidence for Sigma Gravity, MOND, dark
matter, or any other model. The model remains frozen, the baryonic draws carry
no gravity parameters, and observation evaluation happens in separately
content-addressed jobs after each field solve.

## Request contract

A system in `POST /api/v1/batches` may use:

```json
{
  "id": "DDO101",
  "galaxyJobId": "job_content_addressed_id",
  "galaxyArtifact": "volume_density_ensemble",
  "ensembleSelection": {
    "surfaceRealizations": [0, 1, 2],
    "verticalRealizations": "all",
    "maximumChildren": 24
  },
  "observationTargets": ["typed target omitted here"]
}
```

`surface_density_ensemble` accepts only surface indices.
`volume_density_ensemble` requires both surface and vertical selections.
Either axis may be `"all"` or an explicit unique array. One source is capped
at 128 children and the expanded batch remains capped at 1,000 actual solver
runs.

## Integrity path

For every selected draw the gateway:

1. rehashes the completed galaxy-job artifact index and parent ensemble files;
2. verifies the ensemble manifest, axes, shapes, units, array keys, and every
   full-array content hash;
3. slices the named realization and converts `M_sun/kpc^2` or
   `M_sun/kpc^3` to `kg/m^2` or `kg/m^3`;
4. writes a deterministic standard `sigma-array-bundle/1` archive;
5. binds its provenance to the parent ensemble hash and exact indices;
6. creates an immutable upload and ordinary field child; and
7. optionally creates an independently identified observation-evaluation
   child against the same declared observations.

The generic field and observation workers are unchanged. A new formula does
not need an ensemble-specific code branch.

## Deterministic reports

Every batch now contains:

- `per_galaxy.csv`, with parent ID and realization axes on every row;
- `per_realization.csv`, containing only ensemble children;
- `ensemble_summary.json` and `ensemble_summary.csv`, with per-parent counts
  and p16/p50/p84/min/max summaries for convergence and every available typed
  observation score;
- the existing predictions, failures, aggregate, HTML, LLM briefing, child
  manifest, reproduction command, and signed hashes.

The summary status is
`prior_prediction_spread_not_measurement_posterior`. Percentiles are
unweighted across declared prior draws. They are not likelihood-derived
credible intervals.

Version 0.24 extends this contract without removing the prior summaries. When
the parent galaxy job includes gravity-independent baryonic-image likelihood
weights, the batch also publishes weighted score distributions and
`ensemble_prediction_quantiles.csv`, with p16/p50/p84 predicted circular speed
at each observed radius. It carries ESS and weight-quality diagnostics and
keeps `credibleIntervalReady` false. The full conditioning boundary and real
collapsed-weight result are documented in
`BARYONIC_IMAGE_CONDITIONING_MILESTONE.md`.

## Real DDO101 HTTP acceptance

The acceptance path generated a central DDO101 reconstruction plus two 3D
baryonic realizations, materialized and solved both ensemble children with the
published-fixed Newtonian Poisson manifest, and scored the same published
LITTLE THINGS rotation curve against both.

| Check | Result |
|---|---:|
| parent systems | 4 |
| expanded solver children | 5 |
| ensemble children | 2 |
| converged children | 5/5 |
| maximum equation residual | `2.62e-10` |
| scored rotation points | 30 |
| per-object gravity parameters | 0 |
| observation-added gravity parameters | 0 |
| deterministic report artifacts | 16 |
| downloaded artifact hashes | all valid |
| old Newtonian fixture normalized RMSE | `0.08947` |

The aggregate Newtonian rotation RMSE across the anchor and two ensemble
draws was `40.37 km/s`. That poor physical fit is expected for baryons-only
Newtonian gravity and is retained as a result, not treated as an infrastructure
failure.

## What this can and cannot establish

It can establish that a frozen formula was numerically executed across a
declared baryonic prior family, which draws failed, how much its requested
predictions moved, and whether any per-object gravity flexibility entered.

It cannot establish that the prior widths are observationally correct, that a
good score identifies a unique causal theory, or that a galaxy result transfers
to cluster lensing. Version 0.24's first surface likelihood remains incomplete:
the next scientific step is released uncertainty/covariance maps, raw
images/cubes, WCS, PSF/beam forward modeling, distance, inclination,
stellar-population, gas-conversion, dust, morphology, and adaptive sampling.
The next infrastructure step is durable production storage,
queueing, isolated workers, authentication, quotas, cancellation, and audit
logs; Vercel still publishes only the contract and lightweight APIs.

## Reproduce

```powershell
python -m pytest tests/test_galaxy_ensemble_materializer.py -q
node --test hosted-simulator/test/local-batch-service.test.mjs
npm --prefix hosted-simulator run smoke:batches
```
