# Observation-conditioned baryonic uncertainty ensembles

Date: 2026-08-02

## Outcome

The gravity-independent resolved-galaxy worker now saves the uncertainty
realizations it claims to generate. A galaxy job can request a seeded ensemble
over declared baryonic and observation priors, and receives:

- every 2D gas, stellar, and total-baryon surface-density realization;
- every matching 3D vertical realization, with two explicit ensemble axes;
- 16th, 50th, and 84th-percentile surface-density maps;
- one JSON record and one CSV row for every 2D draw;
- content hashes for arrays and bundles;
- mass and morphology summaries; and
- an exact projection error for every component of every 3D draw.

Realization 0 is the unchanged central reconstruction. Later realizations can
vary gas and stellar mass, radial scale, Fourier structure, local residual
structure, center, rotation, distance scale, inclination deprojection, warp,
and a bounded co-spatial unseen-baryon fraction. The same request and seeds
replay the same numeric arrays.

This closes a concrete defect in the earlier path: previous jobs drew several
vertical profiles but retained only realization 0 as density data. The new
`volume_density_ensemble.npz` retains every surface-by-vertical combination.
The original `surface_density` and `volume_density` artifacts remain as the
anchor realization for compatibility with the generic field worker.

## Scientific meaning

The status string is deliberately
`observation_conditioned_prior_not_posterior`. These draws are useful for
testing whether a proposed gravity equation is robust to named baryonic
assumptions. They are not a data-derived posterior because the worker does not
yet evaluate a raw-image or spectral-cube likelihood.

The generator remains isolated from the tested gravity model:

- the parameter package has an empty `gravityParameters` object;
- `velocityTargetsUsed` is false;
- the ensemble request contains no gravity parameter; and
- the job reports zero universal and zero per-object gravity parameters.

Inclination uncertainty is currently a thin-map deprojection: an alternative
inclination rescales the intrinsic minor axis by
`cos(reference inclination) / cos(drawn inclination)` and then preserves total
mass. A warp shifts the vertical midplane outside half the component's r80.
The optional unseen-baryon fraction is intentionally narrow in meaning: it
adds mass in exact proportion to the traced baryon map. It is not a free halo.

## Resource and integrity gates

- 1 to 16 surface realizations.
- 1 to 8 vertical realizations per surface.
- Odd vertical grids from 9 to 129 cells.
- A preflight and worker-side 256 MiB raw 3D ensemble limit.
- Every 3D density must project to its corresponding 2D map to floating-point
  precision.
- Percentile maps must satisfy p16 <= p50 <= p84 pixel by pixel.
- The unchanged anchor must equal the ordinary generated map byte for byte.

## Real local HTTP acceptance

The DDO101 registered baryonic map passed through the real immutable upload,
queue, worker, download, and rehash path with three surface realizations and
two vertical realizations per surface.

| Check | Result |
|---|---:|
| surface ensemble shape | `3 x 65 x 65` |
| volume ensemble shape | `3 x 2 x 65 x 65 x 25` |
| maximum projection relative error | `3.64e-16` |
| extraction artifacts | 21 |
| generation artifacts | 20 |
| downloaded hashes | all valid |
| gateway/worker source hash | identical |
| gravity parameters | 0 |
| velocity targets used | no |
| total-map round-trip normalized L2 | `0.09337` |
| total-map pixel correlation | `0.99408` |

This is an engineering and uncertainty-propagation acceptance. The prior
widths used by the smoke test were illustrative and were not inferred from
DDO101 observations.

## What remains

Version 0.23 now provides deterministic unweighted propagation: a batch can
select named surface and vertical realizations, run one confirmed field model
and the same declared observation targets over every draw, and publish
per-realization plus p16/p50/p84 parent summaries. See
`BARYONIC_ENSEMBLE_PROPAGATION_MILESTONE.md`.

This is not the finished inverse
observing model. A likelihood-derived posterior still needs:

1. raw stellar images and H I / molecular-gas maps with WCS;
2. PSF, beam, channel response, noise covariance, masks, and selection effects;
3. distance, inclination, position-angle, and warp likelihoods;
4. stellar mass-to-light and gas-conversion uncertainty;
5. bulge/disk/bar/arm/clump decomposition and dust attenuation;
6. spectral cubes or 2D velocity fields used only for observation recovery,
   never to infer the baryonic map with the gravity formula under test;
7. posterior sampling and weights, followed by weighted observable
   aggregation rather than the current unweighted prior quantiles; and
8. held-out raw-observation round trips across a morphologically broad sample.

Public Vercel exposes the schema and guide but still does not execute these
jobs. Durable storage, a queue, isolated workers, authentication, and audit
infrastructure remain required for hosted computation.

## Reproduce

```powershell
python -m pytest tests/test_resolved_galaxy_generator.py tests/test_resolved_galaxy_job.py -q
node --test hosted-simulator/test/galaxy-job-preflight.test.mjs
```

The local HTTP end-to-end path is exercised by:

```powershell
npm --prefix hosted-simulator run smoke:galaxy-jobs
npm --prefix hosted-simulator run smoke:batches
```
