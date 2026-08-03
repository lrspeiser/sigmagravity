# Authenticated resolved-galaxy worker milestone

Date: 2026-08-03

## Outcome

The separately deployable authenticated worker now exposes the existing
gravity-independent resolved-galaxy extractor and generator in addition to
confirmed field jobs. A researcher can upload registered gas and stellar
surface-density arrays, extract a content-hashed baryonic parameter package,
retain seeded 2D/3D uncertainty ensembles, and use the returned package to
generate controlled structural variants.

The extraction stage cannot inspect a velocity field, lensing target, dark
matter map, or submitted gravity model. Its parameter accounting reports zero
universal and zero per-object gravity parameters. This separation is essential:
the fake galaxy describes the baryonic input and observation assumptions, not
the answer a candidate gravity theory is expected to produce.

## Authenticated lifecycle

```text
POST /api/v1/data-uploads
PUT  /api/v1/data-uploads/{id}/content
POST /api/v1/galaxy-jobs
GET  /api/v1/galaxy-jobs/{id}
GET  /api/v1/galaxy-jobs/{id}/events
GET  /api/v1/galaxy-jobs/{id}/artifacts
GET  /api/v1/galaxy-jobs/{id}/artifacts/{name}
POST /api/v1/galaxy-jobs/{id}/cancel
```

Every route except `/healthz` requires the server-side worker bearer secret.
The Vercel proxy accepts only the exact upload, field-job, and galaxy-job path
families; traversal, redirects, oversized requests, oversized responses, and
unconfigured or partial credentials fail closed. Observation-evaluation,
inverse-response, batch, and arbitrary-code routes remain outside this worker
boundary.

## Real separated-process acceptance

The acceptance uploaded the registered DDO101 gas and stellar maps, performed
an extraction and round trip, downloaded and rehashed every artifact, then
submitted the returned `parameters.json` to a second generation job.

| Check | Result |
|---|---:|
| extraction lifecycle | queued -> running -> succeeded |
| generation lifecycle | queued -> running -> succeeded |
| extraction artifacts | 23 |
| generation artifacts | 22 |
| total-map normalized L2 error | `0.09337440277450426` |
| total-map pixel correlation | `0.9940761625088308` |
| surface ensemble shape | `3 x 65 x 65` |
| volume ensemble shape | `3 x 2 x 65 x 65 x 25` |
| maximum 3D-to-2D projection error | `2.7396576183397723e-16` |
| requested gas-mass scale | `1.25` |
| measured gas-mass ratio | `1.25` |
| gravity parameters | `0` |
| velocity targets used during extraction | no |
| downloaded hashes and worker-source identity | all valid |

When Git LFS data are unavailable in a clean CI checkout, the same smoke client
uses a deterministic analytic baryonic map. It labels that source explicitly;
it never describes the analytic fixture as a real galaxy.

## Real Linux-container acceptance

GitHub Actions built the non-root production image from the pinned worker
requirements and ran both authenticated HTTP suites successfully:
<https://github.com/lrspeiser/sigmagravity/actions/runs/30795427239>.

| Container check | Result |
|---|---:|
| Cartesian 2D relative field error | `0.0014291183165795315` |
| Cartesian 3D relative field error | `0.0032189644400797907` |
| axisymmetric relative field error | `3.344067911339841e-15` |
| circular-speed RMSE | `3.903582677488951e-15 m/s` |
| photon-deflection RMSE | `5.369872558614447e-26 arcsec` |
| raw image-position RMS | `0.0016920532250985097 arcsec` |
| galaxy total-map normalized L2 error | `0.05478050421226611` |
| galaxy total-map pixel correlation | `0.9972455569454114` |
| extraction / generation artifacts | `23 / 22` |
| maximum ensemble projection error | `2.466495895856214e-16` |
| requested / measured gas-mass scale | `1.25 / 1.2500000000000002` |
| gravity parameters / velocity targets | `0 / none` |

The first container attempt exposed a genuine packaging gap: the field paths
did not import the Astropy-dependent map module, but the galaxy path did. The
image now pins Astropy `7.1.1`; a clean minimal-runtime extraction and the full
Linux container lifecycle both pass. Future CI failures also print only the
bounded worker stderr logs, not uploaded scientific arrays or credentials.

## What this enables

The returned density products use the standard array-bundle contract. They can
be registered as immutable uploads and supplied to any compatible confirmed
field model. This creates the correct conceptual pipeline:

```text
baryonic observations
  -> gravity-blind extraction and uncertainty ensemble
  -> controlled generated galaxy
  -> separately chosen confirmed gravity model
  -> predicted field and observable
  -> comparison with withheld velocity or lensing data
```

Changing the gravity model does not change the extracted baryonic package.
Changing a declared galaxy control creates a new content identity rather than
overwriting the source system.

## What this does not yet enable

- The public Vercel deployment has no external worker or persistent volume, so
  these routes still return HTTP 503 in production.
- The input must already be a registered face-on baryonic mass map. Raw FITS
  calibration, PSF/beam convolution, dust, multiband mass-to-light inference,
  spectral cubes, bulge depth, and adaptive posterior sampling are incomplete.
- The 3D ensemble is a declared prior family, not a uniquely inferred galaxy.
- Galaxy generation and field execution are still two explicit jobs. The
  authenticated batch/observation lifecycle is not exposed yet.
- Large artifact delivery is bounded by the gateway response quota. Production
  needs direct signed S3/R2 downloads instead of relaying large arrays through
  Vercel.
- This validates software identity and reconstruction behavior, not Sigma
  Gravity, MOND, dark matter, or any other physical theory.

## Next production step

Build and run `Dockerfile.worker` on a container host with a verified persistent
volume, configure the Vercel worker origin and rotated secret, and repeat both
the four-case field acceptance and this two-job galaxy acceptance through the
public alias across a worker restart. Postgres metadata and immutable object
storage remain the required architecture after that bounded deployment test.
