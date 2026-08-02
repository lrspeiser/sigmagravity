# Asynchronous resolved-galaxy job API milestone

Date: 2026-08-02

## Outcome

The local reference API now exposes the P0720 gravity-independent galaxy core
through the same short-request lifecycle as generic field jobs:

```text
POST /api/v1/data-uploads
PUT  /api/v1/data-uploads/{id}/content
POST /api/v1/galaxy-jobs
GET  /api/v1/galaxy-jobs/{id}
GET  /api/v1/galaxy-jobs/{id}/events
GET  /api/v1/galaxy-jobs/{id}/artifacts
POST /api/v1/galaxy-jobs/{id}/cancel
```

`extract_roundtrip` accepts content-hashed gas and stellar surface-density maps.
It extracts a baryonic parameter package, regenerates the maps, creates an
explicitly prior-based 3D realization, scores the round trip, and publishes
verified artifacts. `generate` accepts the saved parameter package and declared
changes to mass, size, angular structure, local features, orientation, or
component offsets.

Both operations emit standard `sigma-array-bundle/1` metadata beside their NPZ
surface and volume products. A researcher can register those outputs as a new
immutable upload and feed them to any compatible generic field model. The
galaxy worker does not select or inspect the gravity theory.

## Real HTTP acceptance result

DDO101 from the P0639 resolved-map sample was packaged, uploaded, extracted,
regenerated, lifted to 3D, downloaded, and rehashed through real HTTP. Its
returned parameter package then seeded a separate generation job with a 1.25
gas-mass multiplier and other structural controls.

| Check | Result |
|---|---:|
| extraction lifecycle | queued → running → succeeded |
| generation lifecycle | queued → running → succeeded |
| extraction artifacts | 9 |
| generation artifacts | 8 |
| downloaded artifact hashes | all valid |
| gateway/worker source hash | identical |
| total-map normalized L2 error | 0.09337 |
| total-map pixel correlation | 0.99408 |
| requested gas-mass multiplier | 1.25 |
| measured gas-mass multiplier | 1.25 |
| gravity parameters | 0 |
| velocity targets used for extraction | no |
| extraction operational/scientific IDs | `job_d395f39badf90d431c7a33ca` / `galaxyjob_dde6ebde8f4e58ee7eb9e1a7` |
| generation operational/scientific IDs | `job_83d88216f1a5673974d519ca` / `galaxyjob_d6c07a71b57f23f766d14533` |

## Scientific boundary

- The API proves deterministic extraction/generation behavior and artifact
  integrity, not that the reconstructed 3D density is unique or true.
- The first saved volume is one draw from a declared thickness/flaring prior;
  all ensemble metadata and projection errors are retained.
- Round-trip metrics assess how much structure the representation retained.
  They do not score a law of gravity.
- The current contract consumes already registered face-on mass maps. Raw FITS
  images, PSFs, masks, uncertainty maps, inclination/distance posteriors,
  bulges, warps, and velocity cubes remain future layers.

## Operational boundary

This is still the local, single-user reference backend. The public Vercel route
advertises the schema but returns `production_worker_not_connected`. Durable
object storage, database records, a queue, isolated containers, auth, quotas,
monitoring, and retry policy are still required before public execution.

Reproduce while the development server is running:

```powershell
npm run smoke:galaxy-jobs
```
