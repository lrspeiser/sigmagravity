# Authenticated field-worker deployment milestone

Date: 2026-08-03

## Outcome

The generic field service can now run as a separate authenticated process or
container instead of being embedded only in the local development server. The
Vercel gateway has an opt-in connector for the upload and field-job routes. It
forwards requests only when both a worker HTTPS origin and a server-side bearer
secret are configured; otherwise the public routes retain their explicit 503
responses.

This closes the code boundary for one narrow production slice. It does not
claim that production storage or compute has been deployed.

## Verified request path

```text
researcher
  -> Vercel upload/field-job route
  -> bounded server-side proxy with hidden bearer credential
  -> non-root field-worker container
  -> content-addressed filesystem spool on a mounted volume
  -> confirmed-manifest Python solver subprocess
  -> immutable, quota-limited, rehashed artifacts
```

The worker permits only `data-uploads` and `field-jobs`. Galaxy extraction,
inverse-response jobs, batches, and arbitrary uploaded code are not exposed by
this container boundary. A valid bearer credential is required for every
scientific route. The unauthenticated health route exposes only the execution
class and quota configuration; it never returns the secret or storage path.

## Integrity and resource gates

- Upload tickets bind the declared byte length and SHA-256 before content is
  accepted.
- Job identity binds the confirmed model, input bundle, archive, preflight, and
  worker source hash. Resubmitting the same inputs returns the same job.
- The existing queue persists queued/running records and resumes interrupted
  work after restart.
- The solver subprocess has a wall-time limit, estimated-memory admission
  limit, stored-job limit, and bounded stdout/stderr capture.
- Publication now verifies the scientific manifest hash, artifact-index hash,
  job identity, every artifact byte count and SHA-256, safe unique filenames,
  file count, aggregate bytes, and absence of unindexed files or directories.
- A failed quota or integrity gate becomes an infrastructure failure; its
  output is not exposed as a scientific result.
- The container uses pinned numerical dependencies, runs as UID 10001, and
  writes only to `/var/lib/sigma` plus ordinary process-temporary locations.

## Real acceptance

The separate HTTP worker ran the full field-job smoke suite with a synthetic
test credential and temporary store:

| Case | Result |
|---|---:|
| Cartesian 2D field relative L2 error | `0.0014291183165795044` |
| Cartesian 3D field relative L2 error | `0.003218964440079798` |
| Axisymmetric field relative L2 error | `3.4364145737847694e-15` |
| Axisymmetric circular-speed RMSE | `4.220673123283083e-15 m/s` |
| Axisymmetric photon-deflection RMSE | `5.490987717737826e-26 arcsec` |
| Axisymmetric raw image-position RMS | `0.001692053225097455 arcsec` |

All four jobs passed queued/running/succeeded lifecycle checks, worker/gateway
source-hash agreement, full artifact downloads and rehashing, and zero
per-object gravity parameters. The raw-image case retained 13 verified
artifacts.

These are manufactured numerical answers. They validate the separated service
boundary and solver composition, not a galaxy or cluster theory.

## Build and run contract

Build from `research/galaxy-cluster-unification`:

```powershell
docker build -f Dockerfile.worker -t sigma-field-worker:0.29 .
```

The container requires a secret and should use a mounted persistent volume:

```text
SIMULATOR_WORKER_TOKEN=<at least 32 random bytes>
SIMULATOR_WORKER_STORE=/var/lib/sigma
```

The Vercel gateway connector requires:

```text
SIMULATOR_WORKER_URL=https://<private worker origin>
SIMULATOR_WORKER_TOKEN=<same secret>
```

Optional gateway and worker quotas are explicit environment values. Secrets
must be configured in the hosting control planes and never committed.

## What remains before calling it production

1. Deploy the image to a container platform and verify that `/var/lib/sigma`
   is a real persistent volume rather than an ephemeral path.
2. Configure the Vercel connector and run the same acceptance through the
   public alias.
3. Replace the single-volume spool with Postgres job/model metadata and
   S3/R2-compatible input and artifact objects, preferably using direct signed
   upload/download URLs rather than relaying large arrays through Vercel.
4. Add project authentication, tenant isolation, quotas, audit logs, retries,
   cache policy, license enforcement, and operational monitoring.
5. Add cryptographic result signing with a worker identity separate from the
   shared gateway credential.
6. Execute advanced uploaded plug-ins only in a second, single-use,
   network-disabled sandbox. The safe-manifest worker does not provide that
   capability.

Until items 1 and 2 are verified, the stable public deployment must continue
to report `production_worker_not_connected` and
`production_storage_not_connected`.
