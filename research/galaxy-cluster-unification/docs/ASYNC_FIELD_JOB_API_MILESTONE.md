# Asynchronous 2D/3D field-job API milestone

Date: 2026-08-02

## Outcome

The development API now moves a generic field model and actual NPZ data through
an asynchronous lifecycle without formula-specific application code:

1. `POST /api/v1/data-uploads` registers a `sigma-array-bundle/1` manifest and
   the expected archive byte count and SHA-256.
2. `PUT /api/v1/data-uploads/{id}/content` accepts only those exact bytes.
3. `POST /api/v1/field-jobs` binds the confirmed model, ready upload, numerical
   request, parameter policy, and exact worker-source hash.
4. `GET /api/v1/field-jobs/{id}` polls state without holding a solver request
   open.
5. Events, cancellation, artifact indexes, and allow-listed artifact downloads
   are separate endpoints.

The queue invokes only the safe manifest worker. It does not execute Python,
JavaScript, or plug-in code supplied by a researcher.

## Real HTTP acceptance result

A 25-by-25 2D manufactured Poisson field and a 17-by-17-by-17 3D field were
independently packaged, uploaded as NPZ bytes, queued through HTTP, executed by
the Python worker, polled to completion, and downloaded again through the
artifact API.

| Check | Result |
|---|---:|
| lifecycle | queued -> running -> succeeded |
| 2D relative L2 field error | 0.00142912 (0.143%) |
| 3D relative L2 field error | 0.00321896 (0.322%) |
| downloaded artifact hashes | 16 of 16 valid |
| gateway/worker source hash | identical |
| per-object gravity parameters | 0 |
| 2D operational/scientific IDs | `job_2e42ea82611f5c63f8163713` / `fieldjob_f3260316b4df446be9bb1597` |
| 3D operational/scientific IDs | `job_e7040c3cbbda092b35546ee1` / `fieldjob_718b934ed3dac2f5d2e36766` |

Operational IDs and scientific IDs are deliberately separate. The operational
ID binds the upload, preflight, service contract, and worker source. The
scientific ID is produced by the worker and binds the complete numerical job.

## Integrity and failure behavior

- Uploads are content-addressed and immutable after verification.
- Duplicate submissions return the existing job ID.
- Worker-source changes produce a new operational job identity.
- Jobs recovered after a gateway restart are requeued unless a completed
  scientific manifest already exists.
- Cancellation and result publication are serialized per job.
- Nonconvergence and model failures remain scientific terminal states with
  diagnostics. Internally invalid uploads are `rejected_input`; process,
  timeout, or missing-manifest failures are classified separately as
  infrastructure failures.
- Artifact indexes are checked against the scientific manifest, and each file
  is rehashed immediately before download.
- Artifact names are allow-listed, preventing path traversal.

## Honest boundary

This is a local, single-user reference backend. It proves the API semantics and
real worker handoff. It does not provide multi-tenant security or durable cloud
operation. Production still requires direct S3/R2 uploads, Postgres metadata,
a durable queue, single-use container scheduling, authentication, project
isolation, retry policy, monitoring, cost controls, and signed manifests.
