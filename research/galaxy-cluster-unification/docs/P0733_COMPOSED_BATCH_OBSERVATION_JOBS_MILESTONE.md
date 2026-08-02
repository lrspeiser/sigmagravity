# P0733 composed batch observation jobs

Date: 2026-08-02

## Outcome

P0733 changes the local multi-system batch from one integrated child per
system into an explicit two-stage composition:

1. create or reuse one observation-independent field job;
2. after that field succeeds, create or reuse at most one separately hashed
   observation-evaluation job for the system's declared targets;
3. retain both identities, scientific hashes, artifact hashes, and failure
   states in one deterministic batch report.

This closes the immediate orchestration gap left by P0732. A researcher can
now change measured velocities, masks, beams, uncertainties, or target
declarations without changing the gravity result, provided the model, source
mass distribution, grid, boundary conditions, solver controls, and requested
field observables remain unchanged.

## Contract change

`sigma-batch-submit/1` remains the public submission type. Each system may now
provide both:

- `dataUploadId`: arrays used by the field equation;
- `observationDataUploadId`: independently content-addressed measured arrays
  used only by the observation adapter.

If `observationDataUploadId` is omitted, the field upload is reused for
backward compatibility. Separating the uploads is what makes it possible to
change a velocity map or mask without invalidating the field-child cache.

The batch preflight publishes separate `fieldPreflightSha256`,
`observationBindingSha256`, `inputBundleSha256`, and
`observationBundleSha256` values. The field job receives zero observation
targets. A successful field with targets then becomes the immutable input to a
`sigma-observation-evaluation-job-submit/1` child.

## Deterministic report changes

`child_jobs.json` and `per_galaxy.csv` now retain:

- field job, scientific job, result, and manifest identities;
- observation-evaluation job, scientific job, result, and manifest identities;
- the observation child's full published artifact index;
- separate field and observation states;
- separately classified field, observation-creation, and observation-worker
  failures;
- the number of gravity parameters added by observation evaluation.

Aggregate convergence remains a field-solver quantity. Observation scores are
aggregated only from successful observation children. A failed observation
therefore cannot erase a successful field solve or contribute partial scores.

## Frozen acceptance result

The preregistration is
[`../configs/p0733_composed_batch_observation_jobs.json`](../configs/p0733_composed_batch_observation_jobs.json).
The immutable result is
[`../results/p0733_composed_batch_observation_jobs/report.json`](../results/p0733_composed_batch_observation_jobs/report.json).

All 15 frozen engineering gates passed:

- 65/65 hosted tests passed;
- the real HTTP/Python-worker batch solved three DDO101-derived 3D systems;
- the scored system's field child contained zero observation targets;
- both field-only systems created zero observation children;
- changing the declared uncertainty preserved field job
  `job_01f55e6b415b91018ca7b355`;
- that change produced a different observation-evaluation job;
- repeating the changed submission reused the composed batch identity;
- field rejection created no observation job;
- observation rejection was classified separately and excluded from scores;
- cancellation reached a running observation child but retained its completed
  field;
- restart recovery rebuilt reporting from completed children without rerunning
  either worker;
- aggregate prediction rows exactly preserved the standalone observation-child
  rows, and the report retained the child's score/prediction hashes;
- all downloaded batch artifacts rehashed correctly;
- per-object gravity parameters were zero;
- gravity parameters added by observation evaluation were zero.

The real Newtonian commissioning run retained ten DDO101 points. Its coarse
grid reproduced the earlier frozen Newtonian prediction to `0.496 km/s` RMS,
or `0.0531` normalized RMS. The observed-curve RMSE was `40.04 km/s`; this is a
pipeline conformance fixture, not evidence that baryon-only Newtonian gravity
fits DDO101.

## Reproduce

```powershell
python scripts/run_p0733_composed_batch_observation_jobs.py
```

The result directory is immutable. The regression test also rehashes every
scientific orchestration source named by the report.

## Claim boundary and next work

P0733 validates composition, caching, provenance, lifecycle behavior, and
failure accounting. It changes neither a gravity equation nor an observation
mapping. It does not add photon lensing, pressure support, non-circular motion,
or a blind scientific holdout.

The local queue is still filesystem-backed and single-user. Public execution
still needs durable job metadata, object storage, isolated container workers,
authentication, quotas, retries, and monitoring. The next scientific adapter
milestone is a separately typed photon-lensing path; light must never be
silently evaluated with the massive-tracer rule.
