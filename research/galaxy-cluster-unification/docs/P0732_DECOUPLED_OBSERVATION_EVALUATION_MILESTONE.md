# Decoupled observation-evaluation milestone (P0732)

Date: 2026-08-02

## Outcome

P0732 passed every frozen gate. The local API can now take one immutable,
successful 2D or 3D field job and evaluate new observation data without
executing the field solver again.

```text
model + baryonic field data
    -> field job (expensive; immutable observables.npz)
        -> observation job A (map, beam, mask, uncertainties)
        -> observation job B (different mask or data release)
        -> observation job C (circular-speed table)
```

Each observation job has its own content identity, lifecycle, events,
cancellation, restart recovery, artifact index, and verified downloads. It
references the exact source field job and adds no gravity parameter.

## Frozen acceptance

| Gate | Result |
|---|---:|
| 2D circular-speed score artifact versus integrated field job | byte exact |
| 2D circular-speed prediction CSV versus integrated field job | byte exact |
| 3D resolved velocity-map score artifact versus integrated field job | byte exact |
| 3D resolved velocity-map prediction CSV versus integrated field job | byte exact |
| Field-solver invocations during observation job | 0 |
| Identical field/data/target/worker identity | same job ID |
| Changed observation target or bundle | different job ID |
| Downloaded artifact hashes | all valid |
| Completed-job restart recovery | pass |
| Queued/running cancellation | pass |
| Gravity parameters added by evaluation | 0 |
| Real HTTP upload/queue/poll/download smoke | pass |

The real HTTP smoke retained exactly one source field job while creating and
downloading the separate observation result. Seven artifacts rehashed, four
curve points were scored, and an identical submission returned the cached job
identity.

## API contract

```text
POST /api/v1/observation-evaluation-jobs
GET  /api/v1/observation-evaluation-jobs/{id}
GET  /api/v1/observation-evaluation-jobs/{id}/events
GET  /api/v1/observation-evaluation-jobs/{id}/artifacts
GET  /api/v1/observation-evaluation-jobs/{id}/artifacts/{name}
POST /api/v1/observation-evaluation-jobs/{id}/cancel
```

Example submission:

```json
{
  "schemaVersion": "sigma-observation-evaluation-job-submit/1",
  "fieldJobId": "job_<completed-field-id>",
  "dataUploadId": "upload_<observation-bundle-id>",
  "observationTargets": [
    {
      "schemaVersion": "sigma-observation-target/1",
      "id": "my-resolved-map",
      "kind": "line_of_sight_velocity_field",
      "observable": "massive_tracer_acceleration",
      "centerM": [0, 0, 0],
      "inclinationDeg": 60,
      "handedness": 1,
      "majorCoordinateArrayKey": "major_m",
      "minorCoordinateArrayKey": "minor_m",
      "observedVelocityArrayKey": "velocity_m_s",
      "uncertaintyArrayKey": "uncertainty_m_s",
      "minimumValidPixels": 25,
      "provenance": {"citation": "dataset citation and processing record"},
      "license": {"id": "dataset-license", "redistributionAllowed": false}
    }
  ]
}
```

The source field must be successful and converged. Its manifest, model, field
job, scientific result, observable archive, and content records are checked.
The observation upload must independently pass the existing unit, shape,
provenance, license, byte-hash, and array-content gates.

## Defects the parity gate caught

The first acceptance attempt exposed that field artifacts retained spacing but
not an explicit grid origin. Numerically centered fixtures happened to agree,
but the standalone adapter had to label the origin as inferred. P0732 now makes
the origin part of immutable solved-field geometry.

The first real HTTP attempt also exposed that preflight queued its compact audit
summary instead of the complete validated target. The queue now retains the
summary for reporting and sends the full canonical target to the worker. These
were interface defects, not changes to any gravity formula.

## Honest limits and next work

- Existing multi-system batches still embed observation evaluation in each
  field child. They should be rewired to compose field jobs with cached
  observation jobs.
- The available adapters remain massive-tracer circular-equilibrium mappings.
  P0732 does not add pressure support, non-circular flow, spectral cubes, or
  photon lensing.
- The public Vercel route advertises this schema but intentionally returns a
  worker-not-connected response until durable field storage and isolated
  compute are deployed.
- Local filesystem identity and recovery are reference behavior, not public
  multi-tenant infrastructure.

## Reproduce

```powershell
python scripts/run_p0732_decoupled_observation_evaluation.py
```

The frozen protocol is
`configs/p0732_decoupled_observation_evaluation.json`; deterministic acceptance
artifacts are under `results/p0732_decoupled_observation_evaluation/`.
