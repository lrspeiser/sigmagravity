# Generic 2D/3D field-job milestone

Date: 2026-08-02

## Outcome

The first content-addressed scientific job now runs locally without dispatching
on a theory name. A job binds:

- the exact `sigma-field-model/1` computational hash;
- a verified array bundle whose individual arrays have shape, dtype, unit,
  rank, provenance, license, and content hashes;
- coordinate system, spacing, boundaries, solver, seed, and requested
  observables;
- universal versus per-object parameter policy; and
- worker engine version and source hash.

Runtime timestamps, CPU time, and memory measurements are deliberately outside
the scientific result hash. Two clean executions can therefore prove that the
field and diagnostics agree even though they do not take exactly the same
number of milliseconds.

## End-to-end known-answer result

A 25-by-25 Dirichlet Poisson job used

`u(x,y)=sin(pi*x) sin(pi*y)` and `laplacian(u)=-2*pi^2*u`.

The CLI packaged the source into an immutable array bundle, executed the generic
worker, and independently rechecked every artifact hash.

| Check | Result |
|---|---:|
| state | succeeded |
| converged | true |
| nonlinear/global passes | 2 |
| relative L2 field error | 0.00142912 (0.143%) |
| artifact hashes verified | 8 of 8 |
| per-object parameters | 0 |
| input bundle SHA-256 | `626f75d11176553dac44e8eeed8eb0e318f471dd9971a96b542b9b1d2c76ad75` |
| job ID | `fieldjob_075e66ccae0d17624cc6d837` |
| scientific result SHA-256 | `243c67d85880f42a5497d36c188f0fc6efbeab5534b956e7133f2b400a3e5b95` |

The exact hashes include the worker source. They will correctly change when the
numerical implementation changes.

## Artifacts

Each successful run writes:

- `model.json`
- `input_bundle.json`
- `job.json`
- `scientific_result.json`
- `fields.npz`
- `observables.npz`
- `residual_history.csv`
- `resource_log.json`
- `artifact_index.json`
- `manifest.json`

`scientific_result.json` carries content hashes for every field and observable.
`artifact_index.json` hashes the concrete files. `manifest.json` identifies the
environment and reproduction command. A nonconverged solve is retained as
`failed_nonconvergence`; it is not silently converted into a result.

## API boundary

The hosted gateway can now preflight the same model and array metadata through
`POST /api/v1/field-jobs/prepare`. It checks geometry, required arrays, units,
shape, boundaries, observables, parameter counts, bundle hash, and estimated
resource class. It still reports `array_bytes_not_uploaded` and
`generic_scientific_worker_not_connected`, because the upload store and queue
do not yet exist.

## Container boundary

`worker/Dockerfile` pins Python 3.13.5, NumPy 2.2.6, SciPy 1.16.1, and RFC 8785
canonicalization. It runs as non-root UID/GID 65532. Production invocations are
specified with no network, a read-only root, bounded CPU/memory, and one writable
artifact mount. Docker and Podman are not installed on the current development
machine, so the image definition and dependency lock are checked, but an actual
container build remains a CI/deployment-host acceptance item.

## What this does and does not prove

This proves a formula-independent, reproducible numerical job can move from a
model and arrays to verifiable field artifacts. It does not prove that the
formula describes a real galaxy, that an uploaded map is correctly registered,
or that the worker has been securely deployed. Those require the resolved-data,
observation-forward, upload, queue, and external-replication stages.
