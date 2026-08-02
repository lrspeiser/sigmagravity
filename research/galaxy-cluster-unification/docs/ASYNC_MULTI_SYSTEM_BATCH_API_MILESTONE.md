# Formula-independent multi-system batch API milestone

Date: 2026-08-02

## Outcome

The local reference API can now run one confirmed `sigma-field-model/1`
manifest over multiple uploaded or generated Cartesian 2D/3D systems. The
batch layer does not dispatch on a theory name. It checks every source bundle
against the model's declared dimensions, units, fields, and requested
observables, then creates ordinary content-addressed field jobs.

```text
POST /api/v1/batches
GET  /api/v1/batches/{id}
GET  /api/v1/batches/{id}/events
GET  /api/v1/batches/{id}/artifacts
POST /api/v1/batches/{id}/cancel
```

The contract accepts up to 1,000 systems. It freezes one model hash and one
parameter policy for the entire batch. `published_fixed` and
`universal_fixed` execute now. `universal_fit`,
`train_validation_holdout`, `hierarchical`, and `per_object` are explicit
contract states but fail preflight until their fitting procedures exist. The
service never substitutes an undisclosed per-galaxy fit.

Generated galaxy jobs now emit two field-ready SI products:

- surface density in `kg/m^2` for Cartesian 2D manifests;
- volume density in `kg/m^3` for Cartesian 3D manifests.

The galaxy representation and gravity model remain separate. A generated
system can therefore be reused with Newtonian Poisson, AQUAL, QUMOND,
Refracted Gravity, a two-potential theory, or another compatible equation tree.

## Corrected numerical acceptance

An initial commissioning run exposed a diagnostic error: convergence used the
solver's finite-volume stencil, while the reported residual used composed
NumPy gradients. The two operators disagreed at finite resolution. The worker
now evaluates residuals with the exact harmonic-face finite-volume operator
used to assemble the solve, and a job succeeds only when both relative updates
and equation residuals meet their declared tolerances.

The real HTTP acceptance then performed this complete chain:

1. upload the registered DDO101 gas and stellar maps;
2. extract gravity-independent baryonic parameters;
3. generate replay, compact (`0.78x` radial scale), and diffuse (`1.22x`)
   variants on 25 x 25 x 9 volume grids;
4. materialize each volume as an immutable `kg/m^3` field bundle;
5. run one frozen Newtonian Poisson manifest over all three systems;
6. poll the batch and child jobs, download every report artifact, and re-hash
   every byte.

| Acceptance check | Result |
|---|---:|
| batch state | succeeded |
| systems succeeded | 3 / 3 |
| convergence fraction | 1.0 |
| median iterations | 17 |
| worst equation residual | `2.7144e-10` |
| acceptance ceiling | `1e-7` |
| per-object gravity parameters | 0 |
| downloaded deterministic artifacts | 9 / 9 valid hashes |
| observation scores reported | no |
| batch ID | `batch_38275ba0e1bf7b37e5a7f06b` |
| model SHA-256 | `43738f2c7bb3c4e94193763cf46f39b2a47fa852b25d1c063fef00fa7e1aa661` |
| scientific result SHA-256 | `913ff065a693162355e614534573c107479df44431a7bee93651a0907e5c9388` |

The report set contains the submitted batch and model, child-job identities,
per-system CSV, aggregate JSON, failure CSV, HTML report, deterministic LLM
briefing, reproduction command, artifact index, and signed-by-hash manifest.

## What this proves

- A researcher can use one generic contract for actual 2D or 3D numerical
  fields without a theory-specific API function.
- Generated galaxy density can flow directly into the field solver with
  declared SI units and immutable hashes.
- One model and parameter policy can be enforced across many systems.
- Small iterative updates cannot masquerade as a converged equation.
- Reports can be interpreted without an LLM, while a compact briefing is
  available for optional downstream explanation.

## What this does not prove

This run validates numerical execution, orchestration, integrity, and parameter
accounting. It does not compare predicted velocities, lensing, or images with
observations. The three systems are controlled variants of one dwarf and are
not an independent astrophysical holdout. The 25 x 25 x 9 grid is a bounded API
commissioning size, not a production-resolution scientific claim.

The public Vercel endpoint still returns
`production_worker_not_connected`. Durable object storage, a database/queue,
isolated workers, authentication, quotas, retries, and monitoring remain
required for public execution.

Reproduce while the development server is running:

```powershell
npm run smoke:batches
```
