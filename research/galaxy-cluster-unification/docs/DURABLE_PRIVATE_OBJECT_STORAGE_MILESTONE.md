# Durable private object-storage milestone

Date: 2026-08-03

## Outcome

The Horizon3 Vercel project now has a private object store for scientific
inputs and artifacts. The hosted application uses a provider adapter rather
than exposing credentials or provider URLs in its public contract.

Every accepted object has:

- a bounded size;
- a validated namespace and extension;
- a pathname derived from its SHA-256 content hash;
- private access;
- overwrite protection and idempotent repeated writes; and
- a full byte download and SHA-256 verification after write and on every read.

The acceptance canary was executed in two independent Node processes. Both
returned the same path
`sigma/v1/objects/deployment-canary/sha256/3b3aa125f6eb843feec8c90f35b70223a00375108a63150d5ad7b1761127017b.json`,
the same 160-byte length, and a successful verified read. No credential is
written to the repository or returned by the readiness endpoint.

## What this enables

- A durable home for immutable array uploads, model manifests, worker inputs,
  predictions, reports, and signed-manifest inputs.
- Stable content identity across gateway and worker processes.
- Deduplication and cache keys that do not depend on a machine filesystem.
- Detection of corrupted or substituted bytes before scientific use.

## What it does not enable

It does not make a submitted simulation job durable. Production execution is
still unavailable because the public deployment has no durable queue, no
transactional job/model metadata database, and no stateless scientific worker
that consumes private objects. The API therefore continues to return HTTP 503
for heavy jobs. This is an intentional fail-closed boundary.

`GET /api/v1/storage-readiness` reports each of these layers separately.

## Remaining production build, in order

1. **Implemented and locally accepted:** Postgres tables and transactions for
   projects, models, uploads, jobs, leases, events, attempts, artifacts, and a
   transactional queue outbox. Provisioning awaits Horizon3 acceptance of the
   Neon Marketplace terms.
2. **Implemented for deployment acceptance:** durable at-least-once Vercel
   Queue delivery with a deployment-bound private canary. Job logic tolerates
   duplicate messages and recovers expired leases without accepting stale
   results.
3. Refactor and deploy the Python worker as a stateless executor: download only
   allow-listed hash-bound inputs, use a disposable local scratch directory,
   upload rehashed artifacts, then atomically finalize metadata.
4. Deploy fixed CPU/memory/time resource classes. Keep advanced plug-ins in
   separate, single-use, network-disabled containers with read-only inputs.
5. Add authenticated projects, per-project isolation, rate and storage quotas,
   cancellation, retry ownership, cache policy, dataset-license enforcement,
   audit logs, monitoring, backups, and cost limits.
6. Sign result manifests with model, data, solver, seed, container, parameter
   policy, artifact hashes, and the full reproduction command.
7. Run restart, duplicate-delivery, cancellation, timeout, corruption,
   partial-upload, worker-crash, authorization, and cross-project leakage
   acceptance through the public Vercel alias.

## Remaining scientific build for Sigma Gravity

1. Express one exact Sigma Gravity law as a confirmed manifest with a small
   set of universal constants, explicit photon/matter coupling, units,
   boundaries, and a Solar-System limit.
2. Register morphology-diverse baryonic observations independently of the
   tested gravity law, including maps/cubes, uncertainty, PSF/beam, WCS,
   geometry, masks, provenance, license, and hashes.
3. Infer and calibrate baryonic 2D-to-3D ensembles without velocity, lensing,
   or halo targets; reject collapsed or uncalibrated posterior weights.
4. Freeze universal constants on a named development sample. Do not refit them
   per galaxy or cluster.
5. Predict held-out galaxy velocities and raw cluster image positions, shear,
   magnification, or time delays using the same constants.
6. Compare baryons-only GR/Newtonian, fixed MOND/RAR, and published dark-matter
   models with uncertainty and all gravity, nuisance, hierarchical, and
   per-object parameter counts disclosed.
7. Require Solar-System, resolution, boundary, conservation, stability, and
   blind-holdout gates before treating the formula as viable.

## Remaining scientific build for learning from dark-matter clouds

Dark-matter cloud maps are conditional model products, not direct images of
matter. They may be used to discover candidate response structure, never as
the final validation target.

1. Register several independent lens-model teams' halo posteriors and keep
   them role-separated from baryons and raw observations.
2. Run the shared inverse baryon-to-response engine across systems and
   baryonic realizations, with angular shuffle, phase scramble, system
   permutation, target shuffle, and missing-baryon controls.
3. Quantify which inferred offsets, anisotropies, scales, and topology survive
   lens method and baryonic uncertainty; report null spaces and compatible
   alternatives.
4. Compress any repeatable many-cell response into a small analytic forward
   law based only on baryonic density, gradients, curvature, topology, or a
   declared environmental field. Count every retained coefficient.
5. Freeze that law, remove every halo target, and predict untouched raw
   lensing and galaxy observations. A failure here rejects the interpretation
   even if the reconstructed response resembles a published halo.
