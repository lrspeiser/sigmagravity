# Production research API milestone

Date: 2026-08-03

## Outcome

The durable control-plane semantics are now connected to formula-neutral,
project-scoped API handlers. This is production code with real PostgreSQL
acceptance, but not yet a hosted scientific service: the Horizon3 database,
recurring outbox trigger, and stateless worker remain unconnected.

The authenticated lifecycle is:

```text
POST /api/v1/models
POST /api/v1/data-uploads
PUT  /api/v1/data-uploads/{id}/content
POST /api/v1/jobs
GET  /api/v1/jobs/{id}
GET  /api/v1/jobs/{id}/events
GET  /api/v1/jobs/{id}/artifacts
GET  /api/v1/jobs/{id}/artifacts/{name}
POST /api/v1/jobs/{id}/cancel
```

Typed field, galaxy, observation, inverse-response, and batch collection routes
use the same job repository. Inference and generation aliases are also
published. No route executes researcher code in Vercel.

## Identity and isolation

`production-research-api-v2.sql` adds hashed project credentials and immutable
audit events. A bearer secret is generated once and stored only as SHA-256.
Every model, upload, job, event, artifact, and cancellation query includes the
authenticated project identifier. Job creation checks model and upload
ownership inside the same transaction that applies active-job limits and
creates the job/event/outbox records.

Project limits currently cover:

- active nonterminal jobs;
- registered upload bytes; and
- worker attempts per job.

Per-object parameter policies are accepted only when explicitly disclosed.
The exact policy remains in the immutable worker request and public job record.

## Immutable registration

Model registration accepts the deterministic receipt returned by
`POST /api/v1/models/confirm`. It recomputes the model, document, source-text,
receipt, and acknowledgement identities before storing the canonical model and
receipt as private content-addressed objects.

Upload registration validates the array-bundle manifest and binds its bundle
hash, archive SHA-256, byte count, scientific roles, provenance, and license.
The bounded gateway upload accepts only the exact registered bytes and rehashes
the private object. Archives above the gateway limit are rejected honestly;
private direct-upload tokens are still required for realistic large arrays.

Job submission stores one canonical request object, requires an
`Idempotency-Key`, verifies registered project-owned inputs, creates the job and
outbox atomically, and attempts durable dispatch. The same key and science
returns the same job; the same key with changed science returns a conflict.
Transient publication remains in the outbox. `/api/v1/outbox-dispatch` is a
long-secret-protected retry endpoint, but a recurring production trigger is not
yet connected or marked verified.

## Acceptance evidence

The PostgreSQL integration covers:

- repeatable v1 and v2 migrations and all eleven tables;
- one-time credential generation without raw-token persistence;
- successful authentication and invalid-token rejection;
- exact confirmed-model registration and idempotent replay;
- pending upload registration, wrong-byte rejection, exact-byte finalization,
  and idempotent replay;
- field-job submission, durable queue handoff, ordered events, and
  idempotency conflict;
- database-enforced active-job, upload-byte, and attempt quotas;
- cross-project model, upload, and job invisibility;
- cancellation and immutable mutation audit events; and
- authenticated artifact download with a complete SHA-256 verification.

## Remaining production boundary

Before enabling researcher jobs:

1. An authorized Horizon3 user must accept Neon's Marketplace terms.
2. Provision Neon and run both migrations.
3. Bootstrap the first project credential and store it outside the repository.
4. Connect and verify a recurring outbox trigger.
5. Deploy the stateless scientific worker and its server-only credential.
6. Add private direct uploads for large arrays.
7. Complete an end-to-end hosted job, including retry and worker-restart tests.
8. Add signed result manifests, cache/license policy, monitoring, backups, and
   cost controls before broad access.

Until then the public handlers return
`production_control_plane_not_connected` rather than creating temporary or
in-memory research state.

## Live v0.33 verification

Implementation commit `a8c4ef2b72d1ce2f72c541b1033b22e95c74c475`
passes all 152 hosted tests. GitHub Actions run
<https://github.com/lrspeiser/sigmagravity/actions/runs/30801012875> also built
the non-root Linux worker image and passed real field and galaxy container
acceptance.

Production deployment `dpl_B2jrYAiTfZxGdyWzYXnB4dtQCG5f` is ready at
<https://sigma-gravity-research-simulator-n8vt853cb-horizon3.vercel.app> and
aliased to the stable site. Health and OpenAPI report `0.33.0-preview`; the
guide contains the exact job input, current 503, expected connected lifecycle,
and scientific limits. The public deterministic smoke still reproduces the
175-system catalog and the same content-addressed reference results.

Two queue smokes independently verified deployment identity hash
`b55efdabcc436d6aee6981688284f075166bb3efa82af69f80763e4204cd28d6`
and private acknowledgement hash
`9b7e503b64a9328217a2da2eaf47a515f2f8c036d600a7cff26f009c1f408d48`.
Readiness reports private Blob and Queue connected and consumed, while the
database, recurring outbox scheduler, and stateless worker remain
`not_configured`. A real job submission therefore returns the documented HTTP
503 boundary rather than storing research state outside PostgreSQL.
