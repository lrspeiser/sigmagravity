# Production control-plane milestone

Date: 2026-08-03

## Architecture now implemented

The production job path is split into independently testable layers:

1. The gateway stores the exact confirmed request as a private
   content-addressed object.
2. One Postgres transaction creates the job, its first event, and an outbox
   record. An idempotency key cannot be rebound to different scientific input.
3. The outbox publishes only a small hash-bound job message to Vercel Queue.
   If publication succeeds but its database acknowledgement is lost, repeating
   the same outbox item uses the same queue idempotency key.
4. A private queue consumer claims a database lease. Duplicate deliveries do
   not run a second active attempt; an expired lease can be recovered.
5. The stateless worker receives job identity and private object references,
   never inline array or artifact bytes. Its origin and long bearer secret are
   server-only and fail closed when incomplete.
6. Before success, the finalizer downloads and SHA-256 verifies the result
   manifest and every declared artifact, checks envelope/index agreement, and
   atomically publishes the artifact rows and terminal event.
7. A stale worker cannot publish after losing its lease. Cancellation requested
   during execution wins over result publication.

The queue uses at-least-once rather than exactly-once delivery. Correctness is
therefore enforced by database idempotency and leases, not by assuming a
message is unique.

## Transactional schema

`sql/production-control-plane-v1.sql` creates nine idempotent tables for:

- schema migrations;
- projects;
- confirmed models;
- immutable uploads;
- jobs;
- ordered events;
- worker attempts and leases;
- immutable artifacts; and
- the transactional queue outbox.

Database constraints enforce identifier grammar, state values, SHA-256 form,
private object-reference schemas, bounded attempts, artifact names, lease
pairing, and project/job idempotency.

## Acceptance evidence

The embedded PostgreSQL acceptance runs the actual migration twice and tests:

- atomic job/outbox creation;
- same-key/same-science replay and same-key/different-science rejection;
- transient queue-publication failure and deterministic retry;
- duplicate queue delivery while a lease is active;
- retryable worker failure and a later successful attempt;
- expired lease recovery and stale-worker rejection;
- cancellation before execution and cancellation/result races;
- result manifest, artifact index, pathname, byte count, and SHA agreement;
- complete private-object rehash before terminal success; and
- database rejection of malformed durable state and traversal names.

The Vercel build output contains two private queue-trigger functions:
`sigma-control-plane-canary-v1` and `sigma-control-plane-jobs-v1`. The canary
consumer persists a deterministic acknowledgement bound to the exact Vercel
deployment identity. This proves publish, private invocation, consumption, and
private storage separately from scientific execution.

## Deployment boundary

The production project currently cannot provision its required database until
an authorized Horizon3 user accepts Neon's Marketplace terms at:

<https://vercel.com/horizon3/~/integrations/accept-terms/neon?source=cli>

The requested resource is the free Neon plan in `iad1`, with built-in Neon Auth
disabled and environment variables prefixed `SIGMA_`. No terms were accepted
on the user's behalf.

Until Postgres is connected and migrated and the stateless container is
deployed, all public heavy scientific routes must continue to return HTTP 503.
The live queue canary is infrastructure evidence, not permission to enqueue
researcher jobs and not a scientific result.

## Live v0.32 verification

Production deployment `dpl_FL6AriDMSpit9bxZCd33VSgpYxuZ` serves
`0.32.0-preview` at
<https://sigma-gravity-research-simulator-five.vercel.app>. GitHub Actions run
<https://github.com/lrspeiser/sigmagravity/actions/runs/30798650630> passed the
real non-root field-and-galaxy container acceptance for implementation commit
`4e6a2d4028b12e83f730aff00b421f1ca7eca55c`.

The production queue canary and an independent repeat both verified deployment
hash `c086d4e4d4868fc1d27e939ad7ae8c638daff1737faa7e8248968bd692f82aa4` and
private acknowledgement hash
`ab7746edc88e31a6dd358ea2dbe340bc3f8290ba66beaa05eb077fe84c308c2e`.
Storage readiness reports `durable_storage_and_queue_connected`; database and
stateless-worker states remain `not_configured`, `productionExecution` remains
`not_ready`, and a real field-job submission returns HTTP 503.
