import test from "node:test";
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { PGlite } from "@electric-sql/pglite";
import { canonicalJson, sha256 } from "../lib/canonical.mjs";
import { privateBlobReferenceFor } from "../lib/private-blob-store.mjs";
import { ControlPlaneError, ProductionControlPlane } from "../lib/production-control-plane.mjs";

const PROJECT_ID = "project_0123456789abcdef01234567";

function reference(namespace, value, extension = "json", mediaType = "application/json") {
  return privateBlobReferenceFor({
    namespace,
    bytes: Buffer.from(typeof value === "string" ? value : `${canonicalJson(value)}\n`, "utf8"),
    extension,
    mediaType,
  });
}

async function fixture(t) {
  const database = new PGlite();
  t.after(() => database.close());
  const migration = await readFile(new URL("../sql/production-control-plane-v1.sql", import.meta.url), "utf8");
  await database.exec(migration);
  const apiMigration = await readFile(new URL("../sql/production-research-api-v2.sql", import.meta.url), "utf8");
  await database.exec(apiMigration);
  const adapter = {
    query: (text, parameters = []) => database.query(text, parameters),
    transaction: (callback) => database.transaction(
      (transaction) => callback({ query: (text, parameters = []) => transaction.query(text, parameters) }),
    ),
  };
  const control = new ProductionControlPlane({ database: adapter });
  await control.createProject({
    projectId: PROJECT_ID,
    slug: "acceptance",
    displayName: "Control-plane acceptance",
  });
  return { database, adapter, control };
}

function jobInput(suffix = "one", overrides = {}) {
  const request = {
    schemaVersion: "sigma-production-request-fixture/1",
    suffix,
  };
  const requestObjectReference = reference("job-request", request);
  return {
    projectId: PROJECT_ID,
    jobType: "field",
    idempotencyKey: `field-fixture-${suffix}`,
    requestSha256: requestObjectReference.sha256,
    requestObjectReference,
    parameterPolicy: { kind: "published_fixed", universalGravityParameters: 1, perObjectGravityParameters: 0 },
    maxAttempts: 3,
    ...overrides,
  };
}

test("migration is repeatable and creates the transactional control-plane tables", async (t) => {
  const { database } = await fixture(t);
  const migration = await readFile(new URL("../sql/production-control-plane-v1.sql", import.meta.url), "utf8");
  await database.exec(migration);
  const apiMigration = await readFile(new URL("../sql/production-research-api-v2.sql", import.meta.url), "utf8");
  await database.exec(apiMigration);
  const tables = await database.query(
    "SELECT tablename FROM pg_tables WHERE schemaname = 'public' AND tablename LIKE 'sigma_%' ORDER BY tablename",
  );
  assert.deepEqual(tables.rows.map((row) => row.tablename), [
    "sigma_audit_events",
    "sigma_job_artifacts",
    "sigma_job_attempts",
    "sigma_job_events",
    "sigma_jobs",
    "sigma_models",
    "sigma_outbox",
    "sigma_project_api_keys",
    "sigma_projects",
    "sigma_schema_migrations",
    "sigma_uploads",
  ]);
  const migrationRows = await database.query("SELECT migration_id FROM sigma_schema_migrations ORDER BY migration_id");
  assert.deepEqual(migrationRows.rows, [
    { migration_id: "production-control-plane-v1" },
    { migration_id: "production-research-api-v2" },
  ]);
});

test("job and outbox creation are atomic and idempotency cannot change science", async (t) => {
  const { database, control } = await fixture(t);
  const input = jobInput();
  const first = await control.createJob(input);
  assert.equal(first.created, true);
  assert.equal(first.job.state, "dispatch_pending");
  const second = await control.createJob(input);
  assert.equal(second.created, false);
  assert.equal(second.job.id, first.job.id);
  await assert.rejects(
    control.createJob(jobInput("changed", { idempotencyKey: input.idempotencyKey })),
    (error) => error instanceof ControlPlaneError && error.code === "idempotency_conflict",
  );
  const counts = await database.query(
    "SELECT (SELECT count(*)::int FROM sigma_jobs) AS jobs, (SELECT count(*)::int FROM sigma_outbox) AS outbox",
  );
  assert.deepEqual(counts.rows, [{ jobs: 1, outbox: 1 }]);
  assert.deepEqual((await control.getEvents(PROJECT_ID, first.job.id)).map((event) => event.type), ["accepted"]);
});

test("job mutation rolls back when its project audit actor is invalid", async (t) => {
  const { database, control } = await fixture(t);
  await assert.rejects(
    control.createJob(jobInput("invalid-audit", {
      auditCredentialId: "key_ffffffffffffffffffffffff",
    })),
    (error) => error instanceof ControlPlaneError && error.code === "invalid_audit_actor",
  );
  const counts = await database.query(
    `SELECT
       (SELECT count(*)::int FROM sigma_jobs) AS jobs,
       (SELECT count(*)::int FROM sigma_outbox) AS outbox,
       (SELECT count(*)::int FROM sigma_audit_events) AS audits`,
  );
  assert.deepEqual(counts.rows, [{ jobs: 0, outbox: 0, audits: 0 }]);
});

test("transactional outbox retries safely and marks the job queued only after publish", async (t) => {
  const { database, control } = await fixture(t);
  const created = await control.createJob(jobInput("dispatch"));
  const failed = await control.dispatchAvailable({ async send() { throw new Error("transient queue outage"); } });
  assert.deepEqual(failed.map((value) => value.state), ["retry_pending"]);
  assert.equal((await control.getJob(PROJECT_ID, created.job.id)).state, "dispatch_pending");
  await database.query("UPDATE sigma_outbox SET next_attempt_at = transaction_timestamp()");
  const calls = [];
  const published = await control.dispatchAvailable({
    async send(topic, payload, options) {
      calls.push({ topic, payload, options });
      return { messageId: "msg_dispatch_1" };
    },
  });
  assert.deepEqual(published.map((value) => value.state), ["published"]);
  assert.equal(calls.length, 1);
  assert.equal(calls[0].payload.jobId, created.job.id);
  assert.equal(calls[0].options.idempotencyKey, `job-dispatch-${created.job.id}`);
  assert.equal((await control.getJob(PROJECT_ID, created.job.id)).state, "queued");
  assert.deepEqual((await control.getEvents(PROJECT_ID, created.job.id)).map((event) => event.type), [
    "accepted",
    "queued",
  ]);
  assert.deepEqual(await control.dispatchAvailable({ async send() { throw new Error("must not resend"); } }), []);
});

test("at-least-once deliveries use leases, retry attempts, and idempotent finalization", async (t) => {
  const { control } = await fixture(t);
  const created = await control.createJob(jobInput("lifecycle"));
  await control.dispatchAvailable({ async send() { return { messageId: "msg_lifecycle" }; } });
  const firstClaim = await control.claimJob({
    jobId: created.job.id,
    messageId: "msg_lifecycle",
    deliveryCount: 1,
    workerIdentity: "container-image@sha256:test-one",
  });
  assert.equal(firstClaim.claimed, true);
  assert.equal(firstClaim.attempt, 1);
  const duplicate = await control.claimJob({
    jobId: created.job.id,
    messageId: "msg_lifecycle",
    deliveryCount: 2,
    workerIdentity: "container-image@sha256:test-two",
  });
  assert.equal(duplicate.claimed, false);
  assert.equal(duplicate.reason, "leased");
  const failed = await control.failJob({
    jobId: created.job.id,
    leaseToken: firstClaim.leaseToken,
    error: Object.assign(new Error("worker restarted"), { code: "worker_crash" }),
    retryable: true,
  });
  assert.equal(failed.shouldRetry, true);
  assert.equal(failed.state, "queued");
  const secondClaim = await control.claimJob({
    jobId: created.job.id,
    messageId: "msg_lifecycle",
    deliveryCount: 2,
    workerIdentity: "container-image@sha256:test-two",
  });
  assert.equal(secondClaim.claimed, true);
  assert.equal(secondClaim.attempt, 2);
  const artifactReference = reference("field-artifact", "verified field\n", "npz", "application/x-npz");
  const manifestReference = reference("result-manifest", { jobId: created.job.id, artifacts: [artifactReference.sha256] });
  const completed = await control.completeJob({
    jobId: created.job.id,
    leaseToken: secondClaim.leaseToken,
    resultManifestReference: manifestReference,
    artifacts: [{
      name: "field.npz",
      objectReference: artifactReference,
      sha256: artifactReference.sha256,
      bytes: artifactReference.bytes,
      mediaType: artifactReference.mediaType,
    }],
  });
  assert.equal(completed.state, "succeeded");
  const repeated = await control.completeJob({
    jobId: created.job.id,
    leaseToken: secondClaim.leaseToken,
    resultManifestReference: manifestReference,
    artifacts: [{
      name: "field.npz",
      objectReference: artifactReference,
      sha256: artifactReference.sha256,
      bytes: artifactReference.bytes,
      mediaType: artifactReference.mediaType,
    }],
  });
  assert.equal(repeated.idempotent, true);
  assert.deepEqual((await control.getArtifacts(PROJECT_ID, created.job.id)).map((value) => value.name), ["field.npz"]);
  assert.deepEqual((await control.getEvents(PROJECT_ID, created.job.id)).map((event) => event.type), [
    "accepted",
    "queued",
    "worker_claimed",
    "retry_scheduled",
    "worker_claimed",
    "succeeded",
  ]);
  const redelivery = await control.claimJob({
    jobId: created.job.id,
    messageId: "msg_lifecycle",
    deliveryCount: 3,
    workerIdentity: "container-image@sha256:test-three",
  });
  assert.equal(redelivery.reason, "terminal");
});

test("expired database leases recover and stale workers cannot publish", async (t) => {
  const { database, control } = await fixture(t);
  const created = await control.createJob(jobInput("expired"));
  await control.dispatchAvailable({ async send() { return { messageId: "msg_expired" }; } });
  const oldClaim = await control.claimJob({
    jobId: created.job.id,
    messageId: "msg_expired",
    deliveryCount: 1,
    workerIdentity: "old-worker",
  });
  await database.query(
    "UPDATE sigma_jobs SET lease_expires_at = transaction_timestamp() - interval '1 second' WHERE job_id = $1",
    [created.job.id],
  );
  const recovered = await control.claimJob({
    jobId: created.job.id,
    messageId: "msg_expired",
    deliveryCount: 2,
    workerIdentity: "replacement-worker",
  });
  assert.equal(recovered.claimed, true);
  assert.equal(recovered.attempt, 2);
  await assert.rejects(
    control.failJob({
      jobId: created.job.id,
      leaseToken: oldClaim.leaseToken,
      error: new Error("late stale result"),
    }),
    (error) => error instanceof ControlPlaneError && error.code === "lost_lease",
  );
  const attempts = await database.query(
    "SELECT attempt, state FROM sigma_job_attempts WHERE job_id = $1 ORDER BY attempt",
    [created.job.id],
  );
  assert.deepEqual(attempts.rows, [{ attempt: 1, state: "lease_expired" }, { attempt: 2, state: "running" }]);
});

test("cancellation is terminal before start and wins a race with result publication", async (t) => {
  const { control } = await fixture(t);
  const queued = await control.createJob(jobInput("cancel-queued"));
  const cancelledQueued = await control.requestCancellation(PROJECT_ID, queued.job.id);
  assert.equal(cancelledQueued.state, "cancelled");
  const queuedDelivery = await control.claimJob({
    jobId: queued.job.id,
    messageId: "msg_cancelled",
    deliveryCount: 1,
    workerIdentity: "worker-after-cancel",
  });
  assert.equal(queuedDelivery.reason, "terminal");

  const running = await control.createJob(jobInput("cancel-running"));
  await control.dispatchAvailable({ async send() { return { messageId: "msg_running" }; } });
  const claim = await control.claimJob({
    jobId: running.job.id,
    messageId: "msg_running",
    deliveryCount: 1,
    workerIdentity: "worker-during-cancel",
  });
  const requested = await control.requestCancellation(PROJECT_ID, running.job.id);
  assert.equal(requested.state, "cancel_requested");
  const artifactReference = reference("cancel-artifact", "must not publish\n");
  const completion = await control.completeJob({
    jobId: running.job.id,
    leaseToken: claim.leaseToken,
    resultManifestReference: artifactReference,
    artifacts: [{
      name: "result.json",
      objectReference: artifactReference,
      sha256: artifactReference.sha256,
      bytes: artifactReference.bytes,
      mediaType: artifactReference.mediaType,
    }],
  });
  assert.equal(completion.state, "cancelled");
  assert.deepEqual(await control.getArtifacts(PROJECT_ID, running.job.id), []);
});

test("database constraints reject cross-project and malformed durable state", async (t) => {
  const { database, control } = await fixture(t);
  const created = await control.createJob(jobInput("constraints"));
  await assert.rejects(
    database.query("UPDATE sigma_jobs SET lease_token = 'orphan' WHERE job_id = $1", [created.job.id]),
    /violates check constraint/i,
  );
  await assert.rejects(
    database.query(
      "INSERT INTO sigma_job_artifacts(job_id, name, object_ref, sha256, bytes, media_type) VALUES ($1, '../secret', '{}'::jsonb, $2, 1, 'text/plain')",
      [created.job.id, sha256("bad")],
    ),
    /violates check constraint/i,
  );
});
