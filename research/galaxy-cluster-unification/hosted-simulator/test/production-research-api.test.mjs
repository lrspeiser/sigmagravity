import test from "node:test";
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { PGlite } from "@electric-sql/pglite";
import { canonicalJson, sha256 } from "../lib/canonical.mjs";
import { confirmFieldModel, validateFieldModel } from "../lib/field-model.mjs";
import {
  authenticateProjectRequest,
  createProjectCredential,
  ProductionAuthError,
} from "../lib/production-auth.mjs";
import {
  handleProductionJob,
  handleProductionJobs,
  handleProductionModels,
  handleProductionUpload,
  handleProductionUploads,
} from "../lib/production-api-handler.mjs";
import { contentSha256, privateBlobReferenceFor, validatePrivateBlobReference } from "../lib/private-blob-store.mjs";
import { ControlPlaneError, ProductionControlPlane } from "../lib/production-control-plane.mjs";
import { readProductionDatabaseReadiness } from "../lib/production-database.mjs";
import { ProductionResearchService } from "../lib/production-research-service.mjs";

const PROJECT_A = "project_aaaaaaaaaaaaaaaaaaaaaaaa";
const PROJECT_B = "project_bbbbbbbbbbbbbbbbbbbbbbbb";

class MemoryPrivateStore {
  constructor() {
    this.objects = new Map();
  }

  async putImmutable({ namespace, bytes, mediaType, extension }) {
    bytes = Buffer.from(bytes);
    const reference = privateBlobReferenceFor({ namespace, bytes, mediaType, extension });
    const existing = this.objects.get(reference.pathname);
    if (existing && !existing.equals(bytes)) throw new Error("immutable object collision");
    this.objects.set(reference.pathname, bytes);
    return reference;
  }

  async getVerified(reference) {
    reference = validatePrivateBlobReference(reference);
    const bytes = this.objects.get(reference.pathname);
    if (!bytes || bytes.length !== reference.bytes || contentSha256(bytes) !== reference.sha256) {
      throw new Error("private object failed verification");
    }
    return Buffer.from(bytes);
  }
}

function response() {
  return {
    headers: {},
    statusCode: 200,
    body: null,
    setHeader(name, value) { this.headers[name] = value; },
    status(code) { this.statusCode = code; return this; },
    json(value) { this.body = value; return this; },
    send(value) { this.body = value; return this; },
    end(value) { if (value !== undefined) this.body = value; return this; },
  };
}

async function call(handler, request, options) {
  const output = response();
  await handler(request, output, options);
  return output;
}

async function fixture(t) {
  const database = new PGlite();
  t.after(() => database.close());
  for (const migration of ["production-control-plane-v1", "production-research-api-v2"]) {
    await database.exec(await readFile(new URL(`../sql/${migration}.sql`, import.meta.url), "utf8"));
  }
  const adapter = {
    query: (text, parameters = []) => database.query(text, parameters),
    transaction: (callback) => database.transaction(
      (transaction) => callback({ query: (text, parameters = []) => transaction.query(text, parameters) }),
    ),
  };
  const controlPlane = new ProductionControlPlane({ database: adapter });
  await controlPlane.createProject({ projectId: PROJECT_A, slug: "project-a", displayName: "Project A" });
  await controlPlane.createProject({ projectId: PROJECT_B, slug: "project-b", displayName: "Project B" });
  const keyA = await createProjectCredential({ database: adapter, projectId: PROJECT_A, label: "A acceptance" });
  const keyB = await createProjectCredential({ database: adapter, projectId: PROJECT_B, label: "B acceptance" });
  const requestFor = (token, overrides = {}) => ({
    method: "GET",
    headers: { authorization: `Bearer ${token}` },
    query: {},
    ...overrides,
  });
  const authA = await authenticateProjectRequest(requestFor(keyA.token), adapter);
  const authB = await authenticateProjectRequest(requestFor(keyB.token), adapter);
  const store = new MemoryPrivateStore();
  const sent = [];
  const publisher = {
    async send(topic, payload, options) {
      sent.push({ topic, payload, options });
      return { messageId: `message-${sent.length}` };
    },
  };
  const service = new ProductionResearchService({ controlPlane, store, publisher });
  const runtime = {
    database: adapter,
    controlPlane,
    store,
    publisher,
    service,
    jobSubmissionReady: true,
    executionComponents: { statelessWorker: "configured", outboxScheduler: "verified" },
  };
  return { database, adapter, controlPlane, keyA, keyB, authA, authB, store, sent, service, runtime, requestFor };
}

async function confirmedModelReceipt() {
  const model = JSON.parse(await readFile(new URL("../examples/models/newtonian-poisson.json", import.meta.url), "utf8"));
  const validation = validateFieldModel(model);
  return confirmFieldModel(model, {
    expectedModelSha256: validation.modelSha256,
    acknowledgement: validation.confirmation.acknowledgement,
  });
}

function uploadFixture(bytes = Buffer.from("bounded-npz-fixture")) {
  const core = {
    schemaVersion: "sigma-array-bundle/1",
    geometry: {
      coordinateSystem: "cartesian_2d",
      dimensions: 2,
      spacing: [1, 1],
      lengthUnit: "m",
    },
    arrays: [{
      key: "baryon_density",
      npzKey: "baryon_density",
      unit: "kg/m^3",
      rank: "scalar",
      role: "source",
      scientificRole: "baryonic_input",
      dtype: "<f8",
      shape: [5, 5],
      elementCount: 25,
      contentSha256: "2".repeat(64),
    }],
    provenance: { kind: "acceptance_fixture" },
    license: { id: "CC0-1.0", redistributionAllowed: true },
  };
  const manifest = { ...core, bundleSha256: sha256(core) };
  return {
    bytes,
    registration: {
      schemaVersion: "sigma-production-upload-registration/1",
      manifest,
      archiveSha256: contentSha256(bytes),
      archiveBytes: bytes.length,
    },
  };
}

test("project credentials are one-time secrets and authentication is project-scoped", async (t) => {
  const { adapter, keyA, authA, requestFor } = await fixture(t);
  assert.deepEqual(await readProductionDatabaseReadiness({ database: adapter }), {
    state: "verified_migrated",
    migrations: ["production-control-plane-v1", "production-research-api-v2"],
  });
  assert.equal(authA.project.id, PROJECT_A);
  assert.match(keyA.token, /^sgp_[A-Za-z0-9_-]{43}$/);
  const stored = await adapter.query(
    "SELECT credential_id, token_sha256 FROM sigma_project_api_keys WHERE project_id = $1",
    [PROJECT_A],
  );
  assert.equal(stored.rows.length, 1);
  assert.equal(stored.rows[0].token_sha256, sha256(keyA.token));
  assert.equal(JSON.stringify(stored.rows).includes(keyA.token), false);
  await assert.rejects(
    authenticateProjectRequest(requestFor(`sgp_${"A".repeat(43)}`), adapter),
    (error) => error instanceof ProductionAuthError && error.code === "invalid_credential",
  );
});

test("confirmed model, immutable upload, queued job, cancellation, and audit pass end to end", async (t) => {
  const { adapter, service, authA, authB, sent } = await fixture(t);
  const receipt = await confirmedModelReceipt();
  const model = await service.registerModel(authA, {
    schemaVersion: "sigma-production-model-registration/1",
    confirmationReceipt: receipt,
  });
  assert.equal(model.created, true);
  assert.equal((await service.registerModel(authA, {
    schemaVersion: "sigma-production-model-registration/1",
    confirmationReceipt: receipt,
  })).created, false);

  const uploadFixtureValue = uploadFixture();
  const registration = await service.registerUpload(authA, uploadFixtureValue.registration);
  assert.equal(registration.upload.state, "pending");
  await assert.rejects(
    service.putUploadContent(authA, registration.upload.id, Buffer.from("wrong")),
    (error) => error instanceof ControlPlaneError && error.code === "upload_size_mismatch",
  );
  const finalized = await service.putUploadContent(
    authA,
    registration.upload.id,
    uploadFixtureValue.bytes,
    { suppliedSha256: uploadFixtureValue.registration.archiveSha256 },
  );
  assert.equal(finalized.upload.state, "ready");
  assert.equal((await service.putUploadContent(authA, registration.upload.id, uploadFixtureValue.bytes)).finalized, false);

  const payload = {
    schemaVersion: "sigma-production-job-submit/1",
    jobType: "field",
    modelSha256: receipt.modelSha256,
    dataUploadId: registration.upload.id,
    parameterPolicy: { mode: "published_fixed" },
    request: { schemaVersion: "sigma-field-job-request/1", requestedObservables: ["massive_tracer_acceleration"] },
    maxAttempts: 3,
  };
  const submitted = await service.submitJob(authA, payload, { idempotencyKey: "field-acceptance-001" });
  assert.equal(submitted.created, true);
  assert.equal(submitted.job.state, "queued");
  assert.equal(sent.length, 1);
  assert.equal(sent[0].payload.requestSha256, submitted.job.requestSha256);
  const replay = await service.submitJob(authA, payload, { idempotencyKey: "field-acceptance-001" });
  assert.equal(replay.created, false);
  assert.equal(replay.job.id, submitted.job.id);
  assert.equal(sent.length, 1);
  await assert.rejects(
    service.submitJob(authA, { ...payload, request: { changed: true } }, { idempotencyKey: "field-acceptance-001" }),
    (error) => error instanceof ControlPlaneError && error.code === "idempotency_conflict",
  );
  assert.deepEqual(
    (await service.getEvents(authA, submitted.job.id)).map((event) => event.type),
    ["accepted", "queued"],
  );
  assert.equal((await service.cancelJob(authA, submitted.job.id)).state, "cancelled");

  for (const operation of [
    () => service.getModel(authB, receipt.modelSha256),
    () => service.getUpload(authB, registration.upload.id),
    () => service.getJob(authB, submitted.job.id),
  ]) {
    await assert.rejects(operation(), (error) => error instanceof ControlPlaneError && error.code.startsWith("unknown_"));
  }
  const audits = await adapter.query(
    "SELECT action FROM sigma_audit_events WHERE project_id = $1 ORDER BY audit_id",
    [PROJECT_A],
  );
  assert.deepEqual(audits.rows.map((row) => row.action), [
    "model.registered",
    "upload.registered",
    "upload.finalized",
    "job.submitted",
    "job.cancellation_requested",
  ]);
});

test("active-job and upload quotas are enforced inside PostgreSQL transactions", async (t) => {
  const { adapter, service, authA } = await fixture(t);
  await adapter.query(
    "UPDATE sigma_projects SET max_active_jobs = 1, max_upload_bytes = 8 WHERE project_id = $1",
    [PROJECT_A],
  );
  const refreshed = await authenticateProjectRequest({
    headers: { authorization: `Bearer ${(await createProjectCredential({ database: adapter, projectId: PROJECT_A, label: "quota" })).token}` },
  }, adapter);
  const tooLarge = uploadFixture(Buffer.from("123456789"));
  await assert.rejects(
    service.registerUpload(refreshed, tooLarge.registration),
    (error) => error instanceof ControlPlaneError && error.code === "upload_quota_exceeded",
  );
  const requestReference = await service.store.putImmutable({
    namespace: "job-request-quota",
    bytes: Buffer.from("{}"),
    mediaType: "application/json",
    extension: "json",
  });
  await service.controlPlane.createJob({
    projectId: PROJECT_A,
    jobType: "galaxy",
    idempotencyKey: "quota-job-one",
    requestSha256: requestReference.sha256,
    requestObjectReference: requestReference,
    parameterPolicy: { mode: "published_fixed" },
    maxAttempts: 2,
  });
  await assert.rejects(
    service.controlPlane.createJob({
      projectId: PROJECT_A,
      jobType: "galaxy",
      idempotencyKey: "quota-job-two",
      requestSha256: requestReference.sha256,
      requestObjectReference: requestReference,
      parameterPolicy: { mode: "published_fixed" },
      maxAttempts: 2,
    }),
    (error) => error instanceof ControlPlaneError && error.code === "active_job_quota_exceeded",
  );
});

test("HTTP handlers require bearer auth and preserve immutable artifact bytes", async (t) => {
  const { runtime, requestFor, keyA, authA, service, controlPlane, store } = await fixture(t);
  const runtimeFactory = () => runtime;
  const unauthenticated = await call(handleProductionModels, { method: "GET", headers: {}, query: {} }, { runtimeFactory });
  assert.equal(unauthenticated.statusCode, 401);
  assert.equal(unauthenticated.body.error, "authentication_required");
  const disabledSubmission = await call(handleProductionJobs, requestFor(keyA.token, {
    method: "POST",
    headers: { authorization: `Bearer ${keyA.token}`, "idempotency-key": "disabled-job-001" },
    body: {},
  }), {
    runtimeFactory: () => ({
      ...runtime,
      jobSubmissionReady: false,
      executionComponents: { statelessWorker: "not_configured", outboxScheduler: "not_configured" },
    }),
  });
  assert.equal(disabledSubmission.statusCode, 503);
  assert.equal(disabledSubmission.body.error, "production_execution_not_ready");

  const uploadFixtureValue = uploadFixture();
  const registered = await call(handleProductionUploads, requestFor(keyA.token, {
    method: "POST",
    body: uploadFixtureValue.registration,
  }), { runtimeFactory });
  assert.equal(registered.statusCode, 201);
  const uploaded = await call(handleProductionUpload, requestFor(keyA.token, {
    method: "PUT",
    query: { id: registered.body.upload.id, resource: "content" },
    headers: {
      authorization: `Bearer ${keyA.token}`,
      "content-length": String(uploadFixtureValue.bytes.length),
      "x-content-sha256": uploadFixtureValue.registration.archiveSha256,
    },
    body: uploadFixtureValue.bytes,
  }), { runtimeFactory });
  assert.equal(uploaded.statusCode, 200);
  assert.equal(uploaded.body.upload.state, "ready");

  const submission = await call(handleProductionJobs, requestFor(keyA.token, {
    method: "POST",
    headers: { authorization: `Bearer ${keyA.token}`, "idempotency-key": "http-galaxy-job-01" },
    body: {
      schemaVersion: "sigma-production-job-submit/1",
      jobType: "galaxy",
      dataUploadId: registered.body.upload.id,
      parameterPolicy: { mode: "published_fixed" },
      request: { operation: "extract_roundtrip" },
    },
  }), { runtimeFactory, expectedJobType: "galaxy" });
  assert.equal(submission.statusCode, 202);
  const jobId = submission.body.job.id;
  const claim = await controlPlane.claimJob({
    jobId,
    messageId: "http-message",
    deliveryCount: 1,
    workerIdentity: "acceptance-worker",
  });
  const artifactBytes = Buffer.from("verified-artifact-content");
  const artifactReference = await store.putImmutable({
    namespace: "job-artifact-acceptance",
    bytes: artifactBytes,
    mediaType: "text/plain",
    extension: "txt",
  });
  const manifestReference = await store.putImmutable({
    namespace: "job-manifest-acceptance",
    bytes: Buffer.from(`${canonicalJson({ jobId })}\n`),
    mediaType: "application/json",
    extension: "json",
  });
  await controlPlane.completeJob({
    jobId,
    leaseToken: claim.leaseToken,
    resultManifestReference: manifestReference,
    artifacts: [{
      name: "result.txt",
      objectReference: artifactReference,
      sha256: artifactReference.sha256,
      bytes: artifactReference.bytes,
      mediaType: "text/plain",
    }],
  });
  const downloaded = await call(handleProductionJob, requestFor(keyA.token, {
    query: { id: jobId, resource: "artifact", name: "result.txt" },
  }), { runtimeFactory });
  assert.equal(downloaded.statusCode, 200);
  assert.deepEqual(downloaded.body, artifactBytes);
  assert.equal(downloaded.headers["X-Content-SHA256"], artifactReference.sha256);
  assert.equal((await service.getJob(authA, jobId)).state, "succeeded");
});
