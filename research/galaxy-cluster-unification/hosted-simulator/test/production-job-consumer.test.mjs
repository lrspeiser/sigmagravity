import test from "node:test";
import assert from "node:assert/strict";
import { canonicalJson } from "../lib/canonical.mjs";
import { privateBlobReferenceFor } from "../lib/private-blob-store.mjs";
import {
  processProductionJobMessage,
  verifyStatelessWorkerResult,
} from "../lib/production-job-consumer.mjs";

const JOB_ID = "job_0123456789abcdef01234567";
const PROJECT_ID = "project_0123456789abcdef01234567";

function stored(namespace, value, extension = "json", mediaType = "application/json") {
  const bytes = Buffer.from(typeof value === "string" ? value : `${canonicalJson(value)}\n`, "utf8");
  return {
    bytes,
    reference: privateBlobReferenceFor({ namespace, bytes, extension, mediaType }),
  };
}

function successfulResult() {
  const artifact = stored("result-artifact", "field bytes\n", "npz", "application/x-npz");
  const manifest = {
    schemaVersion: "sigma-scientific-result-manifest/1",
    jobId: JOB_ID,
    artifacts: [{
      name: "field.npz",
      pathname: artifact.reference.pathname,
      sha256: artifact.reference.sha256,
      bytes: artifact.reference.bytes,
    }],
  };
  const manifestObject = stored("result-manifest", manifest);
  const objects = new Map([
    [artifact.reference.pathname, artifact.bytes],
    [manifestObject.reference.pathname, manifestObject.bytes],
  ]);
  return {
    result: {
      schemaVersion: "sigma-stateless-worker-result/1",
      jobId: JOB_ID,
      resultManifestReference: manifestObject.reference,
      artifacts: [{
        name: "field.npz",
        objectReference: artifact.reference,
        sha256: artifact.reference.sha256,
        bytes: artifact.reference.bytes,
        mediaType: artifact.reference.mediaType,
      }],
    },
    store: {
      async getVerified(reference) {
        const bytes = objects.get(reference.pathname);
        if (!bytes) throw new Error("object missing");
        return bytes;
      },
    },
  };
}

function message() {
  return {
    schemaVersion: "sigma-production-job-message/1",
    projectId: PROJECT_ID,
    jobId: JOB_ID,
    jobType: "field",
    requestSha256: "1".repeat(64),
  };
}

function claim() {
  return {
    claimed: true,
    leaseToken: "lease-token-0123456789abcdef",
    attempt: 1,
    requestObjectReference: stored("job-request", { job: JOB_ID }).reference,
    job: {
      id: JOB_ID,
      projectId: PROJECT_ID,
      jobType: "field",
      requestSha256: "1".repeat(64),
      attempt: 1,
      state: "running",
    },
  };
}

const metadata = { messageId: "msg_acceptance", deliveryCount: 1 };

test("verified result requires manifest/envelope agreement and rehashed objects", async () => {
  const { result, store } = successfulResult();
  const verified = await verifyStatelessWorkerResult(JOB_ID, result, { store });
  assert.equal(verified.artifacts[0].name, "field.npz");
  const changed = structuredClone(result);
  changed.artifacts[0].bytes += 1;
  await assert.rejects(verifyStatelessWorkerResult(JOB_ID, changed, { store }), /changed identity/);
  const missing = structuredClone(result);
  missing.jobId = "job_ffffffffffffffffffffffff";
  await assert.rejects(verifyStatelessWorkerResult(JOB_ID, missing, { store }), /envelope is invalid/);
});

test("queue consumer claims, executes, verifies, and atomically completes", async () => {
  const { result, store } = successfulResult();
  const calls = [];
  const controlPlane = {
    async claimJob(input) { calls.push(["claim", input]); return claim(); },
    async completeJob(input) { calls.push(["complete", input]); return { state: "succeeded" }; },
    async failJob() { throw new Error("must not fail"); },
  };
  const output = await processProductionJobMessage(message(), metadata, {
    controlPlane,
    executor: { async execute(input) { calls.push(["execute", input]); return result; } },
    store,
    workerIdentity: "container@sha256:verified",
  });
  assert.deepEqual(output, { acknowledged: true, reason: "completed", state: "succeeded" });
  assert.deepEqual(calls.map(([name]) => name), ["claim", "execute", "complete"]);
  assert.equal(calls[0][1].workerIdentity, "container@sha256:verified");
  assert.equal(calls[2][1].artifacts[0].sha256, result.artifacts[0].sha256);
});

test("duplicate terminal delivery acknowledges without executing", async () => {
  let executed = false;
  const output = await processProductionJobMessage(message(), metadata, {
    controlPlane: {
      async claimJob() { return { claimed: false, reason: "terminal", job: { state: "succeeded" } }; },
    },
    executor: { async execute() { executed = true; } },
    store: {},
    workerIdentity: "worker",
  });
  assert.equal(executed, false);
  assert.deepEqual(output, { acknowledged: true, reason: "terminal", state: "succeeded" });
});

test("retryable failures preserve queue redelivery and nonretryable failures acknowledge", async () => {
  const retryable = Object.assign(new Error("temporary worker outage"), { retryable: true, code: "worker_unreachable" });
  await assert.rejects(
    processProductionJobMessage(message(), metadata, {
      controlPlane: {
        async claimJob() { return claim(); },
        async failJob(input) {
          assert.equal(input.error, retryable);
          return { state: "queued", shouldRetry: true };
        },
      },
      executor: { async execute() { throw retryable; } },
      store: {},
      workerIdentity: "worker",
    }),
    retryable,
  );

  const invalid = Object.assign(new Error("invalid numerical output"), { retryable: false });
  const terminal = await processProductionJobMessage(message(), metadata, {
    controlPlane: {
      async claimJob() { return claim(); },
      async failJob(input) {
        assert.equal(input.retryable, false);
        return { state: "failed", shouldRetry: false };
      },
    },
    executor: { async execute() { throw invalid; } },
    store: {},
    workerIdentity: "worker",
  });
  assert.deepEqual(terminal, { acknowledged: true, reason: "terminal_failure", state: "failed" });
});

test("malformed queue payload is rejected before database access", async () => {
  let claimed = false;
  await assert.rejects(
    processProductionJobMessage({ ...message(), jobId: "../escape" }, metadata, {
      controlPlane: { async claimJob() { claimed = true; } },
      executor: { async execute() {} },
      workerIdentity: "worker",
    }),
    /queue message is invalid/,
  );
  assert.equal(claimed, false);
});
