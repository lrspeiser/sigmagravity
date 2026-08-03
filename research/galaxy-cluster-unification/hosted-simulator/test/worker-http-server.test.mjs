import assert from "node:assert/strict";
import test from "node:test";
import { createWorkerHttpServer } from "../lib/worker-http-server.mjs";

const TOKEN = "worker-test-token-0123456789abcdef";

function serviceFixture() {
  const calls = [];
  return {
    calls,
    maxUploadBytes: 64,
    maxStoredJobs: 7,
    maxEstimatedMemoryBytes: 4096,
    maxArtifactBytes: 8192,
    maxArtifactCount: 12,
    async createUpload(body) {
      calls.push({ method: "createUpload", body });
      return { id: "upload_0123456789abcdef01234567", state: "awaiting_content" };
    },
    async putUploadContent(id, body) {
      calls.push({ method: "putUploadContent", id, body });
      return { id, state: "ready" };
    },
    async createGalaxyJob(body) {
      calls.push({ method: "createGalaxyJob", body });
      return { id: "job_0123456789abcdef01234567", state: "queued" };
    },
  };
}

async function listeningServer(t, options = {}) {
  const service = options.service ?? serviceFixture();
  const server = createWorkerHttpServer({ service, token: TOKEN, ...options });
  await new Promise((resolvePromise) => server.listen(0, "127.0.0.1", resolvePromise));
  t.after(() => new Promise((resolvePromise) => server.close(resolvePromise)));
  const address = server.address();
  return { service, base: `http://127.0.0.1:${address.port}` };
}

test("worker server requires a long shared secret", () => {
  assert.throws(
    () => createWorkerHttpServer({ service: serviceFixture(), token: "short" }),
    /at least 32/,
  );
});

test("worker health discloses quotas and durability without requiring or echoing the secret", async (t) => {
  const { base } = await listeningServer(t, { persistentStoreConfigured: true });
  const response = await fetch(`${base}/healthz`);
  assert.equal(response.status, 200);
  const body = await response.json();
  assert.equal(body.schemaVersion, "sigma-authenticated-simulator-worker-http/2");
  assert.equal(body.jobClasses.field, "available");
  assert.equal(body.jobClasses.galaxyExtractionAndGeneration, "available");
  assert.equal(body.jobClasses.inverseResponse, "not_exposed");
  assert.equal(body.store.backend, "filesystem");
  assert.equal(body.store.storePathConfigured, true);
  assert.equal(body.store.durability, "requires_deployment_volume_verification");
  assert.equal(body.store.productionDurabilityClaim, false);
  assert.equal(body.arbitraryCodeExecution, false);
  assert.equal(body.quotas.maxArtifactBytes, 8192);
  assert.equal(JSON.stringify(body).includes(TOKEN), false);
});

test("scientific routes reject absent and incorrect bearer credentials", async (t) => {
  const { base, service } = await listeningServer(t);
  for (const authorization of [undefined, "Bearer wrong"]) {
    const response = await fetch(`${base}/api/v1/data-uploads`, {
      method: "POST",
      headers: authorization ? { authorization, "content-type": "application/json" } : { "content-type": "application/json" },
      body: "{}",
    });
    assert.equal(response.status, 401);
    assert.equal((await response.json()).error, "unauthorized");
  }
  assert.equal(service.calls.length, 0);
});

test("authorized JSON and binary upload requests reach the allow-listed field service", async (t) => {
  const { base, service } = await listeningServer(t);
  const headers = { authorization: `Bearer ${TOKEN}` };
  const ticketResponse = await fetch(`${base}/api/v1/data-uploads`, {
    method: "POST",
    headers: { ...headers, "content-type": "application/json" },
    body: JSON.stringify({ schemaVersion: "test" }),
  });
  assert.equal(ticketResponse.status, 201);
  const ticket = await ticketResponse.json();
  assert.equal(ticket.state, "awaiting_content");
  const content = Buffer.from("immutable-npz-bytes");
  const contentResponse = await fetch(`${base}/api/v1/data-uploads/${ticket.id}/content`, {
    method: "PUT",
    headers: { ...headers, "content-type": "application/octet-stream" },
    body: content,
  });
  assert.equal(contentResponse.status, 200);
  assert.equal((await contentResponse.json()).state, "ready");
  assert.equal(service.calls[0].body.schemaVersion, "test");
  assert.deepEqual(service.calls[1].body, content);
});

test("authorized gravity-independent galaxy submissions reach the allow-listed service", async (t) => {
  const { base, service } = await listeningServer(t);
  const response = await fetch(`${base}/api/v1/galaxy-jobs`, {
    method: "POST",
    headers: { authorization: `Bearer ${TOKEN}`, "content-type": "application/json" },
    body: JSON.stringify({ schemaVersion: "sigma-galaxy-job-submit/1", operation: "generate" }),
  });
  assert.equal(response.status, 202);
  assert.equal((await response.json()).state, "queued");
  assert.equal(service.calls[0].method, "createGalaxyJob");
  assert.equal(service.calls[0].body.operation, "generate");
});

test("worker boundary rejects unsupported job classes and oversized request bodies", async (t) => {
  const { base } = await listeningServer(t, { maxJsonBytes: 1024 });
  const authorization = { authorization: `Bearer ${TOKEN}`, "content-type": "application/json" };
  const unsupported = await fetch(`${base}/api/v1/inverse-response-jobs`, {
    method: "POST",
    headers: authorization,
    body: "{}",
  });
  assert.equal(unsupported.status, 404);
  const oversized = await fetch(`${base}/api/v1/data-uploads`, {
    method: "POST",
    headers: authorization,
    body: JSON.stringify({ value: "x".repeat(2048) }),
  });
  assert.equal(oversized.status, 413);
  assert.equal((await oversized.json()).error, "request_too_large");
});
