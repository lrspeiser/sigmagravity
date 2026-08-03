import assert from "node:assert/strict";
import test from "node:test";
import {
  proxyWorkerRequest,
  resolveWorkerConnector,
  workerConnectorState,
} from "../lib/remote-worker-proxy.mjs";

const TOKEN = "gateway-worker-token-0123456789abcdef";

function responseFixture() {
  return {
    statusCode: 200,
    headers: {},
    body: Buffer.alloc(0),
    setHeader(name, value) { this.headers[name.toLowerCase()] = value; },
    status(value) { this.statusCode = value; return this; },
    json(value) { this.body = Buffer.from(JSON.stringify(value)); return this; },
    end(value = Buffer.alloc(0)) { this.body = Buffer.isBuffer(value) ? value : Buffer.from(value); return this; },
  };
}

test("worker connector requires one HTTPS origin and a paired long secret", () => {
  assert.equal(resolveWorkerConnector({}), null);
  assert.equal(workerConnectorState({}), "not_configured");
  assert.equal(workerConnectorState({ SIMULATOR_WORKER_URL: "https://worker.example" }), "misconfigured");
  assert.throws(
    () => resolveWorkerConnector({ SIMULATOR_WORKER_URL: "http://worker.example", SIMULATOR_WORKER_TOKEN: TOKEN }),
    /HTTPS/,
  );
  assert.throws(
    () => resolveWorkerConnector({ SIMULATOR_WORKER_URL: "https://worker.example/path", SIMULATOR_WORKER_TOKEN: TOKEN }),
    /origin/,
  );
  const local = resolveWorkerConnector({ SIMULATOR_WORKER_URL: "http://127.0.0.1:8787", SIMULATOR_WORKER_TOKEN: TOKEN });
  assert.equal(local.origin, "http://127.0.0.1:8787");
});

test("unconfigured gateway retains the explicit production-not-connected response", async () => {
  const response = responseFixture();
  await proxyWorkerRequest({
    request: { method: "POST", headers: {}, body: {} },
    response,
    path: "api/v1/field-jobs",
    environment: {},
    fetchImpl: async () => { throw new Error("fetch must not run"); },
  });
  assert.equal(response.statusCode, 503);
  assert.equal(JSON.parse(response.body).error, "production_worker_not_connected");
});

test("configured gateway forwards bounded requests with server-side authorization", async () => {
  const response = responseFixture();
  let captured;
  await proxyWorkerRequest({
    request: { method: "POST", headers: { accept: "application/json" }, body: { schemaVersion: "test/1" } },
    response,
    path: "api/v1/field-jobs",
    environment: { SIMULATOR_WORKER_URL: "https://worker.example", SIMULATOR_WORKER_TOKEN: TOKEN },
    fetchImpl: async (url, options) => {
      captured = { url, options };
      return new Response(JSON.stringify({ id: "job_0123456789abcdef01234567", state: "queued" }), {
        status: 202,
        headers: { "content-type": "application/json", "x-content-sha256": "a".repeat(64) },
      });
    },
  });
  assert.equal(captured.url, "https://worker.example/api/v1/field-jobs");
  assert.equal(captured.options.headers.Authorization, `Bearer ${TOKEN}`);
  assert.deepEqual(JSON.parse(captured.options.body.toString("utf8")), { schemaVersion: "test/1" });
  assert.equal(response.statusCode, 202);
  assert.equal(JSON.parse(response.body).state, "queued");
  assert.equal(response.headers["x-content-sha256"], "a".repeat(64));
  assert.equal(response.body.includes(TOKEN), false);
});

test("gateway forwards an unparsed binary request stream without JSON reinterpretation", async () => {
  const response = responseFixture();
  const chunks = [Buffer.from([0, 1, 2]), Buffer.from([253, 254, 255])];
  const request = {
    method: "PUT",
    headers: { "content-type": "application/octet-stream" },
    async *[Symbol.asyncIterator]() { for (const chunk of chunks) yield chunk; },
  };
  let forwarded;
  await proxyWorkerRequest({
    request,
    response,
    path: "api/v1/data-uploads/upload_0123456789abcdef01234567/content",
    environment: { SIMULATOR_WORKER_URL: "https://worker.example", SIMULATOR_WORKER_TOKEN: TOKEN },
    fetchImpl: async (_url, options) => {
      forwarded = options.body;
      return new Response(JSON.stringify({ state: "ready" }), { status: 200 });
    },
  });
  assert.deepEqual(forwarded, Buffer.concat(chunks));
  assert.equal(response.statusCode, 200);
});

test("gateway rejects unsafe paths and request bodies beyond its proxy quota", async () => {
  let fetchCalls = 0;
  const environment = {
    SIMULATOR_WORKER_URL: "https://worker.example",
    SIMULATOR_WORKER_TOKEN: TOKEN,
    SIMULATOR_GATEWAY_MAX_REQUEST_BYTES: "32",
  };
  const unsafe = responseFixture();
  await proxyWorkerRequest({
    request: { method: "GET", headers: {} },
    response: unsafe,
    path: "api/v1/field-jobs/../../secret",
    environment,
    fetchImpl: async () => { fetchCalls += 1; },
  });
  assert.equal(unsafe.statusCode, 400);
  const oversized = responseFixture();
  await proxyWorkerRequest({
    request: { method: "PUT", headers: {}, body: Buffer.alloc(33) },
    response: oversized,
    path: "api/v1/data-uploads/upload_0123456789abcdef01234567/content",
    environment,
    fetchImpl: async () => { fetchCalls += 1; },
  });
  assert.equal(oversized.statusCode, 413);
  assert.equal(fetchCalls, 0);
});

test("gateway turns timeout or oversized upstream output into a bounded 502", async () => {
  const response = responseFixture();
  await proxyWorkerRequest({
    request: { method: "GET", headers: {} },
    response,
    path: "api/v1/field-jobs/job_0123456789abcdef01234567/artifacts",
    environment: {
      SIMULATOR_WORKER_URL: "https://worker.example",
      SIMULATOR_WORKER_TOKEN: TOKEN,
      SIMULATOR_GATEWAY_MAX_RESPONSE_BYTES: "16",
    },
    fetchImpl: async () => new Response("x".repeat(17), { status: 200 }),
  });
  assert.equal(response.statusCode, 502);
  assert.equal(JSON.parse(response.body).error, "production_worker_unreachable");
});
