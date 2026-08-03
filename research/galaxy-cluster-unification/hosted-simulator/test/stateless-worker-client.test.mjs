import test from "node:test";
import assert from "node:assert/strict";
import {
  StatelessWorkerClient,
  resolveStatelessWorkerConfiguration,
  statelessWorkerState,
} from "../lib/stateless-worker-client.mjs";

const environment = {
  SIGMA_STATELESS_WORKER_URL: "https://worker.example.test",
  SIGMA_STATELESS_WORKER_TOKEN: "x".repeat(48),
};

const execution = {
  job: {
    id: "job_0123456789abcdef01234567",
    projectId: "project_0123456789abcdef01234567",
    jobType: "field",
    requestSha256: "1".repeat(64),
    attempt: 1,
  },
  leaseToken: "lease-token-0123456789abcdef",
  requestObjectReference: { schemaVersion: "sigma-private-blob-object/1" },
};

test("stateless worker configuration requires a paired HTTPS origin and long secret", () => {
  assert.equal(resolveStatelessWorkerConfiguration({}), null);
  assert.equal(statelessWorkerState({}), "not_configured");
  assert.equal(statelessWorkerState({ SIGMA_STATELESS_WORKER_URL: environment.SIGMA_STATELESS_WORKER_URL }), "misconfigured");
  assert.throws(
    () => resolveStatelessWorkerConfiguration({ ...environment, SIGMA_STATELESS_WORKER_URL: "http://worker.example.test" }),
    /HTTPS/,
  );
  assert.throws(
    () => resolveStatelessWorkerConfiguration({ ...environment, SIGMA_STATELESS_WORKER_URL: "https://user:pass@worker.example.test" }),
    /origin without credentials/,
  );
});

test("execution handoff is bounded, authenticated, and contains no artifact bytes", async () => {
  let observed;
  const result = { schemaVersion: "sigma-stateless-worker-result/1", jobId: execution.job.id };
  const client = new StatelessWorkerClient({
    environment,
    async fetchImpl(url, options) {
      observed = { url, options };
      return new Response(JSON.stringify(result), {
        status: 200,
        headers: { "content-type": "application/json" },
      });
    },
  });
  assert.deepEqual(await client.execute(execution), result);
  assert.equal(observed.url, "https://worker.example.test/v1/executions");
  assert.equal(observed.options.redirect, "error");
  assert.equal(observed.options.headers.Authorization, `Bearer ${environment.SIGMA_STATELESS_WORKER_TOKEN}`);
  const body = JSON.parse(observed.options.body.toString("utf8"));
  assert.equal(body.schemaVersion, "sigma-stateless-worker-execution/1");
  assert.equal(body.job.id, execution.job.id);
  assert.deepEqual(body.requestObjectReference, execution.requestObjectReference);
  assert.equal(JSON.stringify(body).includes("artifactBytes"), false);
});

test("worker errors classify retryability without returning credentials", async () => {
  const rejected = new StatelessWorkerClient({
    environment,
    async fetchImpl() {
      return new Response(JSON.stringify({ error: "invalid_model" }), { status: 422 });
    },
  });
  await assert.rejects(
    rejected.execute(execution),
    (error) => error.code === "invalid_model" && error.retryable === false && !error.message.includes("xxxx"),
  );
  const unavailable = new StatelessWorkerClient({
    environment,
    async fetchImpl() { throw new Error("network down"); },
  });
  await assert.rejects(
    unavailable.execute(execution),
    (error) => error.code === "worker_unreachable" && error.retryable === true,
  );
});

test("oversized worker responses are terminal protocol failures", async () => {
  const client = new StatelessWorkerClient({
    environment: { ...environment, SIGMA_STATELESS_WORKER_MAX_RESPONSE_BYTES: "32" },
    async fetchImpl() {
      return new Response(JSON.stringify({ value: "x".repeat(100) }), { status: 200 });
    },
  });
  await assert.rejects(
    client.execute(execution),
    (error) => error.retryable === false && /quota/.test(error.message),
  );
});
