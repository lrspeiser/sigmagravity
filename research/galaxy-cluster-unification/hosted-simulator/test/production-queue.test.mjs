import test from "node:test";
import assert from "node:assert/strict";
import {
  CANARY_TOPIC,
  consumeQueueCanary,
  productionQueueState,
  publishQueueCanary,
  queueCanaryAcknowledgement,
  queueCanaryPayload,
  queueCanaryReference,
  readQueueCanary,
  resolveQueueConfiguration,
} from "../lib/production-queue.mjs";

const environment = {
  VERCEL_DEPLOYMENT_ID: "dpl_test_deployment_0123456789",
  SIMULATOR_QUEUE_REGION: "iad1",
};

test("queue configuration is deployment-bound and fails closed", () => {
  assert.equal(resolveQueueConfiguration({}), null);
  assert.equal(productionQueueState({}), "not_configured");
  assert.equal(productionQueueState({ VERCEL_URL: "" }), "not_configured");
  assert.equal(
    productionQueueState({ VERCEL_URL: "example.vercel.app", SIMULATOR_QUEUE_REGION: "invalid" }),
    "misconfigured",
  );
  const configuration = resolveQueueConfiguration(environment);
  assert.equal(configuration.region, "iad1");
  assert.match(configuration.deploymentIdentitySha256, /^[0-9a-f]{64}$/);
});

test("canary dispatch uses one deployment-scoped idempotency key", async () => {
  const calls = [];
  const result = await publishQueueCanary({
    environment,
    async sendImpl(...args) {
      calls.push(args);
      return { messageId: "msg_test" };
    },
  });
  assert.equal(calls.length, 1);
  assert.equal(calls[0][0], CANARY_TOPIC);
  assert.equal(calls[0][2].region, "iad1");
  assert.equal(calls[0][2].retentionSeconds, 604800);
  assert.match(calls[0][2].idempotencyKey, /^queue-canary-[0-9a-f]{64}$/);
  assert.equal(result.messageId, "msg_test");
  assert.deepEqual(result.acknowledgementReference, queueCanaryReference(queueCanaryPayload(environment)));
});

test("consumer persists the exact deterministic private acknowledgement", async () => {
  const payload = queueCanaryPayload(environment);
  const expected = queueCanaryReference(payload);
  let persisted;
  const store = {
    async putImmutable(input) {
      persisted = input;
      return expected;
    },
    async hasVerified(reference) {
      assert.deepEqual(reference, expected);
      return true;
    },
  };
  const reference = await consumeQueueCanary(payload, { store });
  assert.deepEqual(reference, expected);
  assert.equal(persisted.namespace, expected.namespace);
  assert.deepEqual(JSON.parse(persisted.bytes.toString("utf8")), queueCanaryAcknowledgement(payload));
  const status = await readQueueCanary({ environment, store });
  assert.equal(status.state, "verified_consumed");
  assert.deepEqual(status.acknowledgementReference, expected);
});

test("canary rejects a payload not bound to a deployment hash", async () => {
  await assert.rejects(
    consumeQueueCanary({ schemaVersion: "sigma-production-queue-canary/1", deploymentIdentitySha256: "bad" }, {
      store: { async putImmutable() { throw new Error("must not persist"); } },
    }),
    /payload is invalid/,
  );
});
