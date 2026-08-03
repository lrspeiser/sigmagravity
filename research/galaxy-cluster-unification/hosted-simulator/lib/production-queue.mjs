import { QueueClient, send as queueSend } from "@vercel/queue";
import { canonicalJson, sha256 } from "./canonical.mjs";
import { PrivateBlobStore, privateBlobReferenceFor } from "./private-blob-store.mjs";

export const CONTROL_PLANE_TOPIC = "sigma-control-plane-jobs-v1";
export const CANARY_TOPIC = "sigma-control-plane-canary-v1";
const HASH = /^[0-9a-f]{64}$/;

function boundedIdentity(value) {
  if (typeof value !== "string" || value.length < 1 || value.length > 512) {
    throw new Error("a bounded Vercel deployment identity is required");
  }
  return value;
}

export function resolveQueueConfiguration(environment = process.env) {
  const region = environment.SIMULATOR_QUEUE_REGION ?? "iad1";
  if (!/^[a-z]{3}[1-9]$/.test(region)) throw new Error("SIMULATOR_QUEUE_REGION is invalid");
  const deploymentIdentity = environment.VERCEL_DEPLOYMENT_ID ?? environment.VERCEL_URL;
  if (!deploymentIdentity) return null;
  return {
    region,
    deploymentIdentity: boundedIdentity(deploymentIdentity),
    deploymentIdentitySha256: sha256(deploymentIdentity),
  };
}

export function productionQueueState(environment = process.env) {
  try {
    return resolveQueueConfiguration(environment) ? "configured" : "not_configured";
  } catch {
    return "misconfigured";
  }
}

export function queueCanaryPayload(environment = process.env) {
  const configuration = resolveQueueConfiguration(environment);
  if (!configuration) throw new Error("Vercel Queue is not configured for this environment");
  return {
    schemaVersion: "sigma-production-queue-canary/1",
    deploymentIdentitySha256: configuration.deploymentIdentitySha256,
  };
}

export function queueCanaryAcknowledgement(payload) {
  if (
    !payload
    || payload.schemaVersion !== "sigma-production-queue-canary/1"
    || typeof payload.deploymentIdentitySha256 !== "string"
    || !HASH.test(payload.deploymentIdentitySha256)
  ) {
    throw new Error("queue canary payload is invalid");
  }
  return {
    schemaVersion: "sigma-production-queue-canary-ack/1",
    topic: CANARY_TOPIC,
    deploymentIdentitySha256: payload.deploymentIdentitySha256,
    submittedPayloadSha256: sha256(payload),
  };
}

export function queueCanaryBytes(payload) {
  return Buffer.from(`${canonicalJson(queueCanaryAcknowledgement(payload))}\n`, "utf8");
}

export function queueCanaryReference(payload) {
  const bytes = queueCanaryBytes(payload);
  return privateBlobReferenceFor({
    namespace: `queue-canary-${payload.deploymentIdentitySha256.slice(0, 24)}`,
    bytes,
    mediaType: "application/json",
    extension: "json",
  });
}

export async function publishQueueCanary({
  environment = process.env,
  sendImpl = queueSend,
} = {}) {
  const configuration = resolveQueueConfiguration(environment);
  if (!configuration) throw new Error("Vercel Queue is not configured for this environment");
  const payload = queueCanaryPayload(environment);
  const result = await sendImpl(CANARY_TOPIC, payload, {
    idempotencyKey: `queue-canary-${configuration.deploymentIdentitySha256}`,
    retentionSeconds: 604800,
    region: configuration.region,
  });
  return {
    schemaVersion: "sigma-production-queue-canary-dispatch/1",
    state: "published_or_deduplicated",
    topic: CANARY_TOPIC,
    messageId: result?.messageId ?? null,
    deploymentIdentitySha256: configuration.deploymentIdentitySha256,
    acknowledgementReference: queueCanaryReference(payload),
  };
}

export async function consumeQueueCanary(payload, {
  store = new PrivateBlobStore(),
} = {}) {
  const bytes = queueCanaryBytes(payload);
  const reference = await store.putImmutable({
    namespace: `queue-canary-${payload.deploymentIdentitySha256.slice(0, 24)}`,
    bytes,
    mediaType: "application/json",
    extension: "json",
  });
  const expected = queueCanaryReference(payload);
  if (reference.pathname !== expected.pathname || reference.sha256 !== expected.sha256) {
    throw new Error("queue canary acknowledgement identity changed during persistence");
  }
  return reference;
}

export async function readQueueCanary({
  environment = process.env,
  store = new PrivateBlobStore(),
} = {}) {
  const payload = queueCanaryPayload(environment);
  const reference = queueCanaryReference(payload);
  const consumed = await store.hasVerified(reference);
  return {
    schemaVersion: "sigma-production-queue-canary-status/1",
    state: consumed ? "verified_consumed" : "not_yet_consumed",
    topic: CANARY_TOPIC,
    deploymentIdentitySha256: payload.deploymentIdentitySha256,
    acknowledgementReference: reference,
  };
}

export function createQueueNodeHandler(callback, { environment = process.env } = {}) {
  const configuration = resolveQueueConfiguration(environment);
  const queue = new QueueClient({ region: configuration?.region ?? "iad1" });
  return queue.handleNodeCallback(callback, {
    visibilityTimeoutSeconds: 60,
    retry(error, metadata) {
      if (metadata.deliveryCount > 8) return { acknowledge: true };
      return { afterSeconds: Math.min(300, 2 ** metadata.deliveryCount) };
    },
  });
}

export class VercelQueuePublisher {
  constructor({ environment = process.env, sendImpl = queueSend } = {}) {
    const configuration = resolveQueueConfiguration(environment);
    if (!configuration) throw new Error("Vercel Queue is not configured for this environment");
    this.region = configuration.region;
    this.sendImpl = sendImpl;
  }

  send(topic, payload, options = {}) {
    return this.sendImpl(topic, payload, { ...options, region: this.region });
  }
}

export function createControlPlaneQueueNodeHandler(callback, { environment = process.env } = {}) {
  const configuration = resolveQueueConfiguration(environment);
  const queue = new QueueClient({ region: configuration?.region ?? "iad1" });
  return queue.handleNodeCallback(callback, {
    visibilityTimeoutSeconds: 300,
    retry(error, metadata) {
      if (error?.retryable === false) return { acknowledge: true };
      return { afterSeconds: Math.min(300, 2 ** Math.min(metadata.deliveryCount, 8)) };
    },
  });
}
