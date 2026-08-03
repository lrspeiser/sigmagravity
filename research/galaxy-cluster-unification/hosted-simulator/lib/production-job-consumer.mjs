import { PrivateBlobStore, validatePrivateBlobReference } from "./private-blob-store.mjs";

const HASH = /^[0-9a-f]{64}$/;

function validateMessage(message) {
  if (
    !message
    || message.schemaVersion !== "sigma-production-job-message/1"
    || !/^project_[0-9a-f]{24}$/.test(message.projectId)
    || !/^job_[0-9a-f]{24}$/.test(message.jobId)
    || !new Set(["field", "galaxy", "observation", "inverse_response", "batch", "advanced_plugin"]).has(message.jobType)
    || typeof message.requestSha256 !== "string"
    || !HASH.test(message.requestSha256)
  ) {
    throw Object.assign(new Error("production queue message is invalid"), {
      code: "invalid_queue_message",
      retryable: false,
    });
  }
  return message;
}

function artifactRecord(value) {
  if (!value || typeof value.name !== "string" || !/^[A-Za-z0-9][A-Za-z0-9_.-]{0,159}$/.test(value.name)) {
    throw Object.assign(new Error("worker artifact name is invalid"), { retryable: false });
  }
  const reference = validatePrivateBlobReference(value.objectReference, { maximumBytes: 10 * 1024 * 1024 * 1024 });
  if (value.sha256 !== reference.sha256 || Number(value.bytes) !== reference.bytes || value.mediaType !== reference.mediaType) {
    throw Object.assign(new Error(`worker artifact ${value.name} changed identity`), { retryable: false });
  }
  return { ...value, objectReference: reference, bytes: Number(value.bytes) };
}

export async function verifyStatelessWorkerResult(jobId, result, {
  store = new PrivateBlobStore(),
} = {}) {
  if (
    !result
    || result.schemaVersion !== "sigma-stateless-worker-result/1"
    || result.jobId !== jobId
    || !Array.isArray(result.artifacts)
    || result.artifacts.length < 1
    || result.artifacts.length > 512
  ) {
    throw Object.assign(new Error("stateless worker result envelope is invalid"), { retryable: false });
  }
  const resultManifestReference = validatePrivateBlobReference(result.resultManifestReference, {
    maximumBytes: 64 * 1024 * 1024,
  });
  const artifacts = result.artifacts.map(artifactRecord);
  if (new Set(artifacts.map((artifact) => artifact.name)).size !== artifacts.length) {
    throw Object.assign(new Error("stateless worker returned duplicate artifact names"), { retryable: false });
  }
  const manifestBytes = await store.getVerified(resultManifestReference);
  let manifest;
  try {
    manifest = JSON.parse(manifestBytes.toString("utf8"));
  } catch {
    throw Object.assign(new Error("scientific result manifest is not valid JSON"), { retryable: false });
  }
  if (
    manifest.schemaVersion !== "sigma-scientific-result-manifest/1"
    || manifest.jobId !== jobId
    || !Array.isArray(manifest.artifacts)
  ) {
    throw Object.assign(new Error("scientific result manifest identity is invalid"), { retryable: false });
  }
  const manifestIndex = new Map(manifest.artifacts.map((artifact) => [artifact.name, artifact]));
  if (manifestIndex.size !== artifacts.length) {
    throw Object.assign(new Error("scientific result manifest artifact count differs from the worker envelope"), { retryable: false });
  }
  for (const artifact of artifacts) {
    const indexed = manifestIndex.get(artifact.name);
    if (
      !indexed
      || indexed.sha256 !== artifact.sha256
      || Number(indexed.bytes) !== artifact.bytes
      || indexed.pathname !== artifact.objectReference.pathname
    ) {
      throw Object.assign(new Error(`scientific result manifest changed artifact ${artifact.name}`), { retryable: false });
    }
    await store.getVerified(artifact.objectReference);
  }
  return { resultManifestReference, artifacts };
}

export async function processProductionJobMessage(message, metadata, {
  controlPlane,
  executor,
  store,
  workerIdentity,
} = {}) {
  message = validateMessage(message);
  if (!controlPlane || !executor?.execute) throw new Error("production job consumer is not configured");
  store ??= new PrivateBlobStore();
  const claim = await controlPlane.claimJob({
    jobId: message.jobId,
    messageId: metadata.messageId,
    deliveryCount: metadata.deliveryCount,
    workerIdentity,
    leaseSeconds: 240,
  });
  if (!claim.claimed) return { acknowledged: true, reason: claim.reason, state: claim.job.state };
  try {
    const result = await executor.execute(claim);
    const verified = await verifyStatelessWorkerResult(message.jobId, result, { store });
    const completed = await controlPlane.completeJob({
      jobId: message.jobId,
      leaseToken: claim.leaseToken,
      ...verified,
    });
    return { acknowledged: true, reason: "completed", state: completed.state };
  } catch (error) {
    const failed = await controlPlane.failJob({
      jobId: message.jobId,
      leaseToken: claim.leaseToken,
      error,
      retryable: error.retryable !== false,
    });
    if (failed.shouldRetry) throw error;
    return { acknowledged: true, reason: "terminal_failure", state: failed.state };
  }
}
