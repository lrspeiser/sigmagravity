import { canonicalJson, sha256 } from "./canonical.mjs";
import { validateFieldModel } from "./field-model.mjs";
import { validateArrayBundle } from "./field-job-preflight.mjs";
import { contentSha256, privateBlobReferenceFor } from "./private-blob-store.mjs";
import { ControlPlaneError } from "./production-control-plane.mjs";

const HASH = /^[0-9a-f]{64}$/;
const JOB_TYPES = new Set(["field", "galaxy", "observation", "inverse_response", "batch", "advanced_plugin"]);
const PARAMETER_POLICIES = new Set([
  "published_fixed",
  "universal_fixed",
  "universal_fit",
  "train_validation_holdout",
  "hierarchical",
  "per_object",
]);
const MAXIMUM_JSON_BYTES = 1024 * 1024;

function object(value, label) {
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    throw new ControlPlaneError("invalid_request", `${label} must be an object`);
  }
  return value;
}

function canonicalBytes(value, label, maximumBytes = MAXIMUM_JSON_BYTES) {
  object(value, label);
  const bytes = Buffer.from(`${canonicalJson(value)}\n`, "utf8");
  if (bytes.length > maximumBytes) {
    throw new ControlPlaneError("request_too_large", `${label} exceeds the ${maximumBytes} byte canonical JSON limit`);
  }
  return bytes;
}

function hash(value, label) {
  if (typeof value !== "string" || !HASH.test(value)) {
    throw new ControlPlaneError("invalid_hash", `${label} must be a lowercase SHA-256`);
  }
  return value;
}

function confirmationReceipt(receipt) {
  object(receipt, "confirmationReceipt");
  if (receipt.schemaVersion !== "sigma-model-confirmation/1") {
    throw new ControlPlaneError("invalid_confirmation", "confirmationReceipt schema is invalid");
  }
  const validation = validateFieldModel(receipt.confirmedModel);
  if (!validation.valid || !validation.confirmation.confirmed) {
    throw new ControlPlaneError("invalid_confirmation", "confirmationReceipt does not contain a valid exact-hash-confirmed model", {
      errors: validation.errors,
    });
  }
  const core = {
    schemaVersion: "sigma-model-confirmation/1",
    modelSha256: validation.modelSha256,
    confirmedDocumentSha256: validation.documentSha256,
    sourceFormat: receipt.confirmedModel.source.format,
    sourceTextSha256: sha256(receipt.confirmedModel.source.text),
    acknowledgement: validation.confirmation.acknowledgement,
  };
  const confirmationSha256 = sha256(core);
  if (
    receipt.modelSha256 !== core.modelSha256
    || receipt.confirmedDocumentSha256 !== core.confirmedDocumentSha256
    || receipt.sourceFormat !== core.sourceFormat
    || receipt.sourceTextSha256 !== core.sourceTextSha256
    || receipt.acknowledgement !== core.acknowledgement
    || receipt.confirmationSha256 !== confirmationSha256
    || receipt.id !== `model_confirmation_${confirmationSha256.slice(0, 24)}`
  ) {
    throw new ControlPlaneError("invalid_confirmation", "confirmationReceipt identity does not match its confirmed model");
  }
  return { receipt, validation };
}

function parameterPolicy(value) {
  object(value, "parameterPolicy");
  if (!PARAMETER_POLICIES.has(value.mode)) {
    throw new ControlPlaneError("invalid_parameter_policy", "parameterPolicy.mode is invalid");
  }
  if (value.mode === "per_object" && value.disclosed !== true) {
    throw new ControlPlaneError(
      "undisclosed_per_object_parameters",
      "per-object fitting requires parameterPolicy.disclosed=true",
    );
  }
  return value;
}

function namespace(kind, projectId) {
  return `${kind}-${projectId.slice(-24)}`;
}

export class ProductionResearchService {
  constructor({ controlPlane, store, publisher, maximumJsonBytes = MAXIMUM_JSON_BYTES } = {}) {
    if (!controlPlane || !store?.putImmutable || !store?.getVerified) {
      throw new Error("ProductionResearchService requires a control plane and verified private object store");
    }
    this.controlPlane = controlPlane;
    this.store = store;
    this.publisher = publisher;
    this.maximumJsonBytes = maximumJsonBytes;
  }

  async registerModel(auth, payload) {
    if (payload?.schemaVersion !== "sigma-production-model-registration/1") {
      throw new ControlPlaneError("invalid_request", "model registration must use sigma-production-model-registration/1");
    }
    const { receipt, validation } = confirmationReceipt(payload.confirmationReceipt);
    const modelBytes = canonicalBytes(receipt.confirmedModel, "confirmed model", this.maximumJsonBytes);
    const receiptBytes = canonicalBytes(receipt, "confirmation receipt", this.maximumJsonBytes);
    const canonicalObjectReference = await this.store.putImmutable({
      namespace: namespace("model", auth.project.id),
      bytes: modelBytes,
      mediaType: "application/json",
      extension: "json",
    });
    const confirmationObjectReference = await this.store.putImmutable({
      namespace: namespace("confirmation", auth.project.id),
      bytes: receiptBytes,
      mediaType: "application/json",
      extension: "json",
    });
    const result = await this.controlPlane.registerModel({
      projectId: auth.project.id,
      modelSha256: validation.modelSha256,
      canonicalObjectReference,
      confirmationObjectReference,
      confirmedBy: auth.credentialId,
      confirmedAt: new Date().toISOString(),
      auditCredentialId: auth.credentialId,
    });
    return result;
  }

  listModels(auth, options) {
    return this.controlPlane.listModels(auth.project.id, options);
  }

  getModel(auth, modelSha256) {
    return this.controlPlane.getModel(auth.project.id, modelSha256);
  }

  async registerUpload(auth, payload) {
    if (payload?.schemaVersion !== "sigma-production-upload-registration/1") {
      throw new ControlPlaneError("invalid_request", "upload registration must use sigma-production-upload-registration/1");
    }
    const manifest = object(payload.manifest, "manifest");
    validateArrayBundle(manifest);
    hash(payload.archiveSha256, "archiveSha256");
    if (!Number.isSafeInteger(payload.archiveBytes) || payload.archiveBytes < 1) {
      throw new ControlPlaneError("invalid_request", "archiveBytes must be a positive integer");
    }
    if (payload.archiveBytes > auth.project.limits.maxUploadBytes) {
      throw new ControlPlaneError("upload_quota_exceeded", "archive exceeds the project's upload byte limit", {
        maximumBytes: auth.project.limits.maxUploadBytes,
      });
    }
    const manifestBytes = canonicalBytes(manifest, "array bundle manifest", this.maximumJsonBytes);
    const manifestObjectReference = await this.store.putImmutable({
      namespace: namespace("upload-manifest", auth.project.id),
      bytes: manifestBytes,
      mediaType: "application/json",
      extension: "json",
    });
    const scientificRoles = [...new Set(manifest.arrays.map((record) => record.scientificRole).filter(Boolean))].sort();
    const result = await this.controlPlane.registerUpload({
      projectId: auth.project.id,
      bundleSha256: manifest.bundleSha256,
      archiveSha256: payload.archiveSha256,
      archiveBytes: payload.archiveBytes,
      manifestObjectReference,
      scientificRoles,
      license: manifest.license,
      auditCredentialId: auth.credentialId,
    });
    return result;
  }

  listUploads(auth, options) {
    return this.controlPlane.listUploads(auth.project.id, options);
  }

  getUpload(auth, uploadId, options) {
    return this.controlPlane.getUpload(auth.project.id, uploadId, options);
  }

  async putUploadContent(auth, uploadId, bytes, { suppliedSha256 = null } = {}) {
    if (!Buffer.isBuffer(bytes)) bytes = Buffer.from(bytes ?? []);
    const upload = await this.controlPlane.getUpload(auth.project.id, uploadId, { includeReferences: true });
    if (bytes.length !== upload.archiveBytes) {
      throw new ControlPlaneError("upload_size_mismatch", "archive byte count does not match its registration", {
        expected: upload.archiveBytes,
        received: bytes.length,
      });
    }
    const actualSha256 = contentSha256(bytes);
    if (suppliedSha256 !== null && suppliedSha256 !== actualSha256) {
      throw new ControlPlaneError("upload_hash_mismatch", "X-Content-SHA256 does not match the uploaded bytes");
    }
    if (actualSha256 !== upload.archiveSha256) {
      throw new ControlPlaneError("upload_hash_mismatch", "archive bytes do not match the registered SHA-256");
    }
    const archiveObjectReference = await this.store.putImmutable({
      namespace: namespace("upload", auth.project.id),
      bytes,
      mediaType: "application/octet-stream",
      extension: "npz",
    });
    const result = await this.controlPlane.finalizeUpload({
      projectId: auth.project.id,
      uploadId,
      archiveObjectReference,
      auditCredentialId: auth.credentialId,
    });
    return result;
  }

  async submitJob(auth, payload, { expectedJobType = null, idempotencyKey } = {}) {
    if (!this.publisher?.send) throw new ControlPlaneError("production_queue_not_configured", "durable job queue is not configured");
    if (payload?.schemaVersion !== "sigma-production-job-submit/1") {
      throw new ControlPlaneError("invalid_request", "job submission must use sigma-production-job-submit/1");
    }
    if (!JOB_TYPES.has(payload.jobType) || (expectedJobType && payload.jobType !== expectedJobType)) {
      throw new ControlPlaneError("invalid_job_type", "jobType does not match this endpoint");
    }
    if (typeof idempotencyKey !== "string" || idempotencyKey.length < 8 || idempotencyKey.length > 160) {
      throw new ControlPlaneError("idempotency_key_required", "Idempotency-Key must contain 8-160 characters");
    }
    const policy = parameterPolicy(payload.parameterPolicy);
    const request = object(payload.request, "request");
    const modelSha256 = payload.modelSha256 ?? null;
    const dataUploadId = payload.dataUploadId ?? null;
    if (modelSha256 !== null) hash(modelSha256, "modelSha256");
    if (payload.jobType === "field" && (modelSha256 === null || dataUploadId === null)) {
      throw new ControlPlaneError("invalid_request", "field jobs require modelSha256 and dataUploadId");
    }
    const storedRequest = {
      schemaVersion: "sigma-production-worker-request/1",
      jobType: payload.jobType,
      modelSha256,
      dataUploadId,
      parameterPolicy: policy,
      request,
    };
    const requestBytes = canonicalBytes(storedRequest, "job request", this.maximumJsonBytes);
    const requestNamespace = namespace("job-request", auth.project.id);
    const expectedReference = privateBlobReferenceFor({
      namespace: requestNamespace,
      bytes: requestBytes,
      mediaType: "application/json",
      extension: "json",
    });
    const existing = await this.controlPlane.findJobByIdempotency(auth.project.id, payload.jobType, idempotencyKey);
    if (existing && (
      existing.requestObjectReference.sha256 !== expectedReference.sha256
      || existing.modelSha256 !== modelSha256
      || existing.inputUploadId !== dataUploadId
    )) {
      throw new ControlPlaneError(
        "idempotency_conflict",
        "idempotency key is already bound to different scientific inputs",
        { existingJobId: existing.job.id },
      );
    }
    if (existing) await this.store.getVerified(existing.requestObjectReference);
    const requestObjectReference = existing?.requestObjectReference ?? await this.store.putImmutable({
      namespace: requestNamespace,
      bytes: requestBytes,
      mediaType: "application/json",
      extension: "json",
    });
    const created = await this.controlPlane.createJob({
      projectId: auth.project.id,
      jobType: payload.jobType,
      idempotencyKey,
      requestSha256: requestObjectReference.sha256,
      requestObjectReference,
      modelSha256,
      dataUploadId,
      parameterPolicy: policy,
      maxAttempts: payload.maxAttempts ?? auth.project.limits.maxAttemptsPerJob,
      auditCredentialId: auth.credentialId,
    });
    const dispatch = await this.controlPlane.dispatchAvailable(this.publisher, { limit: 16 });
    return {
      created: created.created,
      job: await this.controlPlane.getJob(auth.project.id, created.job.id),
      dispatch: dispatch.map(({ outboxId, state, messageId }) => ({ outboxId, state, messageId })),
    };
  }

  listJobs(auth, options) {
    return this.controlPlane.listJobs(auth.project.id, options);
  }

  getJob(auth, jobId) {
    return this.controlPlane.getJob(auth.project.id, jobId);
  }

  getEvents(auth, jobId) {
    return this.controlPlane.getEvents(auth.project.id, jobId);
  }

  getArtifacts(auth, jobId) {
    return this.controlPlane.getArtifacts(auth.project.id, jobId);
  }

  async downloadArtifact(auth, jobId, name) {
    const artifact = await this.controlPlane.getArtifact(auth.project.id, jobId, name);
    const bytes = await this.store.getVerified(artifact.objectReference);
    return { artifact, bytes };
  }

  async cancelJob(auth, jobId) {
    return this.controlPlane.requestCancellation(auth.project.id, jobId, {
      auditCredentialId: auth.credentialId,
    });
  }
}
