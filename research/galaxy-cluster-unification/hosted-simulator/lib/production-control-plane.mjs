import { randomUUID } from "node:crypto";
import { canonicalJson, sha256 } from "./canonical.mjs";
import { validatePrivateBlobReference } from "./private-blob-store.mjs";
import { CONTROL_PLANE_TOPIC } from "./production-queue.mjs";

const IDENTIFIERS = {
  project: /^project_[0-9a-f]{24}$/,
  job: /^job_[0-9a-f]{24}$/,
  upload: /^upload_[0-9a-f]{24}$/,
  outbox: /^outbox_[0-9a-f]{24}$/,
  credential: /^key_[0-9a-f]{24}$/,
};
const HASH = /^[0-9a-f]{64}$/;
const TERMINAL = new Set(["succeeded", "failed", "cancelled"]);
const JOB_TYPES = new Set(["field", "galaxy", "observation", "inverse_response", "batch", "advanced_plugin"]);
const ARTIFACT_NAME = /^[A-Za-z0-9][A-Za-z0-9_.-]{0,159}$/;

export class ControlPlaneError extends Error {
  constructor(code, message, details = {}) {
    super(message);
    this.name = "ControlPlaneError";
    this.code = code;
    this.details = details;
  }
}

function rows(result) {
  if (Array.isArray(result)) return result;
  return result?.rows ?? [];
}

function one(result) {
  return rows(result)[0] ?? null;
}

function identifier(value, kind) {
  if (typeof value !== "string" || !IDENTIFIERS[kind].test(value)) {
    throw new ControlPlaneError("invalid_identifier", `${kind} identifier is invalid`);
  }
  return value;
}

function hash(value, label) {
  if (typeof value !== "string" || !HASH.test(value)) {
    throw new ControlPlaneError("invalid_hash", `${label} must be a lowercase SHA-256`);
  }
  return value;
}

function boundedString(value, label, minimum, maximum) {
  if (typeof value !== "string" || value.length < minimum || value.length > maximum) {
    throw new ControlPlaneError("invalid_value", `${label} must contain ${minimum}-${maximum} characters`);
  }
  return value;
}

function positiveInteger(value, fallback, maximum, label) {
  const result = value === undefined ? fallback : Number(value);
  if (!Number.isSafeInteger(result) || result < 1 || result > maximum) {
    throw new ControlPlaneError("invalid_value", `${label} must be an integer from 1 to ${maximum}`);
  }
  return result;
}

function jsonValue(value, label) {
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    throw new ControlPlaneError("invalid_value", `${label} must be an object`);
  }
  JSON.parse(canonicalJson(value));
  return value;
}

function privateReference(value, label) {
  try {
    return validatePrivateBlobReference(value, { maximumBytes: 10 * 1024 * 1024 * 1024 });
  } catch (error) {
    throw new ControlPlaneError("invalid_object_reference", `${label}: ${error.message}`);
  }
}

function jsonParameter(value) {
  return JSON.stringify(value);
}

function parseJson(value) {
  if (typeof value !== "string") return value;
  return JSON.parse(value);
}

function safeError(error) {
  const rawCode = typeof error?.code === "string" ? error.code : "worker_failure";
  const code = /^[A-Za-z0-9_.-]{1,80}$/.test(rawCode) ? rawCode : "worker_failure";
  const rawMessage = typeof error?.message === "string" ? error.message : "Scientific worker failed";
  return {
    code,
    message: rawMessage.slice(0, 500) || "Scientific worker failed",
  };
}

function publicJob(row) {
  if (!row) return null;
  return {
    id: row.job_id,
    projectId: row.project_id,
    jobType: row.job_type,
    state: row.state,
    requestSha256: row.request_sha256,
    modelSha256: row.model_sha256,
    dataUploadId: row.input_upload_id,
    parameterPolicy: parseJson(row.parameter_policy),
    attempt: Number(row.current_attempt),
    maxAttempts: Number(row.max_attempts),
    cancellationRequested: row.cancellation_requested_at !== null,
    createdAt: new Date(row.created_at).toISOString(),
    updatedAt: new Date(row.updated_at).toISOString(),
    startedAt: row.started_at ? new Date(row.started_at).toISOString() : null,
    finishedAt: row.finished_at ? new Date(row.finished_at).toISOString() : null,
    error: parseJson(row.error),
    links: {
      self: `/api/v1/jobs/${row.job_id}`,
      events: `/api/v1/jobs/${row.job_id}/events`,
      artifacts: `/api/v1/jobs/${row.job_id}/artifacts`,
      cancel: `/api/v1/jobs/${row.job_id}/cancel`,
    },
  };
}

function publicModel(row) {
  if (!row) return null;
  return {
    modelSha256: row.model_sha256,
    projectId: row.project_id,
    confirmedBy: row.confirmed_by,
    confirmedAt: new Date(row.confirmed_at).toISOString(),
    createdAt: new Date(row.created_at).toISOString(),
    links: { self: `/api/v1/models/${row.model_sha256}` },
  };
}

function publicUpload(row) {
  if (!row) return null;
  return {
    id: row.upload_id,
    projectId: row.project_id,
    state: row.state,
    bundleSha256: row.bundle_sha256,
    archiveSha256: row.archive_sha256,
    archiveBytes: Number(row.archive_bytes),
    scientificRoles: parseJson(row.scientific_roles),
    license: parseJson(row.license),
    createdAt: new Date(row.created_at).toISOString(),
    readyAt: row.ready_at ? new Date(row.ready_at).toISOString() : null,
    links: {
      self: `/api/v1/data-uploads/${row.upload_id}`,
      content: `/api/v1/data-uploads/${row.upload_id}/content`,
    },
  };
}

async function appendAuditRecord(database, {
  projectId,
  credentialId = null,
  action,
  resourceType,
  resourceId = null,
  metadata = {},
}) {
  identifier(projectId, "project");
  if (credentialId !== null) identifier(credentialId, "credential");
  if (typeof action !== "string" || !/^[a-z][a-z0-9_.-]{0,79}$/.test(action)) {
    throw new ControlPlaneError("invalid_audit", "audit action is invalid");
  }
  if (typeof resourceType !== "string" || !/^[a-z][a-z0-9_-]{0,39}$/.test(resourceType)) {
    throw new ControlPlaneError("invalid_audit", "audit resource type is invalid");
  }
  if (resourceId !== null) boundedString(resourceId, "audit resource identifier", 1, 180);
  jsonValue(metadata, "audit metadata");
  const inserted = one(await database.query(
    `INSERT INTO sigma_audit_events(
       project_id, credential_id, action, resource_type, resource_id, metadata
     )
     SELECT $1, $2, $3, $4, $5, $6::jsonb
      WHERE $2::text IS NULL OR EXISTS (
        SELECT 1 FROM sigma_project_api_keys
         WHERE project_id = $1 AND credential_id = $2
      )
     RETURNING audit_id`,
    [projectId, credentialId, action, resourceType, resourceId, jsonParameter(metadata)],
  ));
  if (!inserted) throw new ControlPlaneError("invalid_audit_actor", "audit credential does not belong to the project");
}

async function appendEvent(transaction, jobId, eventType, state, payload = {}) {
  const updated = one(await transaction.query(
    `UPDATE sigma_jobs
       SET event_sequence = event_sequence + 1,
           updated_at = transaction_timestamp()
     WHERE job_id = $1
     RETURNING event_sequence`,
    [jobId],
  ));
  if (!updated) throw new ControlPlaneError("unknown_job", "job does not exist");
  await transaction.query(
    `INSERT INTO sigma_job_events(job_id, sequence, event_type, state, payload)
     VALUES ($1, $2, $3, $4, $5::jsonb)`,
    [jobId, updated.event_sequence, eventType, state, jsonParameter(payload)],
  );
  return Number(updated.event_sequence);
}

async function cancelledInTransaction(transaction, row, reason) {
  await transaction.query(
    `UPDATE sigma_jobs
       SET state = 'cancelled', lease_token = NULL, lease_expires_at = NULL,
           finished_at = transaction_timestamp(), updated_at = transaction_timestamp()
     WHERE job_id = $1`,
    [row.job_id],
  );
  if (Number(row.current_attempt) > 0) {
    await transaction.query(
      `UPDATE sigma_job_attempts
         SET state = 'cancelled', finished_at = transaction_timestamp()
       WHERE job_id = $1 AND attempt = $2 AND state = 'running'`,
      [row.job_id, row.current_attempt],
    );
  }
  await appendEvent(transaction, row.job_id, "cancelled", "cancelled", { reason });
  return { state: "cancelled", shouldRetry: false };
}

export class ProductionControlPlane {
  constructor({ database, clock = () => new Date(), leaseSeconds = 45 * 60 } = {}) {
    if (!database?.query || !database?.transaction) {
      throw new Error("ProductionControlPlane requires a transactional database");
    }
    this.database = database;
    this.clock = clock;
    this.leaseSeconds = positiveInteger(leaseSeconds, 45 * 60, 3600, "leaseSeconds");
  }

  async createProject({ projectId, slug, displayName }) {
    identifier(projectId, "project");
    if (typeof slug !== "string" || !/^[a-z0-9][a-z0-9-]{0,62}$/.test(slug)) {
      throw new ControlPlaneError("invalid_project", "project slug is invalid");
    }
    boundedString(displayName, "display name", 1, 120);
    const result = await this.database.query(
      `INSERT INTO sigma_projects(project_id, slug, display_name)
       VALUES ($1, $2, $3)
       ON CONFLICT (project_id) DO UPDATE
         SET display_name = EXCLUDED.display_name, updated_at = transaction_timestamp()
       RETURNING project_id, slug, display_name, state,
                 max_active_jobs, max_upload_bytes, max_attempts_per_job`,
      [projectId, slug, displayName],
    );
    const row = one(result);
    return {
      id: row.project_id,
      slug: row.slug,
      displayName: row.display_name,
      state: row.state,
      limits: {
        maxActiveJobs: Number(row.max_active_jobs),
        maxUploadBytes: Number(row.max_upload_bytes),
        maxAttemptsPerJob: Number(row.max_attempts_per_job),
      },
    };
  }

  async appendAudit({ projectId, credentialId = null, action, resourceType, resourceId = null, metadata = {} }) {
    await appendAuditRecord(this.database, { projectId, credentialId, action, resourceType, resourceId, metadata });
  }

  async registerModel({
    projectId,
    modelSha256,
    canonicalObjectReference,
    confirmationObjectReference,
    confirmedBy,
    confirmedAt,
    auditCredentialId = null,
  }) {
    identifier(projectId, "project");
    hash(modelSha256, "modelSha256");
    const canonicalReference = privateReference(canonicalObjectReference, "canonical model object reference");
    const confirmationReference = privateReference(confirmationObjectReference, "confirmation object reference");
    boundedString(confirmedBy, "confirmedBy", 1, 160);
    const confirmedDate = new Date(confirmedAt);
    if (Number.isNaN(confirmedDate.valueOf())) {
      throw new ControlPlaneError("invalid_value", "confirmedAt must be a valid timestamp");
    }
    return this.database.transaction(async (transaction) => {
      const inserted = one(await transaction.query(
        `INSERT INTO sigma_models(
           project_id, model_sha256, canonical_object_ref, confirmation_object_ref,
           confirmed_by, confirmed_at
         ) VALUES ($1, $2, $3::jsonb, $4::jsonb, $5, $6)
         ON CONFLICT (project_id, model_sha256) DO NOTHING
         RETURNING *`,
        [
          projectId, modelSha256, jsonParameter(canonicalReference),
          jsonParameter(confirmationReference), confirmedBy, confirmedDate.toISOString(),
        ],
      ));
      if (inserted) {
        await appendAuditRecord(transaction, {
          projectId,
          credentialId: auditCredentialId,
          action: "model.registered",
          resourceType: "model",
          resourceId: modelSha256,
          metadata: {
            canonicalObjectSha256: canonicalReference.sha256,
            confirmationObjectSha256: confirmationReference.sha256,
          },
        });
        return { created: true, model: publicModel(inserted) };
      }
      const existing = one(await transaction.query(
        "SELECT * FROM sigma_models WHERE project_id = $1 AND model_sha256 = $2 FOR UPDATE",
        [projectId, modelSha256],
      ));
      if (!existing) throw new ControlPlaneError("model_registration_race", "registered model could not be recovered");
      if (
        canonicalJson(parseJson(existing.canonical_object_ref)) !== canonicalJson(canonicalReference)
        || canonicalJson(parseJson(existing.confirmation_object_ref)) !== canonicalJson(confirmationReference)
      ) {
        throw new ControlPlaneError(
          "model_registration_conflict",
          "model hash is already bound to different immutable documents",
        );
      }
      return { created: false, model: publicModel(existing) };
    });
  }

  async getModel(projectId, modelSha256) {
    identifier(projectId, "project");
    hash(modelSha256, "modelSha256");
    const row = one(await this.database.query(
      "SELECT * FROM sigma_models WHERE project_id = $1 AND model_sha256 = $2",
      [projectId, modelSha256],
    ));
    if (!row) throw new ControlPlaneError("unknown_model", "model does not exist");
    return publicModel(row);
  }

  async listModels(projectId, { limit = 100 } = {}) {
    identifier(projectId, "project");
    limit = positiveInteger(limit, 100, 1000, "limit");
    return rows(await this.database.query(
      `SELECT * FROM sigma_models
        WHERE project_id = $1 ORDER BY created_at DESC, model_sha256 LIMIT $2`,
      [projectId, limit],
    )).map(publicModel);
  }

  async registerUpload({
    projectId,
    bundleSha256,
    archiveSha256,
    archiveBytes,
    manifestObjectReference,
    scientificRoles,
    license,
    auditCredentialId = null,
  }) {
    identifier(projectId, "project");
    hash(bundleSha256, "bundleSha256");
    hash(archiveSha256, "archiveSha256");
    archiveBytes = positiveInteger(archiveBytes, undefined, 10 * 1024 * 1024 * 1024, "archiveBytes");
    const manifestReference = privateReference(manifestObjectReference, "upload manifest object reference");
    if (!Array.isArray(scientificRoles) || scientificRoles.length > 32 || scientificRoles.some((value) => typeof value !== "string" || value.length < 1 || value.length > 80)) {
      throw new ControlPlaneError("invalid_value", "scientificRoles must contain at most 32 bounded strings");
    }
    jsonValue(license, "license");
    const uploadId = `upload_${sha256({ projectId, bundleSha256, archiveSha256 }).slice(0, 24)}`;
    return this.database.transaction(async (transaction) => {
      const project = one(await transaction.query(
        "SELECT max_upload_bytes FROM sigma_projects WHERE project_id = $1 AND state = 'active' FOR UPDATE",
        [projectId],
      ));
      if (!project) throw new ControlPlaneError("unknown_project", "project does not exist or is inactive");
      if (archiveBytes > Number(project.max_upload_bytes)) {
        throw new ControlPlaneError("upload_quota_exceeded", "archive exceeds the project's upload byte limit", {
          maximumBytes: Number(project.max_upload_bytes),
        });
      }
      const inserted = one(await transaction.query(
        `INSERT INTO sigma_uploads(
           upload_id, project_id, state, bundle_sha256, archive_sha256, archive_bytes,
           manifest_object_ref, scientific_roles, license
         ) VALUES ($1, $2, 'pending', $3, $4, $5, $6::jsonb, $7::jsonb, $8::jsonb)
         ON CONFLICT (project_id, bundle_sha256, archive_sha256) DO NOTHING
         RETURNING *`,
        [
          uploadId, projectId, bundleSha256, archiveSha256, archiveBytes,
          jsonParameter(manifestReference), jsonParameter([...new Set(scientificRoles)].sort()),
          jsonParameter(license),
        ],
      ));
      if (inserted) {
        await appendAuditRecord(transaction, {
          projectId,
          credentialId: auditCredentialId,
          action: "upload.registered",
          resourceType: "upload",
          resourceId: uploadId,
          metadata: { bundleSha256, archiveSha256, archiveBytes, scientificRoles: [...new Set(scientificRoles)].sort() },
        });
        return { created: true, upload: publicUpload(inserted) };
      }
      const existing = one(await transaction.query(
        `SELECT * FROM sigma_uploads
          WHERE project_id = $1 AND bundle_sha256 = $2 AND archive_sha256 = $3 FOR UPDATE`,
        [projectId, bundleSha256, archiveSha256],
      ));
      if (!existing) throw new ControlPlaneError("upload_registration_race", "registered upload could not be recovered");
      if (
        Number(existing.archive_bytes) !== archiveBytes
        || canonicalJson(parseJson(existing.manifest_object_ref)) !== canonicalJson(manifestReference)
        || canonicalJson(parseJson(existing.license)) !== canonicalJson(license)
      ) {
        throw new ControlPlaneError(
          "upload_registration_conflict",
          "upload hashes are already bound to different immutable metadata",
        );
      }
      return { created: false, upload: publicUpload(existing) };
    });
  }

  async finalizeUpload({ projectId, uploadId, archiveObjectReference, auditCredentialId = null }) {
    identifier(projectId, "project");
    identifier(uploadId, "upload");
    const reference = privateReference(archiveObjectReference, "upload archive object reference");
    return this.database.transaction(async (transaction) => {
      let row = one(await transaction.query(
        "SELECT * FROM sigma_uploads WHERE project_id = $1 AND upload_id = $2 FOR UPDATE",
        [projectId, uploadId],
      ));
      if (!row) throw new ControlPlaneError("unknown_upload", "upload does not exist");
      if (reference.sha256 !== row.archive_sha256 || reference.bytes !== Number(row.archive_bytes)) {
        throw new ControlPlaneError("upload_identity_mismatch", "archive object does not match the registered hash and byte count");
      }
      if (row.state === "ready") {
        if (canonicalJson(parseJson(row.archive_object_ref)) !== canonicalJson(reference)) {
          throw new ControlPlaneError("upload_registration_conflict", "ready upload is bound to a different archive object");
        }
        return { finalized: false, upload: publicUpload(row) };
      }
      if (row.state !== "pending") throw new ControlPlaneError("upload_not_pending", "upload cannot accept content in its current state");
      row = one(await transaction.query(
        `UPDATE sigma_uploads
            SET state = 'ready', archive_object_ref = $3::jsonb, ready_at = transaction_timestamp()
          WHERE project_id = $1 AND upload_id = $2 RETURNING *`,
        [projectId, uploadId, jsonParameter(reference)],
      ));
      await appendAuditRecord(transaction, {
        projectId,
        credentialId: auditCredentialId,
        action: "upload.finalized",
        resourceType: "upload",
        resourceId: uploadId,
        metadata: { archiveSha256: reference.sha256, archiveBytes: reference.bytes },
      });
      return { finalized: true, upload: publicUpload(row) };
    });
  }

  async getUpload(projectId, uploadId, { includeReferences = false } = {}) {
    identifier(projectId, "project");
    identifier(uploadId, "upload");
    const row = one(await this.database.query(
      "SELECT * FROM sigma_uploads WHERE project_id = $1 AND upload_id = $2",
      [projectId, uploadId],
    ));
    if (!row) throw new ControlPlaneError("unknown_upload", "upload does not exist");
    const result = publicUpload(row);
    if (includeReferences) {
      result.manifestObjectReference = parseJson(row.manifest_object_ref);
      result.archiveObjectReference = parseJson(row.archive_object_ref);
    }
    return result;
  }

  async listUploads(projectId, { limit = 100 } = {}) {
    identifier(projectId, "project");
    limit = positiveInteger(limit, 100, 1000, "limit");
    return rows(await this.database.query(
      `SELECT * FROM sigma_uploads
        WHERE project_id = $1 ORDER BY created_at DESC, upload_id LIMIT $2`,
      [projectId, limit],
    )).map(publicUpload);
  }

  async createJob({
    projectId,
    jobType,
    idempotencyKey,
    requestSha256,
    requestObjectReference,
    modelSha256 = null,
    dataUploadId = null,
    parameterPolicy,
    maxAttempts = 4,
    auditCredentialId = null,
  }) {
    identifier(projectId, "project");
    if (!JOB_TYPES.has(jobType)) throw new ControlPlaneError("invalid_job_type", "job type is invalid");
    boundedString(idempotencyKey, "idempotency key", 8, 160);
    hash(requestSha256, "requestSha256");
    const requestReference = privateReference(requestObjectReference, "request object reference");
    if (requestReference.sha256 !== requestSha256) {
      throw new ControlPlaneError("request_hash_mismatch", "request object reference does not match requestSha256");
    }
    if (modelSha256 !== null) hash(modelSha256, "modelSha256");
    if (dataUploadId !== null) identifier(dataUploadId, "upload");
    jsonValue(parameterPolicy, "parameterPolicy");
    maxAttempts = positiveInteger(maxAttempts, 4, 12, "maxAttempts");
    const identity = sha256({ projectId, jobType, idempotencyKey });
    const jobId = `job_${identity.slice(0, 24)}`;
    const outboxId = `outbox_${sha256({ jobId, topic: CONTROL_PLANE_TOPIC }).slice(0, 24)}`;
    const payload = {
      schemaVersion: "sigma-production-job-message/1",
      projectId,
      jobId,
      jobType,
      requestSha256,
    };
    return this.database.transaction(async (transaction) => {
      const project = one(await transaction.query(
        `SELECT max_active_jobs, max_attempts_per_job
           FROM sigma_projects WHERE project_id = $1 AND state = 'active' FOR UPDATE`,
        [projectId],
      ));
      if (!project) throw new ControlPlaneError("unknown_project", "project does not exist or is inactive");
      if (maxAttempts > Number(project.max_attempts_per_job)) {
        throw new ControlPlaneError("attempt_quota_exceeded", "maxAttempts exceeds the project limit", {
          maximum: Number(project.max_attempts_per_job),
        });
      }
      if (modelSha256 !== null) {
        const model = one(await transaction.query(
          "SELECT 1 AS present FROM sigma_models WHERE project_id = $1 AND model_sha256 = $2",
          [projectId, modelSha256],
        ));
        if (!model) throw new ControlPlaneError("unknown_model", "model is not registered in this project");
      }
      if (dataUploadId !== null) {
        const upload = one(await transaction.query(
          "SELECT state FROM sigma_uploads WHERE project_id = $1 AND upload_id = $2",
          [projectId, dataUploadId],
        ));
        if (!upload) throw new ControlPlaneError("unknown_upload", "upload is not registered in this project");
        if (upload.state !== "ready") throw new ControlPlaneError("upload_not_ready", "upload content is not ready");
      }
      const existing = one(await transaction.query(
        `SELECT * FROM sigma_jobs
          WHERE project_id = $1 AND job_type = $2 AND idempotency_key = $3
          FOR UPDATE`,
        [projectId, jobType, idempotencyKey],
      ));
      if (existing) {
        if (
          existing.request_sha256 !== requestSha256
          || existing.model_sha256 !== modelSha256
          || existing.input_upload_id !== dataUploadId
        ) {
          throw new ControlPlaneError(
            "idempotency_conflict",
            "idempotency key is already bound to different scientific inputs",
            { existingJobId: existing.job_id },
          );
        }
        return { created: false, job: publicJob(existing) };
      }
      const active = one(await transaction.query(
        `SELECT count(*)::int AS count FROM sigma_jobs
          WHERE project_id = $1
            AND state IN ('dispatch_pending', 'queued', 'running', 'cancel_requested')`,
        [projectId],
      ));
      if (Number(active.count) >= Number(project.max_active_jobs)) {
        throw new ControlPlaneError("active_job_quota_exceeded", "project has reached its active-job limit", {
          maximum: Number(project.max_active_jobs),
        });
      }
      const inserted = one(await transaction.query(
        `INSERT INTO sigma_jobs(
           job_id, project_id, job_type, state, idempotency_key, request_sha256,
           request_object_ref, model_sha256, input_upload_id, parameter_policy, max_attempts
         ) VALUES ($1, $2, $3, 'dispatch_pending', $4, $5, $6::jsonb, $7, $8, $9::jsonb, $10)
         ON CONFLICT (project_id, job_type, idempotency_key) DO NOTHING
         RETURNING *`,
        [
          jobId, projectId, jobType, idempotencyKey, requestSha256,
          jsonParameter(requestReference), modelSha256, dataUploadId,
          jsonParameter(parameterPolicy), maxAttempts,
        ],
      ));
      if (!inserted) {
        const raced = one(await transaction.query(
          `SELECT * FROM sigma_jobs
           WHERE project_id = $1 AND job_type = $2 AND idempotency_key = $3
           FOR UPDATE`,
          [projectId, jobType, idempotencyKey],
        ));
        if (!raced) throw new ControlPlaneError("idempotency_race", "idempotent job could not be recovered");
        if (
          raced.request_sha256 !== requestSha256
          || raced.model_sha256 !== modelSha256
          || raced.input_upload_id !== dataUploadId
        ) {
          throw new ControlPlaneError(
            "idempotency_conflict",
            "idempotency key is already bound to different scientific inputs",
            { existingJobId: raced.job_id },
          );
        }
        return { created: false, job: publicJob(raced) };
      }
      await transaction.query(
        `INSERT INTO sigma_job_events(job_id, sequence, event_type, state, payload)
         VALUES ($1, 1, 'accepted', 'dispatch_pending', $2::jsonb)`,
        [jobId, jsonParameter({ requestSha256, jobType })],
      );
      await transaction.query(
        `INSERT INTO sigma_outbox(
           outbox_id, project_id, job_id, topic, idempotency_key, payload
         ) VALUES ($1, $2, $3, $4, $5, $6::jsonb)`,
        [outboxId, projectId, jobId, CONTROL_PLANE_TOPIC, `job-dispatch-${jobId}`, jsonParameter(payload)],
      );
      await appendAuditRecord(transaction, {
        projectId,
        credentialId: auditCredentialId,
        action: "job.submitted",
        resourceType: "job",
        resourceId: jobId,
        metadata: { jobType, requestSha256, modelSha256, dataUploadId, parameterPolicy },
      });
      return { created: true, job: publicJob(inserted) };
    });
  }

  async getJob(projectId, jobId) {
    identifier(projectId, "project");
    identifier(jobId, "job");
    const row = one(await this.database.query(
      "SELECT * FROM sigma_jobs WHERE project_id = $1 AND job_id = $2",
      [projectId, jobId],
    ));
    if (!row) throw new ControlPlaneError("unknown_job", "job does not exist");
    return publicJob(row);
  }

  async findJobByIdempotency(projectId, jobType, idempotencyKey) {
    identifier(projectId, "project");
    if (!JOB_TYPES.has(jobType)) throw new ControlPlaneError("invalid_job_type", "job type is invalid");
    boundedString(idempotencyKey, "idempotency key", 8, 160);
    const row = one(await this.database.query(
      `SELECT * FROM sigma_jobs
        WHERE project_id = $1 AND job_type = $2 AND idempotency_key = $3`,
      [projectId, jobType, idempotencyKey],
    ));
    return row ? {
      job: publicJob(row),
      requestObjectReference: parseJson(row.request_object_ref),
      inputUploadId: row.input_upload_id,
      modelSha256: row.model_sha256,
    } : null;
  }

  async listJobs(projectId, { limit = 100, jobType = null } = {}) {
    identifier(projectId, "project");
    limit = positiveInteger(limit, 100, 1000, "limit");
    if (jobType !== null && !JOB_TYPES.has(jobType)) {
      throw new ControlPlaneError("invalid_job_type", "job type is invalid");
    }
    return rows(await this.database.query(
      `SELECT * FROM sigma_jobs
        WHERE project_id = $1 AND ($2::text IS NULL OR job_type = $2)
        ORDER BY created_at DESC, job_id LIMIT $3`,
      [projectId, jobType, limit],
    )).map(publicJob);
  }

  async getEvents(projectId, jobId) {
    await this.getJob(projectId, jobId);
    return rows(await this.database.query(
      `SELECT sequence, event_type, state, payload, created_at
       FROM sigma_job_events WHERE job_id = $1 ORDER BY sequence`,
      [jobId],
    )).map((row) => ({
      sequence: Number(row.sequence),
      type: row.event_type,
      state: row.state,
      payload: parseJson(row.payload),
      createdAt: new Date(row.created_at).toISOString(),
    }));
  }

  async getArtifacts(projectId, jobId) {
    await this.getJob(projectId, jobId);
    return rows(await this.database.query(
      `SELECT name, object_ref, sha256, bytes, media_type, created_at
       FROM sigma_job_artifacts WHERE job_id = $1 ORDER BY name`,
      [jobId],
    )).map((row) => ({
      name: row.name,
      objectReference: parseJson(row.object_ref),
      sha256: row.sha256,
      bytes: Number(row.bytes),
      mediaType: row.media_type,
      createdAt: new Date(row.created_at).toISOString(),
    }));
  }

  async getArtifact(projectId, jobId, name) {
    identifier(projectId, "project");
    identifier(jobId, "job");
    if (typeof name !== "string" || !ARTIFACT_NAME.test(name)) {
      throw new ControlPlaneError("invalid_artifact", "artifact name is invalid");
    }
    await this.getJob(projectId, jobId);
    const row = one(await this.database.query(
      `SELECT name, object_ref, sha256, bytes, media_type, created_at
         FROM sigma_job_artifacts WHERE job_id = $1 AND name = $2`,
      [jobId, name],
    ));
    if (!row) throw new ControlPlaneError("unknown_artifact", "artifact does not exist");
    return {
      name: row.name,
      objectReference: parseJson(row.object_ref),
      sha256: row.sha256,
      bytes: Number(row.bytes),
      mediaType: row.media_type,
      createdAt: new Date(row.created_at).toISOString(),
    };
  }

  async claimOutbox({ leaseSeconds = 60 } = {}) {
    leaseSeconds = positiveInteger(leaseSeconds, 60, 600, "outbox leaseSeconds");
    const leaseToken = randomUUID();
    const result = await this.database.query(
      `WITH candidate AS (
         SELECT outbox_id FROM sigma_outbox
          WHERE (
            (state = 'pending' AND next_attempt_at <= transaction_timestamp())
            OR (state = 'publishing' AND publish_lease_expires_at < transaction_timestamp())
          )
          ORDER BY created_at, outbox_id
          FOR UPDATE SKIP LOCKED
          LIMIT 1
       )
       UPDATE sigma_outbox AS outbox
          SET state = 'publishing',
              publish_attempts = publish_attempts + 1,
              publish_lease_token = $1,
              publish_lease_expires_at = transaction_timestamp() + ($2 * interval '1 second')
         FROM candidate
        WHERE outbox.outbox_id = candidate.outbox_id
       RETURNING outbox.*`,
      [leaseToken, leaseSeconds],
    );
    const row = one(result);
    if (!row) return null;
    return {
      id: row.outbox_id,
      jobId: row.job_id,
      topic: row.topic,
      idempotencyKey: row.idempotency_key,
      payload: parseJson(row.payload),
      attempts: Number(row.publish_attempts),
      leaseToken,
    };
  }

  async markOutboxPublished({ outboxId, leaseToken, messageId = null }) {
    identifier(outboxId, "outbox");
    boundedString(leaseToken, "outbox lease token", 16, 160);
    if (messageId !== null) boundedString(messageId, "queue message id", 1, 256);
    return this.database.transaction(async (transaction) => {
      const outbox = one(await transaction.query(
        `UPDATE sigma_outbox
            SET state = 'published', queue_message_id = $3,
                publish_lease_token = NULL, publish_lease_expires_at = NULL,
                published_at = transaction_timestamp(), last_error = NULL
          WHERE outbox_id = $1 AND state = 'publishing' AND publish_lease_token = $2
          RETURNING job_id`,
        [outboxId, leaseToken, messageId],
      ));
      if (!outbox) return false;
      const transitioned = one(await transaction.query(
        `UPDATE sigma_jobs
            SET state = 'queued', queue_message_id = $2, updated_at = transaction_timestamp()
          WHERE job_id = $1 AND state = 'dispatch_pending'
          RETURNING job_id`,
        [outbox.job_id, messageId],
      ));
      if (transitioned) {
        await appendEvent(transaction, outbox.job_id, "queued", "queued", { messageId });
      }
      return true;
    });
  }

  async releaseOutbox({ outboxId, leaseToken, error }) {
    identifier(outboxId, "outbox");
    boundedString(leaseToken, "outbox lease token", 16, 160);
    const safe = safeError(error);
    return this.database.transaction(async (transaction) => {
      const outbox = one(await transaction.query(
        `UPDATE sigma_outbox
            SET state = CASE WHEN publish_attempts >= 12 THEN 'dead' ELSE 'pending' END,
                publish_lease_token = NULL, publish_lease_expires_at = NULL,
                next_attempt_at = transaction_timestamp()
                  + (LEAST(300, power(2, LEAST(publish_attempts, 8))) * interval '1 second'),
                last_error = $3::jsonb
          WHERE outbox_id = $1 AND state = 'publishing' AND publish_lease_token = $2
          RETURNING job_id, state`,
        [outboxId, leaseToken, jsonParameter(safe)],
      ));
      if (!outbox) return { released: false, dead: false };
      if (outbox.state === "dead") {
        const transitioned = one(await transaction.query(
          `UPDATE sigma_jobs
              SET state = 'failed', error = $2::jsonb,
                  finished_at = transaction_timestamp(), updated_at = transaction_timestamp()
            WHERE job_id = $1 AND state = 'dispatch_pending'
            RETURNING job_id`,
          [outbox.job_id, jsonParameter({ ...safe, phase: "queue_dispatch" })],
        ));
        if (transitioned) {
          await appendEvent(transaction, outbox.job_id, "dispatch_failed", "failed", safe);
        }
      }
      return { released: true, dead: outbox.state === "dead" };
    });
  }

  async dispatchAvailable(publisher, { limit = 16 } = {}) {
    if (!publisher?.send) throw new Error("publisher with send() is required");
    limit = positiveInteger(limit, 16, 100, "dispatch limit");
    const results = [];
    for (let index = 0; index < limit; index += 1) {
      const item = await this.claimOutbox();
      if (!item) break;
      try {
        const published = await publisher.send(item.topic, item.payload, {
          idempotencyKey: item.idempotencyKey,
          retentionSeconds: 604800,
        });
        await this.markOutboxPublished({
          outboxId: item.id,
          leaseToken: item.leaseToken,
          messageId: published?.messageId ?? null,
        });
        results.push({ outboxId: item.id, state: "published", messageId: published?.messageId ?? null });
      } catch (error) {
        const released = await this.releaseOutbox({ outboxId: item.id, leaseToken: item.leaseToken, error });
        results.push({ outboxId: item.id, state: released.dead ? "dead" : "retry_pending" });
      }
    }
    return results;
  }

  async claimJob({ jobId, messageId, deliveryCount, workerIdentity, leaseSeconds = this.leaseSeconds }) {
    identifier(jobId, "job");
    boundedString(messageId, "queue message id", 1, 256);
    deliveryCount = positiveInteger(deliveryCount, 1, 1000000, "deliveryCount");
    boundedString(workerIdentity, "workerIdentity", 1, 256);
    leaseSeconds = positiveInteger(leaseSeconds, this.leaseSeconds, 3600, "leaseSeconds");
    return this.database.transaction(async (transaction) => {
      let row = one(await transaction.query("SELECT * FROM sigma_jobs WHERE job_id = $1 FOR UPDATE", [jobId]));
      if (!row) throw new ControlPlaneError("unknown_job", "job does not exist");
      if (TERMINAL.has(row.state)) return { claimed: false, reason: "terminal", job: publicJob(row) };
      if (row.cancellation_requested_at || row.state === "cancel_requested") {
        await cancelledInTransaction(transaction, row, "cancelled_before_worker_claim");
        row = one(await transaction.query("SELECT * FROM sigma_jobs WHERE job_id = $1", [jobId]));
        return { claimed: false, reason: "cancelled", job: publicJob(row) };
      }
      if (row.state === "running" && new Date(row.lease_expires_at) > this.clock()) {
        return { claimed: false, reason: "leased", job: publicJob(row) };
      }
      if (row.state === "running") {
        await transaction.query(
          `UPDATE sigma_job_attempts SET state = 'lease_expired', finished_at = transaction_timestamp()
           WHERE job_id = $1 AND attempt = $2 AND state = 'running'`,
          [jobId, row.current_attempt],
        );
      }
      if (Number(row.current_attempt) >= Number(row.max_attempts)) {
        const error = { code: "attempts_exhausted", message: "Maximum worker attempts were exhausted" };
        await transaction.query(
          `UPDATE sigma_jobs SET state = 'failed', error = $2::jsonb,
             lease_token = NULL, lease_expires_at = NULL,
             finished_at = transaction_timestamp(), updated_at = transaction_timestamp()
           WHERE job_id = $1`,
          [jobId, jsonParameter(error)],
        );
        await appendEvent(transaction, jobId, "attempts_exhausted", "failed", error);
        row = one(await transaction.query("SELECT * FROM sigma_jobs WHERE job_id = $1", [jobId]));
        return { claimed: false, reason: "attempts_exhausted", job: publicJob(row) };
      }
      const attempt = Number(row.current_attempt) + 1;
      const leaseToken = randomUUID();
      row = one(await transaction.query(
        `UPDATE sigma_jobs
            SET state = 'running', current_attempt = $2, lease_token = $3,
                lease_expires_at = transaction_timestamp() + ($4 * interval '1 second'),
                queue_message_id = $5, started_at = COALESCE(started_at, transaction_timestamp()),
                updated_at = transaction_timestamp()
          WHERE job_id = $1
          RETURNING *`,
        [jobId, attempt, leaseToken, leaseSeconds, messageId],
      ));
      await transaction.query(
        `INSERT INTO sigma_job_attempts(
           job_id, attempt, lease_token, queue_message_id, delivery_count,
           worker_identity, state, lease_expires_at
         ) VALUES (
           $1, $2, $3, $4, $5, $6, 'running',
           transaction_timestamp() + ($7 * interval '1 second')
         )`,
        [jobId, attempt, leaseToken, messageId, deliveryCount, workerIdentity, leaseSeconds],
      );
      await appendEvent(transaction, jobId, "worker_claimed", "running", {
        attempt,
        deliveryCount,
        messageId,
        workerIdentity,
      });
      return {
        claimed: true,
        leaseToken,
        attempt,
        job: publicJob(row),
        requestObjectReference: parseJson(row.request_object_ref),
      };
    });
  }

  async requestCancellation(projectId, jobId, { auditCredentialId = null } = {}) {
    identifier(projectId, "project");
    identifier(jobId, "job");
    return this.database.transaction(async (transaction) => {
      let row = one(await transaction.query(
        "SELECT * FROM sigma_jobs WHERE project_id = $1 AND job_id = $2 FOR UPDATE",
        [projectId, jobId],
      ));
      if (!row) throw new ControlPlaneError("unknown_job", "job does not exist");
      if (TERMINAL.has(row.state)) {
        await appendAuditRecord(transaction, {
          projectId,
          credentialId: auditCredentialId,
          action: "job.cancellation_requested",
          resourceType: "job",
          resourceId: jobId,
          metadata: { resultingState: row.state, alreadyTerminal: true },
        });
        return publicJob(row);
      }
      if (row.state === "running") {
        row = one(await transaction.query(
          `UPDATE sigma_jobs
              SET state = 'cancel_requested', cancellation_requested_at = transaction_timestamp(),
                  updated_at = transaction_timestamp()
            WHERE job_id = $1 RETURNING *`,
          [jobId],
        ));
        await appendEvent(transaction, jobId, "cancellation_requested", "cancel_requested", {});
        await appendAuditRecord(transaction, {
          projectId,
          credentialId: auditCredentialId,
          action: "job.cancellation_requested",
          resourceType: "job",
          resourceId: jobId,
          metadata: { resultingState: "cancel_requested", alreadyTerminal: false },
        });
        return publicJob(row);
      }
      await transaction.query(
        `UPDATE sigma_jobs
            SET cancellation_requested_at = transaction_timestamp()
          WHERE job_id = $1`,
        [jobId],
      );
      await cancelledInTransaction(transaction, { ...row, cancellation_requested_at: this.clock() }, "cancelled_before_start");
      row = one(await transaction.query("SELECT * FROM sigma_jobs WHERE job_id = $1", [jobId]));
      await appendAuditRecord(transaction, {
        projectId,
        credentialId: auditCredentialId,
        action: "job.cancellation_requested",
        resourceType: "job",
        resourceId: jobId,
        metadata: { resultingState: row.state, alreadyTerminal: false },
      });
      return publicJob(row);
    });
  }

  async completeJob({ jobId, leaseToken, resultManifestReference, artifacts }) {
    identifier(jobId, "job");
    boundedString(leaseToken, "lease token", 16, 160);
    const manifest = privateReference(resultManifestReference, "result manifest reference");
    if (!Array.isArray(artifacts) || artifacts.length < 1 || artifacts.length > 512) {
      throw new ControlPlaneError("invalid_artifacts", "artifacts must contain 1-512 entries");
    }
    const normalizedArtifacts = artifacts.map((artifact) => {
      if (!artifact || typeof artifact.name !== "string" || !ARTIFACT_NAME.test(artifact.name)) {
        throw new ControlPlaneError("invalid_artifact", "artifact name is invalid");
      }
      const reference = privateReference(artifact.objectReference, `artifact ${artifact.name}`);
      if (artifact.sha256 !== reference.sha256 || Number(artifact.bytes) !== reference.bytes) {
        throw new ControlPlaneError("artifact_identity_mismatch", `artifact ${artifact.name} metadata changed`);
      }
      boundedString(artifact.mediaType, "artifact media type", 3, 160);
      return { ...artifact, objectReference: reference, bytes: Number(artifact.bytes) };
    });
    if (new Set(normalizedArtifacts.map((value) => value.name)).size !== normalizedArtifacts.length) {
      throw new ControlPlaneError("duplicate_artifact", "artifact names must be unique");
    }
    return this.database.transaction(async (transaction) => {
      let row = one(await transaction.query("SELECT * FROM sigma_jobs WHERE job_id = $1 FOR UPDATE", [jobId]));
      if (!row) throw new ControlPlaneError("unknown_job", "job does not exist");
      if (row.state === "succeeded") return { state: "succeeded", idempotent: true, job: publicJob(row) };
      if (row.cancellation_requested_at || row.state === "cancel_requested") {
        const cancelled = await cancelledInTransaction(transaction, row, "cancelled_during_execution");
        return { ...cancelled, idempotent: false };
      }
      if (row.state !== "running" || row.lease_token !== leaseToken) {
        throw new ControlPlaneError("lost_lease", "worker no longer owns this job lease");
      }
      for (const artifact of normalizedArtifacts) {
        await transaction.query(
          `INSERT INTO sigma_job_artifacts(job_id, name, object_ref, sha256, bytes, media_type)
           VALUES ($1, $2, $3::jsonb, $4, $5, $6)`,
          [
            jobId, artifact.name, jsonParameter(artifact.objectReference),
            artifact.sha256, artifact.bytes, artifact.mediaType,
          ],
        );
      }
      await transaction.query(
        `UPDATE sigma_job_attempts
            SET state = 'succeeded', finished_at = transaction_timestamp()
          WHERE job_id = $1 AND attempt = $2 AND lease_token = $3`,
        [jobId, row.current_attempt, leaseToken],
      );
      row = one(await transaction.query(
        `UPDATE sigma_jobs
            SET state = 'succeeded', result_manifest_ref = $2::jsonb,
                lease_token = NULL, lease_expires_at = NULL, error = NULL,
                finished_at = transaction_timestamp(), updated_at = transaction_timestamp()
          WHERE job_id = $1 RETURNING *`,
        [jobId, jsonParameter(manifest)],
      ));
      await appendEvent(transaction, jobId, "succeeded", "succeeded", {
        artifactCount: normalizedArtifacts.length,
        resultManifestSha256: manifest.sha256,
      });
      return { state: "succeeded", idempotent: false, job: publicJob(row) };
    });
  }

  async failJob({ jobId, leaseToken, error, retryable = true }) {
    identifier(jobId, "job");
    boundedString(leaseToken, "lease token", 16, 160);
    const safe = safeError(error);
    return this.database.transaction(async (transaction) => {
      let row = one(await transaction.query("SELECT * FROM sigma_jobs WHERE job_id = $1 FOR UPDATE", [jobId]));
      if (!row) throw new ControlPlaneError("unknown_job", "job does not exist");
      if (TERMINAL.has(row.state)) return { state: row.state, shouldRetry: false, idempotent: true };
      if (row.cancellation_requested_at || row.state === "cancel_requested") {
        return cancelledInTransaction(transaction, row, "cancelled_after_worker_error");
      }
      if (row.state !== "running" || row.lease_token !== leaseToken) {
        throw new ControlPlaneError("lost_lease", "worker no longer owns this job lease");
      }
      const shouldRetry = Boolean(retryable) && Number(row.current_attempt) < Number(row.max_attempts);
      const state = shouldRetry ? "queued" : "failed";
      await transaction.query(
        `UPDATE sigma_job_attempts
            SET state = $4, error = $5::jsonb, finished_at = transaction_timestamp()
          WHERE job_id = $1 AND attempt = $2 AND lease_token = $3`,
        [jobId, row.current_attempt, leaseToken, shouldRetry ? "retryable_failure" : "failed", jsonParameter(safe)],
      );
      row = one(await transaction.query(
        `UPDATE sigma_jobs
            SET state = $2, error = $3::jsonb, lease_token = NULL, lease_expires_at = NULL,
                finished_at = CASE WHEN $2 = 'failed' THEN transaction_timestamp() ELSE NULL END,
                updated_at = transaction_timestamp()
          WHERE job_id = $1 RETURNING *`,
        [jobId, state, jsonParameter(safe)],
      ));
      await appendEvent(
        transaction,
        jobId,
        shouldRetry ? "retry_scheduled" : "failed",
        state,
        { ...safe, attempt: Number(row.current_attempt) },
      );
      return { state, shouldRetry, idempotent: false, job: publicJob(row) };
    });
  }
}
