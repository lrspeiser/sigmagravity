import { authenticateProjectRequest, ProductionAuthError, requestHeader } from "./production-auth.mjs";
import { ControlPlaneError } from "./production-control-plane.mjs";
import { getProductionRuntime, ProductionRuntimeError } from "./production-runtime.mjs";
import { options, parseBody, send, setCors } from "./http.mjs";

const DEFAULT_GATEWAY_UPLOAD_BYTES = 4 * 1024 * 1024;

function gatewayUploadLimit(environment = process.env) {
  const value = environment.SIGMA_GATEWAY_UPLOAD_MAX_BYTES;
  if (value === undefined) return DEFAULT_GATEWAY_UPLOAD_BYTES;
  const parsed = Number(value);
  if (!Number.isSafeInteger(parsed) || parsed < 1 || parsed > 16 * 1024 * 1024) {
    throw new ControlPlaneError("gateway_upload_limit_invalid", "SIGMA_GATEWAY_UPLOAD_MAX_BYTES is invalid");
  }
  return parsed;
}

function integerQuery(value, fallback = 100) {
  if (value === undefined) return fallback;
  const parsed = Number(value);
  if (!Number.isSafeInteger(parsed) || parsed < 1 || parsed > 1000) {
    throw new ControlPlaneError("invalid_request", "limit must be an integer from 1 to 1000");
  }
  return parsed;
}

function errorStatus(error) {
  if (Number.isInteger(error?.statusCode)) return error.statusCode;
  if (error instanceof ProductionAuthError) return 401;
  if (error instanceof ProductionRuntimeError) return 503;
  if (error instanceof SyntaxError) return 400;
  if (!(error instanceof ControlPlaneError)) return 500;
  if (error.code.startsWith("unknown_")) return 404;
  if (error.code.includes("conflict") || error.code === "idempotency_conflict") return 409;
  if (error.code.includes("quota")) return 429;
  if (error.code === "request_too_large" || error.code === "direct_upload_required") return 413;
  if (error.code.includes("not_configured")) return 503;
  if (error.code === "production_database_migration_required") return 503;
  if (error.code === "production_execution_not_ready") return 503;
  return 422;
}

function sendError(response, error) {
  const status = errorStatus(error);
  const safe = status === 500
    ? { error: "server_error", message: "The production research API could not complete the request" }
    : {
        error: error.code ?? "invalid_request",
        message: error.message,
        ...(error.details && Object.keys(error.details).length ? { details: error.details } : {}),
      };
  if (status === 401) response.setHeader("WWW-Authenticate", 'Bearer realm="sigma-project"');
  send(response, status, safe);
}

async function context(request, runtimeFactory) {
  const runtime = runtimeFactory();
  let auth;
  try {
    auth = await authenticateProjectRequest(request, runtime.database);
  } catch (error) {
    if (error?.code === "42P01" || error?.code === "42703") {
      throw new ControlPlaneError(
        "production_database_migration_required",
        "The production database is configured but the required research API migrations are not verified",
      );
    }
    throw error;
  }
  return { ...runtime, auth };
}

async function readRequestBytes(request, maximumBytes) {
  const declared = requestHeader(request, "content-length");
  if (declared !== undefined) {
    const length = Number(declared);
    if (!Number.isSafeInteger(length) || length < 0 || length > maximumBytes) {
      throw new ControlPlaneError("request_too_large", `request exceeds the ${maximumBytes} byte upload limit`);
    }
  }
  if (Buffer.isBuffer(request.body)) {
    if (request.body.length > maximumBytes) throw new ControlPlaneError("request_too_large", "upload body is too large");
    return request.body;
  }
  if (request.body instanceof Uint8Array) {
    const bytes = Buffer.from(request.body);
    if (bytes.length > maximumBytes) throw new ControlPlaneError("request_too_large", "upload body is too large");
    return bytes;
  }
  if (typeof request.body === "string") {
    throw new ControlPlaneError(
      "invalid_upload_body",
      "upload bytes were text-decoded before verification; send an unparsed application/octet-stream body",
    );
  }
  if (!request[Symbol.asyncIterator]) {
    throw new ControlPlaneError("invalid_upload_body", "upload content must be an unparsed binary request body");
  }
  const chunks = [];
  let total = 0;
  for await (const chunk of request) {
    const bytes = Buffer.from(chunk);
    total += bytes.length;
    if (total > maximumBytes) {
      throw new ControlPlaneError("request_too_large", `request exceeds the ${maximumBytes} byte upload limit`);
    }
    chunks.push(bytes);
  }
  return Buffer.concat(chunks, total);
}

export async function handleProductionModels(request, response, { runtimeFactory = getProductionRuntime } = {}) {
  if (options(request, response)) return;
  if (!["GET", "POST"].includes(request.method)) {
    response.setHeader("Allow", "OPTIONS, GET, POST");
    send(response, 405, { error: "method_not_allowed", allowed: ["GET", "POST"] });
    return;
  }
  response.setHeader("Cache-Control", "private, no-store");
  try {
    const runtime = await context(request, runtimeFactory);
    const { service, auth } = runtime;
    if (request.method === "GET") {
      const items = await service.listModels(auth, { limit: integerQuery(request.query?.limit) });
      send(response, 200, { schemaVersion: "sigma-production-model-list/1", items });
      return;
    }
    const result = await service.registerModel(auth, parseBody(request));
    send(response, result.created ? 201 : 200, {
      schemaVersion: "sigma-production-model-registration-result/1",
      ...result,
    });
  } catch (error) {
    sendError(response, error);
  }
}

export async function handleProductionModel(request, response, { runtimeFactory = getProductionRuntime } = {}) {
  if (options(request, response)) return;
  if (request.method !== "GET") {
    response.setHeader("Allow", "OPTIONS, GET");
    send(response, 405, { error: "method_not_allowed", allowed: ["GET"] });
    return;
  }
  response.setHeader("Cache-Control", "private, no-store");
  try {
    const runtime = await context(request, runtimeFactory);
    const { service, auth } = runtime;
    send(response, 200, await service.getModel(auth, request.query?.sha256));
  } catch (error) {
    sendError(response, error);
  }
}

export async function handleProductionUploads(request, response, { runtimeFactory = getProductionRuntime } = {}) {
  if (options(request, response)) return;
  if (!["GET", "POST"].includes(request.method)) {
    response.setHeader("Allow", "OPTIONS, GET, POST");
    send(response, 405, { error: "method_not_allowed", allowed: ["GET", "POST"] });
    return;
  }
  response.setHeader("Cache-Control", "private, no-store");
  try {
    const { service, auth } = await context(request, runtimeFactory);
    if (request.method === "GET") {
      const items = await service.listUploads(auth, { limit: integerQuery(request.query?.limit) });
      send(response, 200, { schemaVersion: "sigma-production-upload-list/1", items });
      return;
    }
    const result = await service.registerUpload(auth, parseBody(request));
    send(response, result.created ? 201 : 200, {
      schemaVersion: "sigma-production-upload-registration-result/1",
      ...result,
      requiredContentHeaders: {
        "Content-Type": "application/octet-stream",
        "Content-Length": result.upload.archiveBytes,
        "X-Content-SHA256": result.upload.archiveSha256,
      },
    });
  } catch (error) {
    sendError(response, error);
  }
}

export async function handleProductionUpload(request, response, { runtimeFactory = getProductionRuntime } = {}) {
  if (options(request, response)) return;
  const content = request.query?.resource === "content";
  const allowed = content ? ["PUT"] : ["GET"];
  if (!allowed.includes(request.method)) {
    response.setHeader("Allow", `OPTIONS, ${allowed.join(", ")}`);
    send(response, 405, { error: "method_not_allowed", allowed });
    return;
  }
  response.setHeader("Cache-Control", "private, no-store");
  try {
    const { service, auth } = await context(request, runtimeFactory);
    if (!content) {
      send(response, 200, await service.getUpload(auth, request.query?.id));
      return;
    }
    const upload = await service.getUpload(auth, request.query?.id);
    const gatewayMaximum = gatewayUploadLimit();
    if (upload.archiveBytes > gatewayMaximum) {
      throw new ControlPlaneError(
        "direct_upload_required",
        "This archive exceeds the bounded gateway upload path; private direct-upload tokens are not connected yet",
        { gatewayMaximumBytes: gatewayMaximum, archiveBytes: upload.archiveBytes },
      );
    }
    const bytes = await readRequestBytes(request, Math.min(upload.archiveBytes, gatewayMaximum));
    const result = await service.putUploadContent(auth, upload.id, bytes, {
      suppliedSha256: requestHeader(request, "x-content-sha256") ?? null,
    });
    send(response, 200, { schemaVersion: "sigma-production-upload-finalization-result/1", ...result });
  } catch (error) {
    sendError(response, error);
  }
}

export async function handleProductionJobs(request, response, {
  runtimeFactory = getProductionRuntime,
  expectedJobType = null,
  expectedOperation = null,
} = {}) {
  if (options(request, response)) return;
  if (!["GET", "POST"].includes(request.method)) {
    response.setHeader("Allow", "OPTIONS, GET, POST");
    send(response, 405, { error: "method_not_allowed", allowed: ["GET", "POST"] });
    return;
  }
  response.setHeader("Cache-Control", "private, no-store");
  try {
    const runtime = await context(request, runtimeFactory);
    const { service, auth } = runtime;
    if (request.method === "GET") {
      const items = await service.listJobs(auth, {
        limit: integerQuery(request.query?.limit),
        jobType: expectedJobType,
      });
      send(response, 200, { schemaVersion: "sigma-production-job-list/1", items });
      return;
    }
    if (runtime.jobSubmissionReady !== true) {
      throw new ControlPlaneError(
        "production_execution_not_ready",
        "Job submission remains disabled until the outbox scheduler and stateless scientific worker are verified",
        runtime.executionComponents ?? {},
      );
    }
    const body = parseBody(request);
    if (expectedOperation !== null && body?.request?.operation !== expectedOperation) {
      throw new ControlPlaneError(
        "invalid_job_operation",
        `this endpoint requires request.operation=${expectedOperation}`,
      );
    }
    const result = await service.submitJob(auth, body, {
      expectedJobType,
      idempotencyKey: requestHeader(request, "idempotency-key"),
    });
    send(response, result.created ? 202 : 200, {
      schemaVersion: "sigma-production-job-submission-result/1",
      ...result,
    });
  } catch (error) {
    sendError(response, error);
  }
}

export async function handleProductionJob(request, response, { runtimeFactory = getProductionRuntime } = {}) {
  if (options(request, response)) return;
  const resource = request.query?.resource;
  const method = resource === "cancel" ? "POST" : "GET";
  if (request.method !== method) {
    response.setHeader("Allow", `OPTIONS, ${method}`);
    send(response, 405, { error: "method_not_allowed", allowed: [method] });
    return;
  }
  response.setHeader("Cache-Control", "private, no-store");
  try {
    const { service, auth } = await context(request, runtimeFactory);
    const jobId = request.query?.id;
    if (resource === "events") {
      send(response, 200, { schemaVersion: "sigma-production-job-events/1", items: await service.getEvents(auth, jobId) });
      return;
    }
    if (resource === "artifacts") {
      send(response, 200, { schemaVersion: "sigma-production-job-artifacts/1", items: await service.getArtifacts(auth, jobId) });
      return;
    }
    if (resource === "artifact") {
      const { artifact, bytes } = await service.downloadArtifact(auth, jobId, request.query?.name);
      response.setHeader("Access-Control-Allow-Origin", "*");
      response.setHeader("Content-Type", artifact.mediaType);
      response.setHeader("Content-Length", String(bytes.length));
      response.setHeader("ETag", `"sha256-${artifact.sha256}"`);
      response.setHeader("X-Content-SHA256", artifact.sha256);
      response.status(200);
      if (typeof response.send === "function") response.send(bytes);
      else response.end(bytes);
      return;
    }
    if (resource === "cancel") {
      send(response, 202, await service.cancelJob(auth, jobId));
      return;
    }
    if (resource !== undefined && resource !== null && resource !== "") {
      throw new ControlPlaneError("unknown_job_resource", "job resource does not exist");
    }
    send(response, 200, await service.getJob(auth, jobId));
  } catch (error) {
    sendError(response, error);
  }
}

export { readRequestBytes, sendError, setCors };
