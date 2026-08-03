import { createServer } from "node:http";
import { timingSafeEqual } from "node:crypto";
import { createLocalFieldJobRouter } from "./local-field-job-http.mjs";

const WORKER_HTTP_VERSION = "sigma-authenticated-simulator-worker-http/2";
const UPLOAD_CONTENT = /^\/api\/v1\/data-uploads\/upload_[0-9a-f]{24}\/content$/;
const UPLOAD_ROUTE = /^\/api\/v1\/data-uploads(?:\/upload_[0-9a-f]{24}(?:\/content)?)?$/;
const FIELD_JOB_ROUTE = /^\/api\/v1\/field-jobs(?:\/job_[0-9a-f]{24}(?:\/(?:events|cancel|artifacts)(?:\/[^/\\]+)?)?)?$/;
const GALAXY_JOB_ROUTE = /^\/api\/v1\/galaxy-jobs(?:\/job_[0-9a-f]{24}(?:\/(?:events|cancel|artifacts)(?:\/[^/\\]+)?)?)?$/;

class WorkerHttpError extends Error {
  constructor(statusCode, code, message) {
    super(message);
    this.statusCode = statusCode;
    this.code = code;
  }
}

function json(response, statusCode, value) {
  const content = Buffer.from(`${JSON.stringify(value)}\n`, "utf8");
  response.statusCode = statusCode;
  response.setHeader("Content-Type", "application/json; charset=utf-8");
  response.setHeader("Content-Length", String(content.length));
  response.end(content);
}

function adaptResponse(response) {
  response.status = (statusCode) => {
    response.statusCode = statusCode;
    return response;
  };
  response.json = (value) => {
    const content = Buffer.from(`${JSON.stringify(value)}\n`, "utf8");
    response.setHeader("Content-Type", "application/json; charset=utf-8");
    response.setHeader("Content-Length", String(content.length));
    response.end(content);
    return response;
  };
  return response;
}

function validToken(value) {
  return typeof value === "string" && Buffer.byteLength(value, "utf8") >= 32;
}

function authorized(header, token) {
  if (typeof header !== "string" || !header.startsWith("Bearer ")) return false;
  const supplied = Buffer.from(header.slice(7), "utf8");
  const expected = Buffer.from(token, "utf8");
  return supplied.length === expected.length && timingSafeEqual(supplied, expected);
}

function allowedPath(pathname) {
  return UPLOAD_ROUTE.test(pathname) || FIELD_JOB_ROUTE.test(pathname) || GALAXY_JOB_ROUTE.test(pathname);
}

async function requestBody(request, pathname, service, maxJsonBytes) {
  const binary = request.method === "PUT" && UPLOAD_CONTENT.test(pathname);
  const limit = binary ? service.maxUploadBytes : maxJsonBytes;
  const chunks = [];
  let bytes = 0;
  for await (const chunk of request) {
    bytes += chunk.length;
    if (bytes > limit) {
      throw new WorkerHttpError(413, "request_too_large", `request exceeds the ${limit} byte worker limit`);
    }
    chunks.push(chunk);
  }
  if (!chunks.length) return undefined;
  const content = Buffer.concat(chunks);
  if (binary) return content;
  try {
    return JSON.parse(content.toString("utf8"));
  } catch {
    throw new WorkerHttpError(400, "invalid_json", "request body must be valid JSON");
  }
}

export function createWorkerHttpServer({
  service,
  token,
  maxJsonBytes = 1_000_000,
  persistentStoreConfigured = false,
} = {}) {
  if (!service) throw new Error("worker HTTP server requires a field-job service");
  if (!validToken(token)) throw new Error("SIMULATOR_WORKER_TOKEN must contain at least 32 UTF-8 bytes");
  if (!Number.isSafeInteger(maxJsonBytes) || maxJsonBytes < 1024) {
    throw new Error("maxJsonBytes must be an integer of at least 1024 bytes");
  }
  const router = createLocalFieldJobRouter(service);
  const server = createServer(async (request, rawResponse) => {
    const response = adaptResponse(rawResponse);
    response.setHeader("Cache-Control", "no-store");
    response.setHeader("X-Content-Type-Options", "nosniff");
    response.setHeader("Referrer-Policy", "no-referrer");
    response.setHeader("Permissions-Policy", "camera=(), microphone=(), geolocation=()");
    try {
      const url = new URL(request.url, `http://${request.headers.host ?? "worker.invalid"}`);
      if (url.pathname === "/healthz") {
        if (request.method !== "GET") {
          json(response, 405, { error: "method_not_allowed", allowed: ["GET"] });
          return;
        }
        json(response, 200, {
          schemaVersion: WORKER_HTTP_VERSION,
          status: "ok",
          execution: "safe_confirmed_fields_and_gravity_independent_galaxy_generation",
          jobClasses: {
            field: "available",
            galaxyExtractionAndGeneration: "available",
            observationEvaluation: "not_exposed",
            inverseResponse: "not_exposed",
            arbitraryPlugin: "not_exposed",
          },
          arbitraryCodeExecution: false,
          authentication: "bearer_token_required_for_scientific_routes",
          store: {
            backend: "filesystem",
            storePathConfigured: Boolean(persistentStoreConfigured),
            durability: persistentStoreConfigured
              ? "requires_deployment_volume_verification"
              : "ephemeral_development_default",
            productionDurabilityClaim: false,
          },
          quotas: {
            maxUploadBytes: service.maxUploadBytes,
            maxStoredJobs: service.maxStoredJobs,
            maxEstimatedMemoryBytes: service.maxEstimatedMemoryBytes,
            maxArtifactBytes: service.maxArtifactBytes,
            maxArtifactCount: service.maxArtifactCount,
          },
        });
        return;
      }
      if (!allowedPath(url.pathname)) {
        json(response, 404, { error: "not_found" });
        return;
      }
      if (!authorized(request.headers.authorization, token)) {
        response.setHeader("WWW-Authenticate", 'Bearer realm="sigma-simulator-worker"');
        json(response, 401, { error: "unauthorized", message: "valid worker authorization is required" });
        return;
      }
      request.query = Object.fromEntries(url.searchParams.entries());
      request.body = await requestBody(request, url.pathname, service, maxJsonBytes);
      if (!(await router(request, response, url))) {
        json(response, 404, { error: "not_found" });
      }
    } catch (error) {
      if (response.writableEnded) return;
      json(response, error.statusCode ?? 500, {
        error: error.code ?? "worker_http_error",
        message: error.message,
      });
    }
  });
  server.requestTimeout = 30_000;
  server.headersTimeout = 10_000;
  server.keepAliveTimeout = 5_000;
  return server;
}

export { WORKER_HTTP_VERSION };
