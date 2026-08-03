import { randomUUID } from "node:crypto";
import { send } from "./http.mjs";

const DEFAULT_REQUEST_BYTES = 4 * 1024 * 1024;
const DEFAULT_RESPONSE_BYTES = 8 * 1024 * 1024;
const DEFAULT_TIMEOUT_MS = 15_000;

function positiveInteger(value, fallback, label) {
  const result = value === undefined ? fallback : Number(value);
  if (!Number.isSafeInteger(result) || result <= 0) throw new Error(`${label} must be a positive integer`);
  return result;
}

export function resolveWorkerConnector(environment = process.env) {
  const rawUrl = environment.SIMULATOR_WORKER_URL;
  const token = environment.SIMULATOR_WORKER_TOKEN;
  if (!rawUrl && !token) return null;
  if (!rawUrl || !token) throw new Error("SIMULATOR_WORKER_URL and SIMULATOR_WORKER_TOKEN must be configured together");
  if (Buffer.byteLength(token, "utf8") < 32) throw new Error("SIMULATOR_WORKER_TOKEN must contain at least 32 UTF-8 bytes");
  const url = new URL(rawUrl);
  const local = ["127.0.0.1", "localhost", "::1"].includes(url.hostname);
  if (url.protocol !== "https:" && !(local && url.protocol === "http:")) {
    throw new Error("SIMULATOR_WORKER_URL must use HTTPS outside localhost");
  }
  if (url.username || url.password || url.search || url.hash || !["", "/"].includes(url.pathname)) {
    throw new Error("SIMULATOR_WORKER_URL must be an origin without credentials, query, fragment, or path");
  }
  return {
    origin: url.origin,
    token,
    maxRequestBytes: positiveInteger(environment.SIMULATOR_GATEWAY_MAX_REQUEST_BYTES, DEFAULT_REQUEST_BYTES, "SIMULATOR_GATEWAY_MAX_REQUEST_BYTES"),
    maxResponseBytes: positiveInteger(environment.SIMULATOR_GATEWAY_MAX_RESPONSE_BYTES, DEFAULT_RESPONSE_BYTES, "SIMULATOR_GATEWAY_MAX_RESPONSE_BYTES"),
    timeoutMs: positiveInteger(environment.SIMULATOR_WORKER_TIMEOUT_MS, DEFAULT_TIMEOUT_MS, "SIMULATOR_WORKER_TIMEOUT_MS"),
  };
}

export function workerConnectorState(environment = process.env) {
  try {
    return resolveWorkerConnector(environment) ? "configured" : "not_configured";
  } catch {
    return "misconfigured";
  }
}

async function requestContent(request, maximumBytes) {
  let value = request.body;
  if ((value === undefined || value === null) && typeof request[Symbol.asyncIterator] === "function") {
    const chunks = [];
    let bytes = 0;
    for await (const chunk of request) {
      bytes += chunk.length;
      if (bytes > maximumBytes) throw new RangeError("gateway request body exceeds its configured limit");
      chunks.push(chunk);
    }
    value = chunks.length ? Buffer.concat(chunks) : undefined;
  }
  if (value === undefined || value === null) return { body: undefined, contentType: undefined };
  if (Buffer.isBuffer(value)) {
    return { body: value, contentType: request.headers?.["content-type"] ?? "application/octet-stream" };
  }
  if (value instanceof Uint8Array) {
    return { body: Buffer.from(value), contentType: request.headers?.["content-type"] ?? "application/octet-stream" };
  }
  if (typeof value === "string") {
    return { body: Buffer.from(value, "utf8"), contentType: request.headers?.["content-type"] ?? "application/json" };
  }
  return { body: Buffer.from(JSON.stringify(value), "utf8"), contentType: "application/json; charset=utf-8" };
}

function safeWorkerPath(path) {
  const upload = /^api\/v1\/data-uploads(?:\/upload_[0-9a-f]{24}(?:\/content)?)?$/;
  const fieldJob = /^api\/v1\/field-jobs(?:\/job_[0-9a-f]{24}(?:\/(?:events|cancel|artifacts)(?:\/[A-Za-z0-9_.%-]+)?)?)?$/;
  let decodedSegments = [];
  try {
    decodedSegments = typeof path === "string" ? path.split("/").map((value) => decodeURIComponent(value)) : [];
  } catch {
    decodedSegments = [".."];
  }
  if (
    typeof path !== "string"
    || (!upload.test(path) && !fieldJob.test(path))
    || decodedSegments.some((value) => value === "." || value === ".." || value.includes("/") || value.includes("\\"))
  ) {
    throw new Error("worker proxy path is not allow-listed");
  }
  return path;
}

async function readBoundedResponse(upstream, maximumBytes) {
  const declared = Number(upstream.headers.get("content-length"));
  if (Number.isFinite(declared) && declared > maximumBytes) {
    throw new Error(`worker response exceeds the ${maximumBytes} byte gateway limit`);
  }
  const content = Buffer.from(await upstream.arrayBuffer());
  if (content.length > maximumBytes) {
    throw new Error(`worker response exceeds the ${maximumBytes} byte gateway limit`);
  }
  return content;
}

export async function proxyWorkerRequest({
  request,
  response,
  path,
  environment = process.env,
  fetchImpl = fetch,
  unavailable = {
    error: "production_worker_not_connected",
    message: "The authenticated scientific worker connector is not configured for this deployment.",
  },
} = {}) {
  let connector;
  try {
    connector = resolveWorkerConnector(environment);
  } catch (error) {
    send(response, 503, {
      error: "production_worker_misconfigured",
      message: error.message,
    });
    return;
  }
  if (!connector) {
    send(response, 503, unavailable);
    return;
  }
  let workerPath;
  try {
    workerPath = safeWorkerPath(path);
  } catch (error) {
    send(response, 400, { error: "invalid_worker_path", message: error.message });
    return;
  }
  let body;
  let contentType;
  try {
    ({ body, contentType } = await requestContent(request, connector.maxRequestBytes));
  } catch (error) {
    send(response, error instanceof RangeError ? 413 : 400, {
      error: error instanceof RangeError ? "gateway_request_too_large" : "invalid_gateway_request",
      message: error instanceof RangeError
        ? `request exceeds the ${connector.maxRequestBytes} byte gateway limit`
        : "request body could not be serialized for the scientific worker",
    });
    return;
  }
  if (body && body.length > connector.maxRequestBytes) {
    send(response, 413, {
      error: "gateway_request_too_large",
      message: `request exceeds the ${connector.maxRequestBytes} byte gateway limit`,
    });
    return;
  }
  const requestId = randomUUID();
  try {
    const headers = {
      Authorization: `Bearer ${connector.token}`,
      Accept: request.headers?.accept ?? "application/json, application/octet-stream",
      "X-Gateway-Request-Id": requestId,
    };
    if (contentType) headers["Content-Type"] = contentType;
    const upstream = await fetchImpl(`${connector.origin}/${workerPath}`, {
      method: request.method,
      headers,
      body,
      redirect: "error",
      signal: AbortSignal.timeout(connector.timeoutMs),
    });
    const content = await readBoundedResponse(upstream, connector.maxResponseBytes);
    response.statusCode = upstream.status;
    response.setHeader("Cache-Control", "no-store");
    response.setHeader("Content-Type", upstream.headers.get("content-type") ?? "application/octet-stream");
    response.setHeader("Content-Length", String(content.length));
    response.setHeader("X-Gateway-Request-Id", requestId);
    const contentSha256 = upstream.headers.get("x-content-sha256");
    if (contentSha256) response.setHeader("X-Content-SHA256", contentSha256);
    response.end(content);
  } catch (error) {
    send(response, 502, {
      error: "production_worker_unreachable",
      message: "The configured scientific worker did not return a bounded response.",
      requestId,
      reason: error.name === "TimeoutError" ? "timeout" : "connection_or_response_failure",
    });
  }
}
