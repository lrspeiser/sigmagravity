const DEFAULT_REQUEST_BYTES = 128 * 1024;
const DEFAULT_RESPONSE_BYTES = 2 * 1024 * 1024;
const DEFAULT_TIMEOUT_MS = 240_000;

function positiveInteger(value, fallback, label) {
  const result = value === undefined ? fallback : Number(value);
  if (!Number.isSafeInteger(result) || result < 1) throw new Error(`${label} must be a positive integer`);
  return result;
}

export function resolveStatelessWorkerConfiguration(environment = process.env) {
  const rawUrl = environment.SIGMA_STATELESS_WORKER_URL;
  const token = environment.SIGMA_STATELESS_WORKER_TOKEN;
  if (!rawUrl && !token) return null;
  if (!rawUrl || !token) {
    throw new Error("SIGMA_STATELESS_WORKER_URL and SIGMA_STATELESS_WORKER_TOKEN must be configured together");
  }
  if (Buffer.byteLength(token, "utf8") < 32) {
    throw new Error("SIGMA_STATELESS_WORKER_TOKEN must contain at least 32 UTF-8 bytes");
  }
  const url = new URL(rawUrl);
  const local = ["127.0.0.1", "localhost", "::1"].includes(url.hostname);
  if (url.protocol !== "https:" && !(local && url.protocol === "http:")) {
    throw new Error("SIGMA_STATELESS_WORKER_URL must use HTTPS outside localhost");
  }
  if (url.username || url.password || url.search || url.hash || !["", "/"].includes(url.pathname)) {
    throw new Error("SIGMA_STATELESS_WORKER_URL must be an origin without credentials, query, fragment, or path");
  }
  return {
    origin: url.origin,
    token,
    maximumRequestBytes: positiveInteger(
      environment.SIGMA_STATELESS_WORKER_MAX_REQUEST_BYTES,
      DEFAULT_REQUEST_BYTES,
      "SIGMA_STATELESS_WORKER_MAX_REQUEST_BYTES",
    ),
    maximumResponseBytes: positiveInteger(
      environment.SIGMA_STATELESS_WORKER_MAX_RESPONSE_BYTES,
      DEFAULT_RESPONSE_BYTES,
      "SIGMA_STATELESS_WORKER_MAX_RESPONSE_BYTES",
    ),
    timeoutMs: positiveInteger(
      environment.SIGMA_STATELESS_WORKER_TIMEOUT_MS,
      DEFAULT_TIMEOUT_MS,
      "SIGMA_STATELESS_WORKER_TIMEOUT_MS",
    ),
  };
}

export function statelessWorkerState(environment = process.env) {
  try {
    return resolveStatelessWorkerConfiguration(environment) ? "configured" : "not_configured";
  } catch {
    return "misconfigured";
  }
}

async function boundedResponse(response, maximumBytes) {
  const declared = Number(response.headers.get("content-length"));
  if (Number.isFinite(declared) && declared > maximumBytes) {
    throw Object.assign(new Error("stateless worker response exceeds its declared quota"), { retryable: false });
  }
  if (!response.body) return Buffer.alloc(0);
  const reader = response.body.getReader();
  const chunks = [];
  let total = 0;
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    total += value.byteLength;
    if (total > maximumBytes) {
      await reader.cancel("stateless worker response exceeded quota");
      throw Object.assign(new Error("stateless worker response exceeds its byte quota"), { retryable: false });
    }
    chunks.push(Buffer.from(value));
  }
  return Buffer.concat(chunks, total);
}

export class StatelessWorkerClient {
  constructor({ environment = process.env, fetchImpl = fetch } = {}) {
    const configuration = resolveStatelessWorkerConfiguration(environment);
    if (!configuration) throw new Error("stateless scientific worker is not configured");
    this.configuration = configuration;
    this.fetchImpl = fetchImpl;
  }

  async execute({ job, leaseToken, requestObjectReference }) {
    const payload = {
      schemaVersion: "sigma-stateless-worker-execution/1",
      job: {
        id: job.id,
        projectId: job.projectId,
        jobType: job.jobType,
        requestSha256: job.requestSha256,
        attempt: job.attempt,
      },
      leaseToken,
      requestObjectReference,
    };
    const body = Buffer.from(JSON.stringify(payload), "utf8");
    if (body.length > this.configuration.maximumRequestBytes) {
      throw Object.assign(new Error("stateless worker request exceeds its quota"), { retryable: false });
    }
    let response;
    try {
      response = await this.fetchImpl(`${this.configuration.origin}/v1/executions`, {
        method: "POST",
        redirect: "error",
        headers: {
          Authorization: `Bearer ${this.configuration.token}`,
          Accept: "application/json",
          "Content-Type": "application/json",
          "Content-Length": String(body.length),
          "X-Sigma-Job-Id": job.id,
          "X-Sigma-Lease-Token": leaseToken,
        },
        body,
        signal: AbortSignal.timeout(this.configuration.timeoutMs),
      });
    } catch (error) {
      throw Object.assign(new Error("stateless scientific worker is unreachable"), {
        code: error.name === "TimeoutError" ? "worker_timeout" : "worker_unreachable",
        retryable: true,
      });
    }
    const bytes = await boundedResponse(response, this.configuration.maximumResponseBytes);
    let result;
    try {
      result = JSON.parse(bytes.toString("utf8"));
    } catch {
      throw Object.assign(new Error("stateless worker returned invalid JSON"), { retryable: false });
    }
    if (!response.ok) {
      throw Object.assign(new Error("stateless worker rejected the execution"), {
        code: typeof result?.error === "string" ? result.error.slice(0, 80) : "worker_rejected",
        retryable: response.status >= 500 || response.status === 429,
      });
    }
    return result;
  }
}
