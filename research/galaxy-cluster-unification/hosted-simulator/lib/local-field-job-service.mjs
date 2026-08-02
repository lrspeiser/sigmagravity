import { spawn } from "node:child_process";
import { createHash, randomUUID } from "node:crypto";
import {
  copyFile,
  link,
  mkdir,
  readFile,
  readdir,
  rename,
  stat,
  writeFile,
} from "node:fs/promises";
import { delimiter, dirname, resolve } from "node:path";
import { sha256 } from "./canonical.mjs";
import { prepareFieldJob, validateArrayBundle } from "./field-job-preflight.mjs";

const SERVICE_VERSION = "sigma-local-field-job-service/1";
const IDENTIFIER = /^(?:upload|job)_[0-9a-f]{24}$/;
const SHA256 = /^[0-9a-f]{64}$/;
const TERMINAL_STATES = new Set([
  "succeeded",
  "failed",
  "failed_nonconvergence",
  "rejected_input",
  "infrastructure_failed",
  "cancelled",
]);
const SCIENTIFIC_TERMINAL_STATES = new Set(["succeeded", "failed", "failed_nonconvergence"]);

export class LocalServiceError extends Error {
  constructor(statusCode, code, message) {
    super(message);
    this.name = "LocalServiceError";
    this.statusCode = statusCode;
    this.code = code;
  }
}

function now() {
  return new Date().toISOString();
}

function sha256Bytes(value) {
  return createHash("sha256").update(value).digest("hex");
}

async function readJson(path) {
  return JSON.parse(await readFile(path, "utf8"));
}

async function atomicWrite(path, value) {
  await mkdir(dirname(path), { recursive: true });
  const temporary = `${path}.tmp-${process.pid}-${randomUUID()}`;
  await writeFile(temporary, `${JSON.stringify(value, null, 2)}\n`, "utf8");
  for (let attempt = 0; ; attempt++) {
    try {
      await rename(temporary, path);
      break;
    } catch (error) {
      if (attempt >= 4 || !["EACCES", "EPERM"].includes(error.code)) throw error;
      await new Promise((resolvePromise) => setTimeout(resolvePromise, 10 * (attempt + 1)));
    }
  }
}

async function exists(path) {
  try {
    await stat(path);
    return true;
  } catch (error) {
    if (error.code === "ENOENT") return false;
    throw error;
  }
}

async function workerSourceSha256(projectRoot) {
  const digest = createHash("sha256");
  for (const name of ["field_job.py", "generic_field_worker.py"]) {
    const path = resolve(projectRoot, "src", "voidscreen", name);
    digest.update(name, "utf8");
    digest.update(Buffer.from([0]));
    digest.update(await readFile(path));
    digest.update(Buffer.from([0]));
  }
  return digest.digest("hex");
}

function assertIdentifier(value, prefix) {
  if (typeof value !== "string" || !IDENTIFIER.test(value) || !value.startsWith(`${prefix}_`)) {
    throw new LocalServiceError(404, "not_found", `unknown ${prefix} identifier`);
  }
  return value;
}

function publicUpload(record) {
  return {
    ...record,
    links: {
      self: `/api/v1/data-uploads/${record.id}`,
      content: `/api/v1/data-uploads/${record.id}/content`,
    },
  };
}

function publicJob(record) {
  return {
    ...record,
    links: {
      self: `/api/v1/field-jobs/${record.id}`,
      events: `/api/v1/field-jobs/${record.id}/events`,
      artifacts: `/api/v1/field-jobs/${record.id}/artifacts`,
      cancel: `/api/v1/field-jobs/${record.id}/cancel`,
    },
  };
}

function applyScientificManifest(record, manifest) {
  record.state = manifest.state;
  record.updatedAt = now();
  record.finishedAt = record.updatedAt;
  record.scientificJobId = manifest.jobId;
  record.scientificResultSha256 = manifest.scientificResultSha256 ?? null;
  record.failureSha256 = manifest.failureSha256 ?? null;
  record.manifestSha256 = manifest.manifestSha256;
  return record;
}

function parseWorkerInputFailure(stderr) {
  for (const line of String(stderr ?? "").trim().split("\n").reverse()) {
    try {
      const value = JSON.parse(line);
      if (value?.schemaVersion === "sigma-field-job-cli-error/1") return value;
    } catch {
      // Continue past ordinary diagnostic lines.
    }
  }
  return { state: "rejected_input", errorType: "InputValidationError", message: "worker rejected the submitted input" };
}

function defaultRunner({ projectRoot, pythonExecutable, requestPath, timeoutMs, signal }) {
  return new Promise((resolvePromise, rejectPromise) => {
    const script = resolve(projectRoot, "scripts", "run_generic_field_job.py");
    const source = resolve(projectRoot, "src");
    const pythonPath = process.env.PYTHONPATH
      ? `${source}${delimiter}${process.env.PYTHONPATH}`
      : source;
    const child = spawn(
      pythonExecutable,
      [script, "run", "--request", requestPath],
      {
        cwd: projectRoot,
        env: { ...process.env, PYTHONPATH: pythonPath, PYTHONUNBUFFERED: "1" },
        windowsHide: true,
      },
    );
    const stdout = [];
    const stderr = [];
    let outputBytes = 0;
    let settled = false;
    let timedOut = false;
    const capture = (target) => (chunk) => {
      outputBytes += chunk.length;
      if (outputBytes <= 1_000_000) target.push(chunk);
    };
    child.stdout.on("data", capture(stdout));
    child.stderr.on("data", capture(stderr));
    const terminate = () => {
      if (!child.killed) child.kill();
    };
    const onAbort = () => terminate();
    signal?.addEventListener("abort", onAbort, { once: true });
    const timer = setTimeout(() => {
      timedOut = true;
      terminate();
    }, timeoutMs);
    child.on("error", (error) => {
      if (settled) return;
      settled = true;
      clearTimeout(timer);
      signal?.removeEventListener("abort", onAbort);
      rejectPromise(error);
    });
    child.on("close", (exitCode, exitSignal) => {
      if (settled) return;
      settled = true;
      clearTimeout(timer);
      signal?.removeEventListener("abort", onAbort);
      resolvePromise({
        exitCode,
        exitSignal,
        timedOut,
        stdout: Buffer.concat(stdout).toString("utf8"),
        stderr: Buffer.concat(stderr).toString("utf8"),
      });
    });
  });
}

export class LocalFieldJobService {
  constructor({
    root,
    projectRoot,
    pythonExecutable = process.env.SIMULATOR_PYTHON ?? "python",
    maxUploadBytes = Number(process.env.SIMULATOR_MAX_UPLOAD_BYTES ?? 128 * 1024 * 1024),
    maxStoredJobs = Number(process.env.SIMULATOR_MAX_STORED_JOBS ?? 100),
    maxEstimatedMemoryBytes = Number(process.env.SIMULATOR_MAX_ESTIMATED_MEMORY_BYTES ?? 4 * 1024 ** 3),
    timeoutMs = Number(process.env.SIMULATOR_JOB_TIMEOUT_MS ?? 10 * 60 * 1000),
    runner = defaultRunner,
  }) {
    if (!root || !projectRoot) throw new Error("local service requires root and projectRoot");
    this.root = resolve(root);
    this.projectRoot = resolve(projectRoot);
    this.pythonExecutable = pythonExecutable;
    this.maxUploadBytes = maxUploadBytes;
    this.maxStoredJobs = maxStoredJobs;
    this.maxEstimatedMemoryBytes = maxEstimatedMemoryBytes;
    this.timeoutMs = timeoutMs;
    this.runner = runner;
    this.queue = [];
    this.running = null;
    this.abortController = null;
    this.drainPromise = null;
    this.jobLocks = new Map();
    this.workerSourceSha256 = null;
    this.shuttingDown = false;
  }

  get capabilities() {
    return {
      serviceVersion: SERVICE_VERSION,
      executionMode: "local_single_user_safe_manifest_only",
      arbitraryCodeExecution: false,
      maxUploadBytes: this.maxUploadBytes,
      maxStoredJobs: this.maxStoredJobs,
      maxEstimatedMemoryBytes: this.maxEstimatedMemoryBytes,
      concurrency: 1,
      workerSourceSha256: this.workerSourceSha256,
    };
  }

  async initialize() {
    this.shuttingDown = false;
    this.workerSourceSha256 = await workerSourceSha256(this.projectRoot);
    await mkdir(this.#uploadsRoot(), { recursive: true });
    await mkdir(this.#jobsRoot(), { recursive: true });
    for (const entry of await readdir(this.#jobsRoot(), { withFileTypes: true })) {
      if (!entry.isDirectory() || !IDENTIFIER.test(entry.name) || !entry.name.startsWith("job_")) continue;
      const recordPath = this.#jobRecordPath(entry.name);
      if (!(await exists(recordPath))) continue;
      const record = await readJson(recordPath);
      if (record.state === "queued" || record.state === "running") {
        const manifestPath = resolve(this.#jobDirectory(record.id), "artifacts", "manifest.json");
        if (await exists(manifestPath)) {
          applyScientificManifest(record, await readJson(manifestPath));
          record.recoveredAfterRestart = true;
          await atomicWrite(recordPath, record);
          await this.#appendEvent(record.id, record.state, {
            message: "Recovered a completed scientific manifest after gateway restart.",
            recoveredAfterRestart: true,
          });
          continue;
        }
        record.state = "queued";
        record.updatedAt = now();
        record.recoveredAfterRestart = true;
        await atomicWrite(recordPath, record);
        this.queue.push(record.id);
      }
    }
    this.#kick();
  }

  async close() {
    this.shuttingDown = true;
    this.abortController?.abort();
    await this.drainPromise?.catch(() => undefined);
  }

  async createUpload(payload) {
    if (!payload || typeof payload !== "object" || Array.isArray(payload)) {
      throw new LocalServiceError(422, "invalid_upload", "upload request must be an object");
    }
    if (payload.schemaVersion !== "sigma-data-upload-request/1") {
      throw new LocalServiceError(422, "invalid_upload", "upload request must use sigma-data-upload-request/1");
    }
    const inputBundle = payload.inputBundle;
    validateArrayBundle(inputBundle);
    if (!inputBundle.provenance || typeof inputBundle.provenance !== "object") {
      throw new LocalServiceError(422, "missing_provenance", "input bundle must declare provenance");
    }
    if (!inputBundle.license || typeof inputBundle.license.id !== "string") {
      throw new LocalServiceError(422, "missing_license", "input bundle must declare a license");
    }
    const archive = payload.archive;
    if (!archive || !SHA256.test(archive.sha256 ?? "")) {
      throw new LocalServiceError(422, "invalid_archive", "archive.sha256 must be a lowercase SHA-256");
    }
    if (!Number.isSafeInteger(archive.bytes) || archive.bytes <= 0) {
      throw new LocalServiceError(422, "invalid_archive", "archive.bytes must be a positive integer");
    }
    if (archive.bytes > this.maxUploadBytes) {
      throw new LocalServiceError(413, "upload_too_large", `archive exceeds ${this.maxUploadBytes} byte limit`);
    }
    const identity = {
      schemaVersion: "sigma-data-upload-identity/1",
      inputBundleSha256: inputBundle.bundleSha256,
      archiveSha256: archive.sha256,
      archiveBytes: archive.bytes,
    };
    const id = `upload_${sha256(identity).slice(0, 24)}`;
    const recordPath = this.#uploadRecordPath(id);
    if (await exists(recordPath)) return publicUpload(await readJson(recordPath));
    const record = {
      schemaVersion: "sigma-data-upload/1",
      id,
      state: "awaiting_content",
      identity,
      inputBundle,
      archive: {
        sha256: archive.sha256,
        bytes: archive.bytes,
        mediaType: "application/x-npz",
      },
      createdAt: now(),
      updatedAt: now(),
    };
    await mkdir(this.#uploadDirectory(id), { recursive: false });
    await atomicWrite(recordPath, record);
    return publicUpload(record);
  }

  async putUploadContent(idValue, content) {
    const id = assertIdentifier(idValue, "upload");
    if (!Buffer.isBuffer(content)) throw new LocalServiceError(400, "invalid_content", "upload content must be bytes");
    const record = await this.#readUpload(id);
    if (content.length !== record.archive.bytes) {
      throw new LocalServiceError(422, "archive_size_mismatch", "archive byte count does not match its ticket");
    }
    const actualSha256 = sha256Bytes(content);
    if (actualSha256 !== record.archive.sha256) {
      throw new LocalServiceError(422, "archive_hash_mismatch", "archive SHA-256 does not match its ticket");
    }
    const archivePath = this.#uploadArchivePath(id);
    if (record.state === "ready") {
      if (!(await exists(archivePath)) || sha256Bytes(await readFile(archivePath)) !== actualSha256) {
        throw new LocalServiceError(409, "immutable_upload_changed", "ready upload content failed integrity verification");
      }
      return publicUpload(record);
    }
    if (record.state !== "awaiting_content") {
      throw new LocalServiceError(409, "upload_not_writable", `upload is ${record.state}`);
    }
    const temporary = `${archivePath}.tmp-${process.pid}-${randomUUID()}`;
    await writeFile(temporary, content);
    await rename(temporary, archivePath);
    record.state = "ready";
    record.updatedAt = now();
    record.integrity = { archiveSha256Verified: true, archiveBytesVerified: true };
    await atomicWrite(this.#uploadRecordPath(id), record);
    return publicUpload(record);
  }

  async getUpload(idValue) {
    return publicUpload(await this.#readUpload(assertIdentifier(idValue, "upload")));
  }

  async createFieldJob(payload) {
    if (!payload || typeof payload !== "object" || Array.isArray(payload)) {
      throw new LocalServiceError(422, "invalid_job", "field job request must be an object");
    }
    if (payload.schemaVersion !== "sigma-field-job-submit/1") {
      throw new LocalServiceError(422, "invalid_job", "field job request must use sigma-field-job-submit/1");
    }
    const upload = await this.#readUpload(assertIdentifier(payload.dataUploadId, "upload"));
    if (upload.state !== "ready") {
      throw new LocalServiceError(409, "upload_not_ready", "data upload content is not ready");
    }
    const preflight = prepareFieldJob({
      model: payload.model,
      inputBundle: upload.inputBundle,
      request: payload.request,
    });
    if (!preflight.valid) {
      const error = new LocalServiceError(422, "invalid_model", "model failed preflight validation");
      error.details = preflight.errors;
      throw error;
    }
    if (preflight.resourceEstimate.estimatedMemoryBytes > this.maxEstimatedMemoryBytes) {
      throw new LocalServiceError(413, "resource_quota_exceeded", "estimated worker memory exceeds the local quota");
    }
    const identity = {
      schemaVersion: "sigma-field-job-submission-identity/1",
      serviceVersion: SERVICE_VERSION,
      preflightSha256: preflight.preflightSha256,
      archiveSha256: upload.archive.sha256,
      workerSourceSha256: this.workerSourceSha256,
    };
    const id = `job_${sha256(identity).slice(0, 24)}`;
    const recordPath = this.#jobRecordPath(id);
    if (await exists(recordPath)) return { ...publicJob(await readJson(recordPath)), duplicate: true };
    const jobCount = (await readdir(this.#jobsRoot(), { withFileTypes: true })).filter((entry) => entry.isDirectory()).length;
    if (jobCount >= this.maxStoredJobs) {
      throw new LocalServiceError(429, "job_quota_exceeded", "local stored-job quota has been reached");
    }
    const jobDirectory = this.#jobDirectory(id);
    const bundleDirectory = resolve(jobDirectory, "bundle");
    await mkdir(bundleDirectory, { recursive: true });
    await atomicWrite(resolve(jobDirectory, "model.json"), payload.model);
    await atomicWrite(resolve(bundleDirectory, "bundle.json"), upload.inputBundle);
    try {
      await link(this.#uploadArchivePath(upload.id), resolve(bundleDirectory, "arrays.npz"));
    } catch (error) {
      if (!["EXDEV", "EPERM", "EACCES"].includes(error.code)) throw error;
      await copyFile(this.#uploadArchivePath(upload.id), resolve(bundleDirectory, "arrays.npz"));
    }
    const envelope = {
      schemaVersion: "sigma-field-job-cli/1",
      modelPath: "model.json",
      inputBundlePath: "bundle",
      outputDirectory: "artifacts",
      request: payload.request,
    };
    await atomicWrite(resolve(jobDirectory, "request.json"), envelope);
    const record = {
      schemaVersion: "sigma-field-job-record/1",
      id,
      identity,
      state: "queued",
      dataUploadId: upload.id,
      preflight,
      workerSourceSha256: this.workerSourceSha256,
      parameterAccounting: preflight.parameterAccounting,
      createdAt: now(),
      updatedAt: now(),
      scientificJobId: null,
      scientificResultSha256: null,
      failureSha256: null,
    };
    await atomicWrite(recordPath, record);
    await this.#appendEvent(id, "queued", { message: "Job accepted by the local single-worker queue." });
    this.queue.push(id);
    this.#kick();
    return { ...publicJob(record), duplicate: false };
  }

  async listFieldJobs() {
    const records = [];
    for (const entry of await readdir(this.#jobsRoot(), { withFileTypes: true })) {
      if (!entry.isDirectory() || !entry.name.startsWith("job_")) continue;
      const path = this.#jobRecordPath(entry.name);
      if (await exists(path)) records.push(publicJob(await readJson(path)));
    }
    records.sort((left, right) => right.createdAt.localeCompare(left.createdAt));
    return { schemaVersion: "sigma-field-job-list/1", items: records };
  }

  async getFieldJob(idValue) {
    return publicJob(await this.#readJob(assertIdentifier(idValue, "job")));
  }

  async getEvents(idValue) {
    const id = assertIdentifier(idValue, "job");
    await this.#readJob(id);
    const path = this.#eventsPath(id);
    if (!(await exists(path))) return { schemaVersion: "sigma-field-job-events/1", jobId: id, items: [] };
    const lines = (await readFile(path, "utf8")).split("\n").filter(Boolean);
    return { schemaVersion: "sigma-field-job-events/1", jobId: id, items: lines.map((line) => JSON.parse(line)) };
  }

  async getArtifacts(idValue) {
    const id = assertIdentifier(idValue, "job");
    const record = await this.#readJob(id);
    if (!SCIENTIFIC_TERMINAL_STATES.has(record.state)) {
      throw new LocalServiceError(409, "artifacts_not_ready", `job is ${record.state}`);
    }
    const root = resolve(this.#jobDirectory(id), "artifacts");
    const manifest = await readJson(resolve(root, "manifest.json"));
    const indexBytes = await readFile(resolve(root, "artifact_index.json"));
    if (sha256Bytes(indexBytes) !== manifest.artifactIndexSha256) {
      throw new LocalServiceError(409, "artifact_integrity_failed", "artifact index no longer matches the scientific manifest");
    }
    const index = JSON.parse(indexBytes.toString("utf8"));
    return {
      schemaVersion: "sigma-field-job-artifact-response/1",
      jobId: id,
      manifest,
      artifactIndex: index,
      items: index.artifacts.map((item) => ({
        ...item,
        url: `/api/v1/field-jobs/${id}/artifacts/${encodeURIComponent(item.path)}`,
      })),
    };
  }

  async getArtifact(idValue, nameValue) {
    const id = assertIdentifier(idValue, "job");
    const name = decodeURIComponent(nameValue);
    if (!name || name.includes("/") || name.includes("\\") || name === "." || name === "..") {
      throw new LocalServiceError(404, "artifact_not_found", "unknown artifact");
    }
    const response = await this.getArtifacts(id);
    if (!response.artifactIndex.artifacts.some((item) => item.path === name)) {
      throw new LocalServiceError(404, "artifact_not_found", "unknown artifact");
    }
    const record = response.artifactIndex.artifacts.find((item) => item.path === name);
    const content = await readFile(resolve(this.#jobDirectory(id), "artifacts", name));
    if (content.length !== record.bytes || sha256Bytes(content) !== record.sha256) {
      throw new LocalServiceError(409, "artifact_integrity_failed", `artifact ${name} failed its recorded hash`);
    }
    return { content, record };
  }

  async cancelFieldJob(idValue) {
    const id = assertIdentifier(idValue, "job");
    return this.#withJobLock(id, async () => {
      const record = await this.#readJob(id);
      if (TERMINAL_STATES.has(record.state)) return publicJob(record);
      if (record.state === "queued") this.queue = this.queue.filter((value) => value !== id);
      if (record.state === "running" && this.running === id) this.abortController?.abort();
      record.state = "cancelled";
      record.updatedAt = now();
      await atomicWrite(this.#jobRecordPath(id), record);
      await this.#appendEvent(id, "cancelled", { message: "Cancellation requested." });
      return publicJob(record);
    });
  }

  async waitForIdle(timeoutMs = 30_000) {
    const deadline = Date.now() + timeoutMs;
    while ((this.running || this.queue.length || this.drainPromise) && Date.now() < deadline) {
      await new Promise((resolvePromise) => setTimeout(resolvePromise, 20));
    }
    if (this.running || this.queue.length || this.drainPromise) throw new Error("local job service did not become idle");
  }

  #uploadsRoot() { return resolve(this.root, "uploads"); }
  #jobsRoot() { return resolve(this.root, "jobs"); }
  #uploadDirectory(id) { return resolve(this.#uploadsRoot(), id); }
  #uploadRecordPath(id) { return resolve(this.#uploadDirectory(id), "upload.json"); }
  #uploadArchivePath(id) { return resolve(this.#uploadDirectory(id), "arrays.npz"); }
  #jobDirectory(id) { return resolve(this.#jobsRoot(), id); }
  #jobRecordPath(id) { return resolve(this.#jobDirectory(id), "record.json"); }
  #eventsPath(id) { return resolve(this.#jobDirectory(id), "events.jsonl"); }

  async #readUpload(id) {
    const path = this.#uploadRecordPath(id);
    if (!(await exists(path))) throw new LocalServiceError(404, "not_found", "unknown upload identifier");
    return readJson(path);
  }

  async #readJob(id) {
    const path = this.#jobRecordPath(id);
    if (!(await exists(path))) throw new LocalServiceError(404, "not_found", "unknown job identifier");
    return readJson(path);
  }

  async #appendEvent(id, state, details = {}) {
    const path = this.#eventsPath(id);
    const existing = (await exists(path)) ? (await readFile(path, "utf8")).split("\n").filter(Boolean).length : 0;
    const event = { sequence: existing + 1, at: now(), state, ...details };
    const current = (await exists(path)) ? await readFile(path, "utf8") : "";
    await writeFile(path, `${current}${JSON.stringify(event)}\n`, "utf8");
  }

  #kick() {
    if (this.shuttingDown || this.drainPromise || this.running || !this.queue.length) return;
    this.drainPromise = this.#drain().finally(() => {
      this.drainPromise = null;
      if (this.queue.length) this.#kick();
    });
  }

  async #drain() {
    while (this.queue.length && !this.shuttingDown) {
      const id = this.queue.shift();
      const record = await this.#readJob(id);
      if (record.state !== "queued") continue;
      await this.#run(id, record);
    }
  }

  async #run(id, record) {
    this.running = id;
    this.abortController = new AbortController();
    await this.#withJobLock(id, async () => {
      record.state = "running";
      record.startedAt = now();
      record.updatedAt = record.startedAt;
      await atomicWrite(this.#jobRecordPath(id), record);
      await this.#appendEvent(id, "running", { message: "Safe manifest worker started." });
    });
    const jobDirectory = this.#jobDirectory(id);
    try {
      const execution = await this.runner({
        projectRoot: this.projectRoot,
        pythonExecutable: this.pythonExecutable,
        requestPath: resolve(jobDirectory, "request.json"),
        jobDirectory,
        timeoutMs: this.timeoutMs,
        signal: this.abortController.signal,
      });
      await writeFile(resolve(jobDirectory, "worker_stdout.log"), execution.stdout ?? "", "utf8");
      await writeFile(resolve(jobDirectory, "worker_stderr.log"), execution.stderr ?? "", "utf8");
      if (execution.timedOut) throw new LocalServiceError(504, "worker_timeout", "worker exceeded its runtime limit");
      if (execution.exitCode === 2) {
        await this.#withJobLock(id, async () => {
          const latest = await this.#readJob(id);
          if (latest.state === "cancelled") return;
          latest.state = "rejected_input";
          latest.updatedAt = now();
          latest.finishedAt = latest.updatedAt;
          latest.inputFailure = parseWorkerInputFailure(execution.stderr);
          await atomicWrite(this.#jobRecordPath(id), latest);
          await this.#appendEvent(id, "rejected_input", latest.inputFailure);
        });
        return;
      }
      if (execution.exitCode !== 0) {
        throw new LocalServiceError(500, "worker_process_failed", `worker exited with code ${execution.exitCode}`);
      }
      const manifest = await readJson(resolve(jobDirectory, "artifacts", "manifest.json"));
      await this.#withJobLock(id, async () => {
        const latest = await this.#readJob(id);
        if (latest.state === "cancelled") return;
        applyScientificManifest(latest, manifest);
        await atomicWrite(this.#jobRecordPath(id), latest);
        await this.#appendEvent(id, latest.state, {
          message: latest.state === "succeeded" ? "Scientific worker completed." : "Scientific worker retained a terminal diagnostic result.",
          scientificJobId: latest.scientificJobId,
        });
      });
    } catch (error) {
      await this.#withJobLock(id, async () => {
        const latest = await this.#readJob(id);
        if (latest.state !== "cancelled" && this.shuttingDown) {
          latest.state = "queued";
          latest.updatedAt = now();
          latest.interruptedByGatewayShutdown = true;
          await atomicWrite(this.#jobRecordPath(id), latest);
          await this.#appendEvent(id, "queued", {
            message: "Worker interrupted by gateway shutdown; job will resume after restart.",
            interruptedByGatewayShutdown: true,
          });
        } else if (latest.state !== "cancelled") {
          latest.state = "infrastructure_failed";
          latest.updatedAt = now();
          latest.finishedAt = latest.updatedAt;
          latest.infrastructureFailure = { code: error.code ?? "worker_infrastructure_error", message: error.message };
          await atomicWrite(this.#jobRecordPath(id), latest);
          await this.#appendEvent(id, "infrastructure_failed", latest.infrastructureFailure);
        }
      });
    } finally {
      this.running = null;
      this.abortController = null;
    }
  }

  async #withJobLock(id, operation) {
    const previous = this.jobLocks.get(id) ?? Promise.resolve();
    const current = previous.catch(() => undefined).then(operation);
    this.jobLocks.set(id, current);
    try {
      return await current;
    } finally {
      if (this.jobLocks.get(id) === current) this.jobLocks.delete(id);
    }
  }
}
