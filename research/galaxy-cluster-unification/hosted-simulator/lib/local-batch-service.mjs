import { createHash, randomUUID } from "node:crypto";
import { mkdir, readFile, readdir, rename, stat, writeFile } from "node:fs/promises";
import { dirname, resolve } from "node:path";
import { sha256 } from "./canonical.mjs";
import { prepareBatch } from "./batch-preflight.mjs";
import { LocalServiceError } from "./local-field-job-service.mjs";

const SERVICE_VERSION = "sigma-local-batch-service/1";
const IDENTIFIER = /^batch_[0-9a-f]{24}$/;
const TERMINAL_CHILD_STATES = new Set([
  "succeeded",
  "failed",
  "failed_nonconvergence",
  "rejected_input",
  "infrastructure_failed",
  "cancelled",
]);
const SCIENTIFIC_CHILD_STATES = new Set(["succeeded", "failed", "failed_nonconvergence"]);
const TERMINAL_BATCH_STATES = new Set(["succeeded", "completed_with_failures", "cancelled"]);

function now() {
  return new Date().toISOString();
}

function digest(value) {
  return createHash("sha256").update(value).digest("hex");
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

async function readJson(path) {
  return JSON.parse(await readFile(path, "utf8"));
}

async function atomicWrite(path, value) {
  await mkdir(dirname(path), { recursive: true });
  const temporary = `${path}.tmp-${process.pid}-${randomUUID()}`;
  await writeFile(temporary, `${JSON.stringify(value, null, 2)}\n`, "utf8");
  for (let attempt = 0; ; attempt += 1) {
    try {
      await rename(temporary, path);
      break;
    } catch (error) {
      if (attempt >= 4 || !["EACCES", "EPERM"].includes(error.code)) throw error;
      await new Promise((resolvePromise) => setTimeout(resolvePromise, 10 * (attempt + 1)));
    }
  }
}

function csvCell(value) {
  const text = value === null || value === undefined ? "" : String(value);
  return /[",\n\r]/.test(text) ? `"${text.replaceAll('"', '""')}"` : text;
}

function csv(columns, rows) {
  return `${[columns, ...rows.map((row) => columns.map((column) => row[column]))]
    .map((values) => values.map(csvCell).join(","))
    .join("\n")}\n`;
}

function maximumNumeric(value) {
  if (typeof value === "number" && Number.isFinite(value)) return Math.abs(value);
  if (Array.isArray(value)) return Math.max(0, ...value.map(maximumNumeric));
  if (value && typeof value === "object") return Math.max(0, ...Object.values(value).map(maximumNumeric));
  return 0;
}

function median(values) {
  const sorted = values.filter(Number.isFinite).sort((left, right) => left - right);
  if (!sorted.length) return null;
  const middle = Math.floor(sorted.length / 2);
  return sorted.length % 2 ? sorted[middle] : 0.5 * (sorted[middle - 1] + sorted[middle]);
}

function htmlEscape(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

function publicBatch(record) {
  return {
    ...record,
    links: {
      self: `/api/v1/batches/${record.id}`,
      events: `/api/v1/batches/${record.id}/events`,
      artifacts: `/api/v1/batches/${record.id}/artifacts`,
      cancel: `/api/v1/batches/${record.id}/cancel`,
    },
  };
}

export class LocalBatchService {
  constructor({ root, fieldService, maxBatches = 100, pollMilliseconds = 25 }) {
    if (!root || !fieldService) throw new Error("local batch service requires root and fieldService");
    this.root = resolve(root);
    this.fieldService = fieldService;
    this.maxBatches = maxBatches;
    this.pollMilliseconds = pollMilliseconds;
    this.monitors = new Map();
    this.shuttingDown = false;
  }

  get capabilities() {
    return {
      serviceVersion: SERVICE_VERSION,
      maximumSystemsPerContract: 1000,
      maximumStoredBatches: this.maxBatches,
      executableParameterPolicies: ["published_fixed", "universal_fixed"],
      deterministicArtifacts: [
        "manifest.json",
        "model.json",
        "per_galaxy.csv",
        "aggregate_scores.json",
        "failures.csv",
        "report.html",
        "llm_briefing.md",
        "reproduction_command.txt",
      ],
    };
  }

  async initialize() {
    this.shuttingDown = false;
    await mkdir(this.#batchesRoot(), { recursive: true });
    for (const entry of await readdir(this.#batchesRoot(), { withFileTypes: true })) {
      if (!entry.isDirectory() || !IDENTIFIER.test(entry.name)) continue;
      const path = this.#recordPath(entry.name);
      if (!(await exists(path))) continue;
      const record = await readJson(path);
      if (!TERMINAL_BATCH_STATES.has(record.state)) this.#monitor(record.id);
    }
  }

  async close() {
    this.shuttingDown = true;
    await Promise.allSettled([...this.monitors.values()]);
  }

  async createBatch(payload) {
    if (!payload || typeof payload !== "object" || Array.isArray(payload)) {
      throw new LocalServiceError(422, "invalid_batch", "batch request must be an object");
    }
    const resolvedSystems = [];
    const uploadIds = [];
    for (const source of payload.systems ?? []) {
      let upload;
      let sourceIdentity;
      if (source.dataUploadId) {
        upload = await this.fieldService.getUpload(source.dataUploadId);
        sourceIdentity = { kind: "data_upload", id: upload.id };
      } else if (source.galaxyJobId) {
        upload = await this.fieldService.createUploadFromGalaxyArtifact(
          source.galaxyJobId,
          source.galaxyArtifact,
        );
        sourceIdentity = {
          kind: "galaxy_job_artifact",
          galaxyJobId: source.galaxyJobId,
          artifact: source.galaxyArtifact,
          materializedUploadId: upload.id,
        };
      } else {
        throw new LocalServiceError(422, "invalid_batch_source", `system ${source.id ?? "?"} requires one data source`);
      }
      if (upload.state !== "ready") throw new LocalServiceError(409, "upload_not_ready", `system ${source.id} upload is not ready`);
      uploadIds.push(upload.id);
      resolvedSystems.push({
        id: source.id,
        source: sourceIdentity,
        inputBundle: upload.inputBundle,
      });
    }
    let preflight;
    try {
      preflight = prepareBatch({ submission: payload, resolvedSystems });
    } catch (cause) {
      const error = new LocalServiceError(422, "invalid_batch", cause.message);
      error.cause = cause;
      throw error;
    }
    if (!preflight.executionReadiness.executable) {
      const error = new LocalServiceError(422, "parameter_policy_not_executable", preflight.executionReadiness.blockers.join("; "));
      error.details = preflight.executionReadiness.blockers;
      throw error;
    }
    const identity = {
      schemaVersion: "sigma-batch-submission-identity/1",
      serviceVersion: SERVICE_VERSION,
      preflightSha256: preflight.preflightSha256,
      fieldWorkerSourceSha256: this.fieldService.workerSourceSha256,
    };
    const id = `batch_${sha256(identity).slice(0, 24)}`;
    if (await exists(this.#recordPath(id))) {
      return { ...publicBatch(await readJson(this.#recordPath(id))), duplicate: true };
    }
    const count = (await readdir(this.#batchesRoot(), { withFileTypes: true })).filter((entry) => entry.isDirectory()).length;
    if (count >= this.maxBatches) throw new LocalServiceError(429, "batch_quota_exceeded", "local stored-batch quota has been reached");
    await mkdir(this.#batchDirectory(id), { recursive: false });
    await atomicWrite(this.#submissionPath(id), payload);
    const childJobs = [];
    for (let index = 0; index < resolvedSystems.length; index += 1) {
      const submission = await this.fieldService.createFieldJob({
        schemaVersion: "sigma-field-job-submit/1",
        model: payload.model,
        dataUploadId: uploadIds[index],
        request: payload.fieldRequest,
      });
      childJobs.push({
        systemId: resolvedSystems[index].id,
        source: resolvedSystems[index].source,
        inputBundleSha256: resolvedSystems[index].inputBundle.bundleSha256,
        fieldJobId: submission.id,
      });
    }
    const record = {
      schemaVersion: "sigma-batch-record/1",
      id,
      identity,
      state: "running",
      preflight,
      parameterPolicy: preflight.parameterPolicy,
      systemCount: resolvedSystems.length,
      childJobs,
      completedChildren: 0,
      successfulChildren: 0,
      createdAt: now(),
      updatedAt: now(),
      scientificResultSha256: null,
      manifestSha256: null,
    };
    await atomicWrite(this.#recordPath(id), record);
    await this.#appendEvent(id, "running", {
      message: `One frozen model queued across ${record.systemCount} systems.`,
    });
    this.#monitor(id);
    return { ...publicBatch(record), duplicate: false };
  }

  async listBatches() {
    const records = [];
    for (const entry of await readdir(this.#batchesRoot(), { withFileTypes: true })) {
      if (!entry.isDirectory() || !IDENTIFIER.test(entry.name)) continue;
      const path = this.#recordPath(entry.name);
      if (await exists(path)) records.push(publicBatch(await readJson(path)));
    }
    records.sort((left, right) => right.createdAt.localeCompare(left.createdAt));
    return { schemaVersion: "sigma-batch-list/1", items: records };
  }

  async getBatch(idValue) {
    return publicBatch(await this.#readBatch(this.#id(idValue)));
  }

  async getEvents(idValue) {
    const id = this.#id(idValue);
    await this.#readBatch(id);
    if (!(await exists(this.#eventsPath(id)))) return { schemaVersion: "sigma-batch-events/1", batchId: id, items: [] };
    const lines = (await readFile(this.#eventsPath(id), "utf8")).split("\n").filter(Boolean);
    return { schemaVersion: "sigma-batch-events/1", batchId: id, items: lines.map(JSON.parse) };
  }

  async cancelBatch(idValue) {
    const id = this.#id(idValue);
    const record = await this.#readBatch(id);
    if (TERMINAL_BATCH_STATES.has(record.state)) return publicBatch(record);
    await Promise.allSettled(record.childJobs.map((child) => this.fieldService.cancelFieldJob(child.fieldJobId)));
    record.state = "cancelled";
    record.updatedAt = now();
    await atomicWrite(this.#recordPath(id), record);
    await this.#appendEvent(id, "cancelled", { message: "Batch and nonterminal child jobs were cancelled." });
    return publicBatch(record);
  }

  async getArtifacts(idValue) {
    const id = this.#id(idValue);
    const record = await this.#readBatch(id);
    if (!new Set(["succeeded", "completed_with_failures"]).has(record.state)) {
      throw new LocalServiceError(409, "artifacts_not_ready", `batch is ${record.state}`);
    }
    const root = this.#artifactsDirectory(id);
    const manifest = await readJson(resolve(root, "manifest.json"));
    const indexBytes = await readFile(resolve(root, "artifact_index.json"));
    if (digest(indexBytes) !== manifest.artifactIndexSha256) {
      throw new LocalServiceError(409, "artifact_integrity_failed", "batch artifact index failed integrity verification");
    }
    const artifactIndex = JSON.parse(indexBytes.toString("utf8"));
    return {
      schemaVersion: "sigma-batch-artifact-response/1",
      batchId: id,
      manifest,
      artifactIndex,
      items: artifactIndex.artifacts.map((item) => ({
        ...item,
        url: `/api/v1/batches/${id}/artifacts/${encodeURIComponent(item.path)}`,
      })),
    };
  }

  async getArtifact(idValue, nameValue) {
    const id = this.#id(idValue);
    const name = decodeURIComponent(nameValue);
    if (!name || name.includes("/") || name.includes("\\") || name === "." || name === "..") {
      throw new LocalServiceError(404, "artifact_not_found", "unknown batch artifact");
    }
    const response = await this.getArtifacts(id);
    const record = response.artifactIndex.artifacts.find((item) => item.path === name);
    if (!record) throw new LocalServiceError(404, "artifact_not_found", "unknown batch artifact");
    const content = await readFile(resolve(this.#artifactsDirectory(id), name));
    if (content.length !== record.bytes || digest(content) !== record.sha256) {
      throw new LocalServiceError(409, "artifact_integrity_failed", `batch artifact ${name} failed integrity verification`);
    }
    return { content, record };
  }

  async waitForIdle(timeoutMilliseconds = 60_000) {
    const deadline = Date.now() + timeoutMilliseconds;
    while (this.monitors.size && Date.now() < deadline) {
      await new Promise((resolvePromise) => setTimeout(resolvePromise, 20));
    }
    if (this.monitors.size) throw new Error("local batch service did not become idle");
  }

  #batchesRoot() { return resolve(this.root, "batches"); }
  #batchDirectory(id) { return resolve(this.#batchesRoot(), id); }
  #recordPath(id) { return resolve(this.#batchDirectory(id), "record.json"); }
  #submissionPath(id) { return resolve(this.#batchDirectory(id), "submission.json"); }
  #eventsPath(id) { return resolve(this.#batchDirectory(id), "events.jsonl"); }
  #artifactsDirectory(id) { return resolve(this.#batchDirectory(id), "artifacts"); }

  #id(value) {
    if (typeof value !== "string" || !IDENTIFIER.test(value)) throw new LocalServiceError(404, "not_found", "unknown batch identifier");
    return value;
  }

  async #readBatch(id) {
    const path = this.#recordPath(id);
    if (!(await exists(path))) throw new LocalServiceError(404, "not_found", "unknown batch identifier");
    return readJson(path);
  }

  async #appendEvent(id, state, details = {}) {
    const path = this.#eventsPath(id);
    const current = (await exists(path)) ? await readFile(path, "utf8") : "";
    const sequence = current.split("\n").filter(Boolean).length + 1;
    await writeFile(path, `${current}${JSON.stringify({ sequence, at: now(), state, ...details })}\n`, "utf8");
  }

  #monitor(id) {
    if (this.monitors.has(id)) return this.monitors.get(id);
    const monitor = this.#monitorUntilTerminal(id).finally(() => this.monitors.delete(id));
    this.monitors.set(id, monitor);
    return monitor;
  }

  async #monitorUntilTerminal(id) {
    while (!this.shuttingDown) {
      const record = await this.#readBatch(id);
      if (TERMINAL_BATCH_STATES.has(record.state)) return;
      const children = await Promise.all(
        record.childJobs.map((child) => this.fieldService.getFieldJob(child.fieldJobId)),
      );
      const completed = children.filter((child) => TERMINAL_CHILD_STATES.has(child.state)).length;
      const succeeded = children.filter((child) => child.state === "succeeded").length;
      if (completed !== record.completedChildren || succeeded !== record.successfulChildren) {
        record.completedChildren = completed;
        record.successfulChildren = succeeded;
        record.updatedAt = now();
        await atomicWrite(this.#recordPath(id), record);
      }
      if (completed === children.length) {
        await this.#finalize(id, record, children);
        return;
      }
      await new Promise((resolvePromise) => setTimeout(resolvePromise, this.pollMilliseconds));
    }
  }

  async #finalize(id, record, children) {
    const submission = await readJson(this.#submissionPath(id));
    const rows = [];
    const failureRows = [];
    const childManifest = [];
    for (let index = 0; index < children.length; index += 1) {
      const child = children[index];
      const definition = record.childJobs[index];
      let scientific = null;
      if (SCIENTIFIC_CHILD_STATES.has(child.state)) {
        try {
          scientific = JSON.parse((await this.fieldService.getArtifact(child.id, "scientific_result.json")).content.toString("utf8"));
        } catch {
          scientific = null;
        }
      }
      const equationResidual = maximumNumeric(scientific?.equationResiduals);
      const row = {
        system_id: definition.systemId,
        source_kind: definition.source.kind,
        source_id: definition.source.id ?? definition.source.galaxyJobId,
        field_job_id: child.id,
        scientific_job_id: child.scientificJobId,
        state: child.state,
        converged: scientific?.converged ?? false,
        iterations: scientific?.iterations ?? null,
        maximum_relative_update: scientific?.maximumRelativeUpdate ?? null,
        maximum_equation_residual: equationResidual,
        universal_gravity_parameters: child.parameterAccounting?.universal ?? 0,
        per_object_gravity_parameters: child.parameterAccounting?.perObject ?? 0,
      };
      rows.push(row);
      childManifest.push({
        systemId: definition.systemId,
        source: definition.source,
        inputBundleSha256: definition.inputBundleSha256,
        fieldJobId: child.id,
        scientificJobId: child.scientificJobId,
        state: child.state,
        scientificResultSha256: child.scientificResultSha256,
      });
      if (child.state !== "succeeded") {
        failureRows.push({
          system_id: definition.systemId,
          field_job_id: child.id,
          state: child.state,
          category: child.state === "failed_nonconvergence" ? "scientific_nonconvergence" : "execution_failure",
          message: child.inputFailure?.message ?? child.infrastructureFailure?.message ?? "child field job did not succeed",
        });
      }
    }
    const successfulRows = rows.filter((row) => row.state === "succeeded");
    const aggregate = {
      schemaVersion: "sigma-batch-aggregate/1",
      batchId: id,
      systemCount: rows.length,
      succeededSystems: successfulRows.length,
      failedSystems: rows.length - successfulRows.length,
      convergenceFraction: successfulRows.length / rows.length,
      medianIterations: median(successfulRows.map((row) => row.iterations)),
      medianMaximumRelativeUpdate: median(successfulRows.map((row) => row.maximum_relative_update)),
      maximumEquationResidual: Math.max(0, ...rows.map((row) => row.maximum_equation_residual)),
      modelSha256: record.preflight.modelSha256,
      parameterPolicy: record.parameterPolicy,
      universalGravityParameters: Math.max(0, ...rows.map((row) => row.universal_gravity_parameters)),
      perObjectGravityParameters: Math.max(0, ...rows.map((row) => row.per_object_gravity_parameters)),
      observationScoresAvailable: false,
      claimBoundary: record.preflight.claimBoundary,
    };
    const scientificCore = {
      schemaVersion: "sigma-batch-scientific-result/1",
      batchId: id,
      modelSha256: record.preflight.modelSha256,
      parameterPolicy: record.parameterPolicy,
      children: childManifest,
      aggregate,
    };
    const scientificResultSha256 = sha256(scientificCore);
    const artifacts = this.#artifactsDirectory(id);
    await mkdir(artifacts, { recursive: false });
    await atomicWrite(resolve(artifacts, "batch.json"), submission);
    await atomicWrite(resolve(artifacts, "model.json"), submission.model);
    await atomicWrite(resolve(artifacts, "child_jobs.json"), {
      schemaVersion: "sigma-batch-child-jobs/1",
      items: childManifest,
    });
    await atomicWrite(resolve(artifacts, "aggregate_scores.json"), aggregate);
    await writeFile(
      resolve(artifacts, "per_galaxy.csv"),
      csv(
        [
          "system_id",
          "source_kind",
          "source_id",
          "field_job_id",
          "scientific_job_id",
          "state",
          "converged",
          "iterations",
          "maximum_relative_update",
          "maximum_equation_residual",
          "universal_gravity_parameters",
          "per_object_gravity_parameters",
        ],
        rows,
      ),
      "utf8",
    );
    await writeFile(
      resolve(artifacts, "failures.csv"),
      csv(["system_id", "field_job_id", "state", "category", "message"], failureRows),
      "utf8",
    );
    const tableRows = rows.map((row) => `<tr><td>${htmlEscape(row.system_id)}</td><td>${htmlEscape(row.state)}</td><td>${htmlEscape(row.iterations ?? "")}</td><td>${htmlEscape(row.maximum_equation_residual)}</td></tr>`).join("");
    await writeFile(
      resolve(artifacts, "report.html"),
      `<!doctype html><html lang="en"><meta charset="utf-8"><title>Batch ${htmlEscape(id)}</title><style>body{font:16px system-ui;max-width:960px;margin:3rem auto;padding:0 1rem;color:#182026}table{border-collapse:collapse;width:100%}th,td{border:1px solid #ccd4da;padding:.5rem;text-align:left}code{background:#eef1f3;padding:.1rem .3rem}</style><h1>Formula-independent field batch</h1><p><code>${htmlEscape(id)}</code></p><p>One confirmed model was run over ${rows.length} systems with <strong>${htmlEscape(record.parameterPolicy.mode)}</strong> parameters. ${successfulRows.length} child solves succeeded.</p><p><strong>Scientific boundary:</strong> this report measures numerical execution and convergence only. It does not yet compare predicted velocities or lensing with observations.</p><table><thead><tr><th>System</th><th>State</th><th>Iterations</th><th>Maximum equation residual</th></tr></thead><tbody>${tableRows}</tbody></table></html>`,
      "utf8",
    );
    await writeFile(
      resolve(artifacts, "llm_briefing.md"),
      `# Batch briefing\n\n- Batch: \`${id}\`\n- Model SHA-256: \`${record.preflight.modelSha256}\`\n- Parameter policy: \`${record.parameterPolicy.mode}\`\n- Systems: ${rows.length}\n- Successful numerical solves: ${successfulRows.length}\n- Per-object gravity parameters: ${aggregate.perObjectGravityParameters}\n- Observation scores available: no\n\nThis deterministic briefing summarizes numerical field execution. It must not be described as a rotation-curve or lensing validation.\n`,
      "utf8",
    );
    await writeFile(
      resolve(artifacts, "reproduction_command.txt"),
      "curl -X POST -H \"content-type: application/json\" --data @batch.json <base-url>/api/v1/batches\n",
      "utf8",
    );
    const artifactNames = [
      "aggregate_scores.json",
      "batch.json",
      "child_jobs.json",
      "failures.csv",
      "llm_briefing.md",
      "model.json",
      "per_galaxy.csv",
      "report.html",
      "reproduction_command.txt",
    ];
    const artifactIndex = {
      schemaVersion: "sigma-batch-artifact-index/1",
      batchId: id,
      artifacts: await Promise.all(artifactNames.map(async (name) => {
        const content = await readFile(resolve(artifacts, name));
        return { path: name, bytes: content.length, sha256: digest(content) };
      })),
    };
    await atomicWrite(resolve(artifacts, "artifact_index.json"), artifactIndex);
    const manifestCore = {
      schemaVersion: "sigma-batch-manifest/1",
      batchId: id,
      state: failureRows.length ? "completed_with_failures" : "succeeded",
      scientificResultSha256,
      modelSha256: record.preflight.modelSha256,
      parameterPolicy: record.parameterPolicy,
      systemCount: rows.length,
      artifactIndexSha256: digest(await readFile(resolve(artifacts, "artifact_index.json"))),
      fieldWorkerSourceSha256: this.fieldService.workerSourceSha256,
      reportScope: "numerical_execution_not_observation_validation",
    };
    const manifest = { ...manifestCore, manifestSha256: sha256(manifestCore) };
    await atomicWrite(resolve(artifacts, "manifest.json"), manifest);
    record.state = manifest.state;
    record.completedChildren = rows.length;
    record.successfulChildren = successfulRows.length;
    record.scientificResultSha256 = scientificResultSha256;
    record.manifestSha256 = manifest.manifestSha256;
    record.updatedAt = now();
    record.finishedAt = record.updatedAt;
    await atomicWrite(this.#recordPath(id), record);
    await this.#appendEvent(id, record.state, {
      message: `Batch report published with ${successfulRows.length}/${rows.length} successful child solves.`,
      scientificResultSha256,
    });
  }
}
