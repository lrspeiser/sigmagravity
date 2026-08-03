import { createHash, randomUUID } from "node:crypto";
import { mkdir, readFile, readdir, rename, stat, writeFile } from "node:fs/promises";
import { dirname, resolve } from "node:path";
import { sha256 } from "./canonical.mjs";
import { prepareBatch } from "./batch-preflight.mjs";
import { LocalServiceError } from "./local-field-job-service.mjs";

// Report contents participate in batch identity through this version. Bump it
// whenever deterministic batch artifacts or aggregation semantics change.
const SERVICE_VERSION = "sigma-local-batch-service/5";
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
      // Windows can briefly lock the destination while the background batch
      // poller and a cancellation request publish adjacent state changes.
      if (attempt >= 12 || !["EACCES", "EBUSY", "EPERM"].includes(error.code)) throw error;
      const delayMs = Math.min(100, 10 * (2 ** attempt));
      await new Promise((resolvePromise) => setTimeout(resolvePromise, delayMs));
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

function quantile(values, probability) {
  const sorted = values.filter(Number.isFinite).sort((left, right) => left - right);
  if (!sorted.length) return null;
  if (sorted.length === 1) return sorted[0];
  const position = (sorted.length - 1) * probability;
  const lower = Math.floor(position);
  const upper = Math.ceil(position);
  const fraction = position - lower;
  return sorted[lower] * (1 - fraction) + sorted[upper] * fraction;
}

function distributionSummary(values) {
  const finite = values.filter(Number.isFinite);
  return {
    count: finite.length,
    p16: quantile(finite, 0.16),
    p50: quantile(finite, 0.50),
    p84: quantile(finite, 0.84),
    minimum: finite.length ? Math.min(...finite) : null,
    maximum: finite.length ? Math.max(...finite) : null,
  };
}

function realizationSystemId(parentSystemId, realization) {
  const surface = String(realization.surfaceRealization).padStart(3, "0");
  const vertical = realization.verticalRealization === undefined
    ? ""
    : `::v${String(realization.verticalRealization).padStart(3, "0")}`;
  return `${parentSystemId}::s${surface}${vertical}`;
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
        "observation_predictions.csv",
        "observation_velocity_field_predictions.csv",
        "per_galaxy.csv",
        "per_realization.csv",
        "ensemble_summary.csv",
        "ensemble_summary.json",
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
    const observationUploadIds = [];
    for (const source of payload.systems ?? []) {
      const ensembleArtifact = new Set([
        "surface_density_ensemble",
        "volume_density_ensemble",
      ]).has(source.galaxyArtifact);
      if (ensembleArtifact) {
        if (!source.galaxyJobId || !source.ensembleSelection || source.dataUploadId) {
          throw new LocalServiceError(
            422,
            "invalid_batch_source",
            `system ${source.id ?? "?"} requires a galaxy ensemble job, artifact, and selection only`,
          );
        }
        const description = await this.fieldService.describeGalaxyEnsembleArtifact(
          source.galaxyJobId,
          source.galaxyArtifact,
          source.ensembleSelection,
        );
        if (resolvedSystems.length + description.realizations.length > 1000) {
          throw new LocalServiceError(
            413,
            "batch_child_quota_exceeded",
            "expanded ensemble selections exceed the 1000-child batch contract",
          );
        }
        const fixedObservationUpload = source.observationDataUploadId
          ? await this.fieldService.getUpload(source.observationDataUploadId)
          : null;
        if (fixedObservationUpload && fixedObservationUpload.state !== "ready") {
          throw new LocalServiceError(409, "upload_not_ready", `system ${source.id} observation upload is not ready`);
        }
        for (const realization of description.realizations) {
          const upload = await this.fieldService.createUploadFromGalaxyEnsembleArtifact(
            source.galaxyJobId,
            source.galaxyArtifact,
            realization,
          );
          const observationUpload = fixedObservationUpload ?? upload;
          uploadIds.push(upload.id);
          observationUploadIds.push(observationUpload.id);
          resolvedSystems.push({
            id: realizationSystemId(source.id, realization),
            parentSystemId: source.id,
            realization,
            source: {
              kind: "galaxy_job_ensemble_realization",
              galaxyJobId: source.galaxyJobId,
              artifact: source.galaxyArtifact,
              parentEnsembleBundleSha256: description.bundleSha256,
              uncertaintyStatus: description.uncertaintyStatus,
              realization,
              materializedUploadId: upload.id,
            },
            inputBundle: upload.inputBundle,
            observationBundle: observationUpload.inputBundle,
            observationTargets: source.observationTargets ?? [],
          });
        }
        continue;
      }
      if (source.ensembleSelection) {
        throw new LocalServiceError(422, "invalid_batch_source", `system ${source.id ?? "?"} uses ensembleSelection with a non-ensemble source`);
      }
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
      const observationUpload = source.observationDataUploadId
        ? await this.fieldService.getUpload(source.observationDataUploadId)
        : upload;
      if (observationUpload.state !== "ready") {
        throw new LocalServiceError(409, "upload_not_ready", `system ${source.id} observation upload is not ready`);
      }
      uploadIds.push(upload.id);
      observationUploadIds.push(observationUpload.id);
      resolvedSystems.push({
        id: source.id,
        parentSystemId: source.id,
        realization: null,
        source: sourceIdentity,
        inputBundle: upload.inputBundle,
        observationBundle: observationUpload.inputBundle,
        observationTargets: source.observationTargets ?? [],
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
      schemaVersion: "sigma-batch-submission-identity/3",
      serviceVersion: SERVICE_VERSION,
      preflightSha256: preflight.preflightSha256,
      fieldWorkerSourceSha256: this.fieldService.workerSourceSha256,
      observationWorkerSourceSha256: this.fieldService.observationWorkerSourceSha256,
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
        request: payload.fieldRequest ?? {},
      });
      childJobs.push({
        systemId: resolvedSystems[index].id,
        parentSystemId: resolvedSystems[index].parentSystemId,
        realization: resolvedSystems[index].realization,
        source: resolvedSystems[index].source,
        dataUploadId: uploadIds[index],
        observationDataUploadId: observationUploadIds[index],
        inputBundleSha256: resolvedSystems[index].inputBundle.bundleSha256,
        observationBundleSha256: resolvedSystems[index].observationBundle.bundleSha256,
        observationTargets: preflight.systems[index].observationTargets,
        fieldJobId: submission.id,
        observationEvaluationJobId: null,
      });
    }
    const record = {
      schemaVersion: "sigma-batch-record/2",
      id,
      identity,
      state: "running",
      phase: "field",
      preflight,
      parameterPolicy: preflight.parameterPolicy,
      systemCount: resolvedSystems.length,
      submittedSystemCount: (payload.systems ?? []).length,
      ensembleRealizationCount: resolvedSystems.filter((system) => system.realization).length,
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
      message: `One frozen model queued across ${record.systemCount} resolved system realization(s).`,
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
    record.state = "cancelled";
    record.phase = "cancelled";
    record.updatedAt = now();
    await atomicWrite(this.#recordPath(id), record);
    await this.#appendEvent(id, "cancelled", { message: "Batch cancellation requested for both child phases." });
    await Promise.allSettled([
      ...record.childJobs.map((child) => this.fieldService.cancelFieldJob(child.fieldJobId)),
      ...record.childJobs
        .filter((child) => child.observationEvaluationJobId)
        .map((child) => this.fieldService.cancelObservationEvaluationJob(child.observationEvaluationJobId)),
    ]);
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
      const fieldChildren = await Promise.all(
        record.childJobs.map((child) => this.fieldService.getFieldJob(child.fieldJobId)),
      );
      let recordChanged = false;
      for (let index = 0; index < fieldChildren.length; index += 1) {
        const fieldChild = fieldChildren[index];
        const definition = record.childJobs[index];
        if (
          fieldChild.state === "succeeded"
          && definition.observationTargets.length > 0
          && !definition.observationEvaluationJobId
          && !definition.observationCreationFailure
        ) {
          try {
            const submission = await this.fieldService.createObservationEvaluationJob({
              schemaVersion: "sigma-observation-evaluation-job-submit/1",
              fieldJobId: definition.fieldJobId,
              dataUploadId: definition.observationDataUploadId,
              observationTargets: definition.observationTargets,
            });
            definition.observationEvaluationJobId = submission.id;
            recordChanged = true;
            await this.#appendEvent(id, "observation_queued", {
              message: `Observation evaluation queued for ${definition.systemId}.`,
              systemId: definition.systemId,
              fieldJobId: definition.fieldJobId,
              observationEvaluationJobId: submission.id,
              duplicate: submission.duplicate,
            });
          } catch (cause) {
            definition.observationCreationFailure = {
              code: cause.code ?? "observation_creation_failed",
              message: cause.message,
            };
            recordChanged = true;
          }
        }
      }
      const composedChildren = await Promise.all(record.childJobs.map(async (definition, index) => ({
        field: fieldChildren[index],
        observation: definition.observationEvaluationJobId
          ? await this.fieldService.getObservationEvaluationJob(definition.observationEvaluationJobId)
          : null,
      })));
      const systemStates = composedChildren.map(({ field, observation }, index) => {
        const definition = record.childJobs[index];
        if (!TERMINAL_CHILD_STATES.has(field.state)) return "running";
        if (field.state !== "succeeded") return field.state;
        if (definition.observationTargets.length === 0) return "succeeded";
        if (definition.observationCreationFailure) return "rejected_input";
        return observation?.state ?? "running";
      });
      const completed = systemStates.filter((state) => TERMINAL_CHILD_STATES.has(state)).length;
      const succeeded = systemStates.filter((state) => state === "succeeded").length;
      const phase = fieldChildren.some((child) => !TERMINAL_CHILD_STATES.has(child.state))
        ? "field"
        : systemStates.some((state) => !TERMINAL_CHILD_STATES.has(state))
          ? "observation"
          : "reporting";
      if (TERMINAL_BATCH_STATES.has((await this.#readBatch(id)).state)) return;
      if (
        recordChanged
        || completed !== record.completedChildren
        || succeeded !== record.successfulChildren
        || phase !== record.phase
      ) {
        record.completedChildren = completed;
        record.successfulChildren = succeeded;
        record.phase = phase;
        record.updatedAt = now();
        await atomicWrite(this.#recordPath(id), record);
      }
      if (completed === composedChildren.length) {
        await this.#finalize(id, record, composedChildren);
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
    const observationPredictionRows = [];
    const velocityFieldPredictionRows = [];
    const multipleImagePredictionRows = [];
    const multipleImageFamilyRows = [];
    const observationKinds = new Set();
    for (let index = 0; index < children.length; index += 1) {
      const { field: fieldChild, observation: observationChild } = children[index];
      const definition = record.childJobs[index];
      let fieldScientific = null;
      if (SCIENTIFIC_CHILD_STATES.has(fieldChild.state)) {
        try {
          fieldScientific = JSON.parse((await this.fieldService.getArtifact(fieldChild.id, "scientific_result.json")).content.toString("utf8"));
        } catch {
          fieldScientific = null;
        }
      }
      let observationScientific = null;
      let observationArtifactIndex = null;
      if (observationChild && SCIENTIFIC_CHILD_STATES.has(observationChild.state)) {
        try {
          observationArtifactIndex = (await this.fieldService.getArtifacts(observationChild.id)).artifactIndex;
          observationScientific = JSON.parse((await this.fieldService.getArtifact(
            observationChild.id,
            "scientific_result.json",
          )).content.toString("utf8"));
        } catch {
          observationScientific = null;
        }
      }
      const equationResidual = maximumNumeric(fieldScientific?.equationResiduals);
      const observation = observationScientific?.observationEvaluation ?? null;
      const channelAggregates = observation?.channelAggregates
        ?? (observation?.sumSquaredResidualM2PerS2 !== null
          && observation?.sumSquaredResidualM2PerS2 !== undefined
          ? {
            velocity_m_s: {
              channel: "velocity_m_s",
              unit: "m/s",
              scoredTargetCount: observation.scoredTargetCount ?? 0,
              validPoints: observation.validScoredPoints ?? 0,
              fittedNuisanceParameters: 0,
              sumSquaredResidual: observation.sumSquaredResidualM2PerS2,
              inverseVarianceWeightedSquaredResidual: observation.inverseVarianceWeightedSquaredResidual ?? 0,
              inverseVarianceWeightSum: observation.inverseVarianceWeightSum ?? 0,
              chiSquare: observation.chiSquare ?? 0,
              degreesFreedom: observation.degreesFreedom ?? 0,
              gaussianLogLikelihood: 0,
            },
          }
          : {});
      if ((observation?.targetCount ?? 0) > 0 && observationChild?.state === "succeeded") {
        const targetKinds = new Set(
          observation.targetKinds
          ?? (definition.observationTargets ?? []).map((target) => target.kind),
        );
        for (const kind of targetKinds) observationKinds.add(kind);
        if (targetKinds.has("circular_speed_curve")) {
          const predictionArtifact = await this.fieldService.getArtifact(
            observationChild.id,
            "observation_predictions.csv",
          );
          const lines = predictionArtifact.content.toString("utf8").trimEnd().split("\n");
          const expectedHeader = "target_id,point_index,radius_m,predicted_speed_m_s,observed_speed_m_s,uncertainty_m_s,residual_m_s,azimuthal_coverage,mean_inward_acceleration_m_s2";
          if (lines[0] !== expectedHeader) {
            throw new Error(`observation job ${observationChild.id} returned an incompatible circular-speed prediction table`);
          }
          for (const line of lines.slice(1)) {
            if (line) observationPredictionRows.push(`${csvCell(definition.systemId)},${line}`);
          }
        }
        if (targetKinds.has("line_of_sight_velocity_field")) {
          const predictionArtifact = await this.fieldService.getArtifact(
            observationChild.id,
            "observation_velocity_field_predictions.csv",
          );
          const lines = predictionArtifact.content.toString("utf8").trimEnd().split("\n");
          const expectedHeader = "target_id,point_index,row_index,column_index,disk_major_coordinate_m,disk_minor_coordinate_m,circular_radius_m,predicted_circular_speed_m_s,predicted_velocity_m_s,observed_velocity_m_s,uncertainty_m_s,residual_m_s,declared_weight,inward_acceleration_m_s2";
          if (lines[0] !== expectedHeader) {
            throw new Error(`observation job ${observationChild.id} returned an incompatible velocity-field prediction table`);
          }
          for (const line of lines.slice(1)) {
            if (line) velocityFieldPredictionRows.push(`${csvCell(definition.systemId)},${line}`);
          }
        }
        if (targetKinds.has("multiple_image_systems")) {
          const predictionArtifact = await this.fieldService.getArtifact(
            observationChild.id,
            "observation_multiple_image_predictions.csv",
          );
          const predictionLines = predictionArtifact.content.toString("utf8").trimEnd().split("\n");
          const predictionHeader = "target_id,family_id,family_index,image_index,assignment_state,observed_east_arcsec,observed_north_arcsec,position_uncertainty_arcsec,predicted_root_index,predicted_east_arcsec,predicted_north_arcsec,residual_east_arcsec,residual_north_arcsec,separation_arcsec,root_closure_arcsec,root_absolute_magnification";
          if (predictionLines[0] !== predictionHeader) {
            throw new Error(`observation job ${observationChild.id} returned an incompatible multiple-image prediction table`);
          }
          for (const line of predictionLines.slice(1)) {
            if (line) multipleImagePredictionRows.push(`${csvCell(definition.systemId)},${line}`);
          }
          const familyArtifact = await this.fieldService.getArtifact(
            observationChild.id,
            "observation_multiple_image_families.csv",
          );
          const familyLines = familyArtifact.content.toString("utf8").trimEnd().split("\n");
          const familyHeader = "target_id,family_id,family_index,distance_ratio,profiled_source_east_arcsec,profiled_source_north_arcsec,observed_images,predicted_roots,matched_images,complete_observed_assignment,excess_predicted_roots,critical_curve_points,state,image_plane_rms_arcsec,matched_subset_diagnostic_rms_arcsec,chi_square,degrees_freedom,fitted_observation_nuisance_parameters,gravity_parameters_added";
          if (familyLines[0] !== familyHeader) {
            throw new Error(`observation job ${observationChild.id} returned an incompatible multiple-image family table`);
          }
          for (const line of familyLines.slice(1)) {
            if (line) multipleImageFamilyRows.push(`${csvCell(definition.systemId)},${line}`);
          }
        }
      }
      const systemState = fieldChild.state !== "succeeded"
        ? fieldChild.state
        : definition.observationTargets.length === 0
          ? "succeeded"
          : definition.observationCreationFailure
            ? "rejected_input"
            : observationChild?.state ?? "failed";
      const row = {
        system_id: definition.systemId,
        parent_system_id: definition.parentSystemId ?? definition.systemId,
        surface_realization: definition.realization?.surfaceRealization ?? null,
        vertical_realization: definition.realization?.verticalRealization ?? null,
        source_kind: definition.source.kind,
        source_id: definition.source.id ?? definition.source.galaxyJobId,
        field_job_id: fieldChild.id,
        observation_evaluation_job_id: observationChild?.id ?? null,
        field_scientific_job_id: fieldChild.scientificJobId,
        observation_scientific_job_id: observationChild?.scientificJobId ?? null,
        state: systemState,
        field_state: fieldChild.state,
        observation_state: definition.observationTargets.length
          ? observationChild?.state ?? (definition.observationCreationFailure ? "creation_failed" : "not_created")
          : "not_requested",
        converged: fieldScientific?.converged ?? false,
        iterations: fieldScientific?.iterations ?? null,
        maximum_relative_update: fieldScientific?.maximumRelativeUpdate ?? null,
        maximum_equation_residual: equationResidual,
        universal_gravity_parameters: fieldChild.parameterAccounting?.universal ?? 0,
        per_object_gravity_parameters: fieldChild.parameterAccounting?.perObject ?? 0,
        observation_added_gravity_parameters: observationChild?.evaluationAddedGravityParameters ?? 0,
        observation_targets: observation?.targetCount ?? 0,
        scored_observation_targets: observation?.scoredTargetCount ?? 0,
        valid_observation_points: observation?.validScoredPoints ?? 0,
        observation_rmse_m_s: observation?.rmseMPerS ?? null,
        observation_weighted_rmse_m_s: observation?.inverseVarianceWeightedRmseMPerS ?? null,
        observation_sum_squared_residual_m2_s2: observation?.sumSquaredResidualM2PerS2 ?? null,
        observation_weighted_squared_residual: observation?.inverseVarianceWeightedSquaredResidual ?? null,
        observation_weight_sum: observation?.inverseVarianceWeightSum ?? null,
        observation_chi_square: observation?.chiSquare ?? null,
        observation_degrees_freedom: observation?.degreesFreedom ?? null,
        observation_deflection_rmse_arcsec: channelAggregates.deflection_arcsec?.rmse ?? null,
        observation_deflection_weighted_rmse_arcsec: channelAggregates.deflection_arcsec?.inverseVarianceWeightedRmse ?? null,
        observation_reduced_shear_rmse: channelAggregates.reduced_shear_dimensionless?.rmse ?? null,
        observation_reduced_shear_weighted_rmse: channelAggregates.reduced_shear_dimensionless?.inverseVarianceWeightedRmse ?? null,
        observation_image_position_rmse_arcsec: channelAggregates.image_position_arcsec?.rmse ?? null,
        observation_image_position_weighted_rmse_arcsec: channelAggregates.image_position_arcsec?.inverseVarianceWeightedRmse ?? null,
        observation_incomplete_topology_targets: observation?.targets?.filter((target) => target.state === "incomplete_topology").length ?? 0,
        observation_channel_aggregates: channelAggregates,
      };
      rows.push(row);
      childManifest.push({
        systemId: definition.systemId,
        parentSystemId: definition.parentSystemId ?? definition.systemId,
        realization: definition.realization ?? null,
        source: definition.source,
        inputBundleSha256: definition.inputBundleSha256,
        observationBundleSha256: definition.observationBundleSha256,
        state: systemState,
        fieldJobId: fieldChild.id,
        fieldState: fieldChild.state,
        fieldScientificJobId: fieldChild.scientificJobId,
        fieldScientificResultSha256: fieldChild.scientificResultSha256,
        fieldManifestSha256: fieldChild.manifestSha256 ?? null,
        observationEvaluationJobId: observationChild?.id ?? null,
        observationState: row.observation_state,
        observationScientificJobId: observationChild?.scientificJobId ?? null,
        observationScientificResultSha256: observationChild?.scientificResultSha256 ?? null,
        observationManifestSha256: observationChild?.manifestSha256 ?? null,
        observationArtifacts: observationArtifactIndex?.artifacts ?? [],
        observationCreationFailure: definition.observationCreationFailure ?? null,
        observationTargets: definition.observationTargets,
      });
      if (systemState !== "succeeded") {
        const fieldFailed = fieldChild.state !== "succeeded";
        const category = fieldFailed
          ? fieldChild.state === "failed_nonconvergence"
            ? "scientific_nonconvergence"
            : "field_execution_failure"
          : definition.observationCreationFailure
            ? "observation_creation_failure"
            : "observation_execution_failure";
        failureRows.push({
          system_id: definition.systemId,
          parent_system_id: definition.parentSystemId ?? definition.systemId,
          field_job_id: fieldChild.id,
          observation_evaluation_job_id: observationChild?.id ?? null,
          state: systemState,
          category,
          message: fieldFailed
            ? fieldChild.inputFailure?.message ?? fieldChild.infrastructureFailure?.message ?? "child field job did not succeed"
            : definition.observationCreationFailure?.message
              ?? observationChild?.inputFailure?.message
              ?? observationChild?.infrastructureFailure?.message
              ?? "child observation evaluation did not succeed",
        });
      }
    }
    const successfulRows = rows.filter((row) => row.state === "succeeded");
    const fieldSuccessfulRows = rows.filter((row) => row.field_state === "succeeded");
    const requestedObservationRows = rows.filter((row) => row.observation_state !== "not_requested");
    const successfulObservationRows = rows.filter((row) => row.observation_state === "succeeded");
    const scoredObservationTargets = rows.reduce((sum, row) => sum + row.scored_observation_targets, 0);
    const channelMembers = new Map();
    for (const row of rows) {
      for (const [channel, value] of Object.entries(row.observation_channel_aggregates)) {
        if (!channelMembers.has(channel)) channelMembers.set(channel, []);
        channelMembers.get(channel).push(value);
      }
    }
    const observationChannelAggregates = {};
    for (const [channel, members] of [...channelMembers.entries()].sort(([left], [right]) => left.localeCompare(right))) {
      const validPoints = members.reduce((sum, value) => sum + value.validPoints, 0);
      const sumSquaredResidual = members.reduce((sum, value) => sum + value.sumSquaredResidual, 0);
      const weightedSquared = members.reduce((sum, value) => sum + value.inverseVarianceWeightedSquaredResidual, 0);
      const weightSum = members.reduce((sum, value) => sum + value.inverseVarianceWeightSum, 0);
      const chiSquare = members.reduce((sum, value) => sum + value.chiSquare, 0);
      const degreesFreedom = members.reduce((sum, value) => sum + value.degreesFreedom, 0);
      observationChannelAggregates[channel] = {
        channel,
        unit: members[0].unit,
        scoredTargetCount: members.reduce((sum, value) => sum + value.scoredTargetCount, 0),
        validPoints,
        fittedNuisanceParameters: members.reduce((sum, value) => sum + value.fittedNuisanceParameters, 0),
        sumSquaredResidual,
        rmse: validPoints ? Math.sqrt(sumSquaredResidual / validPoints) : null,
        inverseVarianceWeightedSquaredResidual: weightedSquared,
        inverseVarianceWeightSum: weightSum,
        inverseVarianceWeightedRmse: weightSum ? Math.sqrt(weightedSquared / weightSum) : null,
        chiSquare,
        degreesFreedom,
        reducedChiSquare: degreesFreedom ? chiSquare / degreesFreedom : null,
        gaussianLogLikelihood: members.reduce((sum, value) => sum + value.gaussianLogLikelihood, 0),
      };
    }
    const velocityAggregate = observationChannelAggregates.velocity_m_s ?? null;
    const validObservationPoints = Object.values(observationChannelAggregates)
      .reduce((sum, value) => sum + value.validPoints, 0);
    const ensembleMetricDefinitions = [
      ["iterations", "iterations"],
      ["maximum_relative_update", "maximumRelativeUpdate"],
      ["maximum_equation_residual", "maximumEquationResidual"],
      ["observation_rmse_m_s", "observationRmseMPerS"],
      ["observation_weighted_rmse_m_s", "observationWeightedRmseMPerS"],
      ["observation_chi_square", "observationChiSquare"],
      ["observation_deflection_rmse_arcsec", "observationDeflectionRmseArcsec"],
      ["observation_reduced_shear_rmse", "observationReducedShearRmse"],
      ["observation_image_position_rmse_arcsec", "observationImagePositionRmseArcsec"],
    ];
    const ensembleGroups = new Map();
    for (let index = 0; index < rows.length; index += 1) {
      const definition = record.childJobs[index];
      if (!definition.realization) continue;
      const parentSystemId = definition.parentSystemId ?? definition.systemId;
      if (!ensembleGroups.has(parentSystemId)) ensembleGroups.set(parentSystemId, []);
      ensembleGroups.get(parentSystemId).push({ row: rows[index], definition });
    }
    const ensembleSummaries = [...ensembleGroups.entries()]
      .sort(([left], [right]) => left.localeCompare(right))
      .map(([parentSystemId, members]) => ({
        schemaVersion: "sigma-batch-ensemble-system-summary/1",
        parentSystemId,
        galaxyJobId: members[0].definition.source.galaxyJobId,
        artifact: members[0].definition.source.artifact,
        parentEnsembleBundleSha256: members[0].definition.source.parentEnsembleBundleSha256,
        uncertaintyStatus: members[0].definition.source.uncertaintyStatus,
        realizationCount: members.length,
        succeededRealizations: members.filter(({ row }) => row.state === "succeeded").length,
        fieldSucceededRealizations: members.filter(({ row }) => row.field_state === "succeeded").length,
        observationSucceededRealizations: members.filter(({ row }) => row.observation_state === "succeeded").length,
        metrics: Object.fromEntries(ensembleMetricDefinitions.map(([rowKey, outputKey]) => [
          outputKey,
          distributionSummary(members.map(({ row }) => row[rowKey])),
        ])),
      }));
    const parentSystemCount = new Set(
      record.childJobs.map((definition) => definition.parentSystemId ?? definition.systemId),
    ).size;
    const aggregate = {
      schemaVersion: "sigma-batch-aggregate/2",
      batchId: id,
      systemCount: rows.length,
      succeededSystems: successfulRows.length,
      failedSystems: rows.length - successfulRows.length,
      fieldSucceededSystems: fieldSuccessfulRows.length,
      requestedObservationSystems: requestedObservationRows.length,
      observationSucceededSystems: successfulObservationRows.length,
      convergenceFraction: fieldSuccessfulRows.length / rows.length,
      medianIterations: median(fieldSuccessfulRows.map((row) => row.iterations)),
      medianMaximumRelativeUpdate: median(fieldSuccessfulRows.map((row) => row.maximum_relative_update)),
      maximumEquationResidual: Math.max(0, ...rows.map((row) => row.maximum_equation_residual)),
      modelSha256: record.preflight.modelSha256,
      parameterPolicy: record.parameterPolicy,
      universalGravityParameters: Math.max(0, ...rows.map((row) => row.universal_gravity_parameters)),
      perObjectGravityParameters: Math.max(0, ...rows.map((row) => row.per_object_gravity_parameters)),
      observationAddedGravityParameters: Math.max(
        0,
        ...rows.map((row) => row.observation_added_gravity_parameters),
      ),
      observationScoresAvailable: scoredObservationTargets > 0,
      scoredObservationTargets,
      validObservationPoints,
      parentSystemCount,
      withinSystemUncertainty: {
        status: ensembleSummaries.length
          ? "prior_prediction_spread_not_measurement_posterior"
          : "not_requested",
        ensembleParentCount: ensembleSummaries.length,
        ensembleRealizationCount: [...ensembleGroups.values()].reduce((sum, members) => sum + members.length, 0),
        summaryArtifact: ensembleSummaries.length ? "ensemble_summary.json" : null,
        interpretation: ensembleSummaries.length
          ? "Percentiles summarize predictions across declared baryonic prior realizations; they are not parameter-fit or likelihood-derived credible intervals."
          : "No galaxy density ensemble was propagated in this batch.",
      },
      observationChannelAggregates,
      observationRmseMPerS: velocityAggregate?.rmse ?? null,
      observationInverseVarianceWeightedRmseMPerS: velocityAggregate?.inverseVarianceWeightedRmse ?? null,
      observationChiSquare: velocityAggregate?.chiSquare ?? null,
      observationDegreesFreedom: velocityAggregate?.degreesFreedom ?? null,
      observationReducedChiSquare: velocityAggregate?.reducedChiSquare ?? null,
      claimBoundary: record.preflight.claimBoundary,
    };
    const scientificCore = {
      schemaVersion: "sigma-batch-scientific-result/2",
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
      schemaVersion: "sigma-batch-child-jobs/2",
      items: childManifest,
    });
    await atomicWrite(resolve(artifacts, "aggregate_scores.json"), aggregate);
    await atomicWrite(resolve(artifacts, "ensemble_summary.json"), {
      schemaVersion: "sigma-batch-ensemble-summary/1",
      batchId: id,
      status: aggregate.withinSystemUncertainty.status,
      interpretation: aggregate.withinSystemUncertainty.interpretation,
      systems: ensembleSummaries,
    });
    const perGalaxyColumns = [
      "system_id",
      "parent_system_id",
      "surface_realization",
      "vertical_realization",
      "source_kind",
      "source_id",
      "field_job_id",
      "observation_evaluation_job_id",
      "field_scientific_job_id",
      "observation_scientific_job_id",
      "state",
      "field_state",
      "observation_state",
      "converged",
      "iterations",
      "maximum_relative_update",
      "maximum_equation_residual",
      "universal_gravity_parameters",
      "per_object_gravity_parameters",
      "observation_added_gravity_parameters",
      "observation_targets",
      "scored_observation_targets",
      "valid_observation_points",
      "observation_rmse_m_s",
      "observation_weighted_rmse_m_s",
      "observation_chi_square",
      "observation_degrees_freedom",
      "observation_deflection_rmse_arcsec",
      "observation_deflection_weighted_rmse_arcsec",
      "observation_reduced_shear_rmse",
      "observation_reduced_shear_weighted_rmse",
      "observation_image_position_rmse_arcsec",
      "observation_image_position_weighted_rmse_arcsec",
      "observation_incomplete_topology_targets",
    ];
    await writeFile(
      resolve(artifacts, "per_galaxy.csv"),
      csv(perGalaxyColumns, rows),
      "utf8",
    );
    await writeFile(
      resolve(artifacts, "per_realization.csv"),
      csv(perGalaxyColumns, rows.filter((row) => row.surface_realization !== null)),
      "utf8",
    );
    const ensembleMetricKeys = ensembleMetricDefinitions.map(([, outputKey]) => outputKey);
    const ensembleSummaryColumns = [
      "parent_system_id",
      "galaxy_job_id",
      "artifact",
      "parent_ensemble_bundle_sha256",
      "uncertainty_status",
      "realization_count",
      "succeeded_realizations",
      "field_succeeded_realizations",
      "observation_succeeded_realizations",
      ...ensembleMetricKeys.flatMap((key) => ["count", "p16", "p50", "p84", "minimum", "maximum"]
        .map((statistic) => `${key}_${statistic}`)),
    ];
    const ensembleSummaryRows = ensembleSummaries.map((summary) => ({
      parent_system_id: summary.parentSystemId,
      galaxy_job_id: summary.galaxyJobId,
      artifact: summary.artifact,
      parent_ensemble_bundle_sha256: summary.parentEnsembleBundleSha256,
      uncertainty_status: summary.uncertaintyStatus,
      realization_count: summary.realizationCount,
      succeeded_realizations: summary.succeededRealizations,
      field_succeeded_realizations: summary.fieldSucceededRealizations,
      observation_succeeded_realizations: summary.observationSucceededRealizations,
      ...Object.fromEntries(ensembleMetricKeys.flatMap((key) => Object.entries(summary.metrics[key])
        .map(([statistic, value]) => [`${key}_${statistic}`, value]))),
    }));
    await writeFile(
      resolve(artifacts, "ensemble_summary.csv"),
      csv(ensembleSummaryColumns, ensembleSummaryRows),
      "utf8",
    );
    await writeFile(
      resolve(artifacts, "failures.csv"),
      csv([
        "system_id",
        "parent_system_id",
        "field_job_id",
        "observation_evaluation_job_id",
        "state",
        "category",
        "message",
      ], failureRows),
      "utf8",
    );
    await writeFile(
      resolve(artifacts, "observation_predictions.csv"),
      `system_id,target_id,point_index,radius_m,predicted_speed_m_s,observed_speed_m_s,uncertainty_m_s,residual_m_s,azimuthal_coverage,mean_inward_acceleration_m_s2\n${observationPredictionRows.length ? `${observationPredictionRows.join("\n")}\n` : ""}`,
      "utf8",
    );
    await writeFile(
      resolve(artifacts, "observation_velocity_field_predictions.csv"),
      `system_id,target_id,point_index,row_index,column_index,disk_major_coordinate_m,disk_minor_coordinate_m,circular_radius_m,predicted_circular_speed_m_s,predicted_velocity_m_s,observed_velocity_m_s,uncertainty_m_s,residual_m_s,declared_weight,inward_acceleration_m_s2\n${velocityFieldPredictionRows.length ? `${velocityFieldPredictionRows.join("\n")}\n` : ""}`,
      "utf8",
    );
    await writeFile(
      resolve(artifacts, "observation_multiple_image_predictions.csv"),
      `system_id,target_id,family_id,family_index,image_index,assignment_state,observed_east_arcsec,observed_north_arcsec,position_uncertainty_arcsec,predicted_root_index,predicted_east_arcsec,predicted_north_arcsec,residual_east_arcsec,residual_north_arcsec,separation_arcsec,root_closure_arcsec,root_absolute_magnification\n${multipleImagePredictionRows.length ? `${multipleImagePredictionRows.join("\n")}\n` : ""}`,
      "utf8",
    );
    await writeFile(
      resolve(artifacts, "observation_multiple_image_families.csv"),
      `system_id,target_id,family_id,family_index,distance_ratio,profiled_source_east_arcsec,profiled_source_north_arcsec,observed_images,predicted_roots,matched_images,complete_observed_assignment,excess_predicted_roots,critical_curve_points,state,image_plane_rms_arcsec,matched_subset_diagnostic_rms_arcsec,chi_square,degrees_freedom,fitted_observation_nuisance_parameters,gravity_parameters_added\n${multipleImageFamilyRows.length ? `${multipleImageFamilyRows.join("\n")}\n` : ""}`,
      "utf8",
    );
    const tableRows = rows.map((row) => `<tr><td>${htmlEscape(row.system_id)}</td><td>${htmlEscape(row.parent_system_id)}</td><td>${htmlEscape(row.field_state)}</td><td>${htmlEscape(row.observation_state)}</td><td>${htmlEscape(row.iterations ?? "")}</td><td>${htmlEscape(row.maximum_equation_residual)}</td><td>${htmlEscape(row.observation_rmse_m_s ?? "")}</td><td>${htmlEscape(row.observation_deflection_rmse_arcsec ?? "")}</td><td>${htmlEscape(row.observation_reduced_shear_rmse ?? "")}</td><td>${htmlEscape(row.observation_image_position_rmse_arcsec ?? "")}</td><td>${htmlEscape(row.observation_incomplete_topology_targets)}</td></tr>`).join("");
    const observationScope = aggregate.observationScoresAvailable
      ? `Typed observations (${[...observationKinds].sort().join(", ")}) were scored for ${aggregate.scoredObservationTargets} target(s). Velocity, deflection, and reduced-shear scores remain separate.`
      : requestedObservationRows.length
        ? `Observation targets (${[...observationKinds].sort().join(", ")}) were evaluated but none produced a complete score; incomplete topology remains an explicit non-score.`
        : "No observation targets were supplied, so this report measures numerical execution and convergence only.";
    const ensembleScope = ensembleSummaries.length
      ? `${aggregate.withinSystemUncertainty.ensembleRealizationCount} baryonic prior realization(s) from ${ensembleSummaries.length} parent system(s) were propagated. See <code>ensemble_summary.json</code> for p16/p50/p84 prediction spread; these are not likelihood-derived credible intervals.`
      : "No baryonic density ensemble was requested in this batch.";
    await writeFile(
      resolve(artifacts, "report.html"),
      `<!doctype html><html lang="en"><meta charset="utf-8"><title>Batch ${htmlEscape(id)}</title><style>body{font:16px system-ui;max-width:1180px;margin:3rem auto;padding:0 1rem;color:#182026}table{border-collapse:collapse;width:100%}th,td{border:1px solid #ccd4da;padding:.5rem;text-align:left}code{background:#eef1f3;padding:.1rem .3rem}</style><h1>Formula-independent field batch</h1><p><code>${htmlEscape(id)}</code></p><p>One confirmed model was run over ${rows.length} resolved system realization(s) from ${parentSystemCount} submitted parent system(s) with <strong>${htmlEscape(record.parameterPolicy.mode)}</strong> parameters. ${fieldSuccessfulRows.length} field solve(s) and ${successfulObservationRows.length}/${requestedObservationRows.length} requested observation evaluation(s) succeeded.</p><p><strong>Scientific boundary:</strong> ${htmlEscape(observationScope)}</p><p><strong>Baryonic uncertainty:</strong> ${ensembleScope}</p><table><thead><tr><th>System realization</th><th>Parent system</th><th>Field state</th><th>Observation state</th><th>Iterations</th><th>Maximum equation residual</th><th>Velocity RMSE (m/s)</th><th>Deflection RMSE (arcsec)</th><th>Reduced-shear RMSE</th><th>Image-position coordinate RMSE (arcsec)</th><th>Incomplete-topology targets</th></tr></thead><tbody>${tableRows}</tbody></table></html>`,
      "utf8",
    );
    await writeFile(
      resolve(artifacts, "llm_briefing.md"),
      `# Batch briefing\n\n- Batch: \`${id}\`\n- Model SHA-256: \`${record.preflight.modelSha256}\`\n- Parameter policy: \`${record.parameterPolicy.mode}\`\n- Parent systems: ${parentSystemCount}\n- Resolved system realizations: ${rows.length}\n- Baryonic ensemble realizations: ${aggregate.withinSystemUncertainty.ensembleRealizationCount}\n- Baryonic uncertainty status: \`${aggregate.withinSystemUncertainty.status}\`\n- Successful numerical solves: ${fieldSuccessfulRows.length}\n- Successful requested observation evaluations: ${successfulObservationRows.length}/${requestedObservationRows.length}\n- Per-object gravity parameters: ${aggregate.perObjectGravityParameters}\n- Gravity parameters added by observation evaluation: ${aggregate.observationAddedGravityParameters}\n- Observation scores available: ${aggregate.observationScoresAvailable ? `yes (${[...observationKinds].sort().join(", ")})` : "no"}\n- Scored observation targets: ${aggregate.scoredObservationTargets}\n- Velocity RMSE (m/s): ${aggregate.observationRmseMPerS ?? "not available"}\n- Deflection RMSE (arcsec): ${aggregate.observationChannelAggregates.deflection_arcsec?.rmse ?? "not available"}\n- Reduced-shear RMSE: ${aggregate.observationChannelAggregates.reduced_shear_dimensionless?.rmse ?? "not available"}\n- Raw image-position coordinate RMSE (arcsec): ${aggregate.observationChannelAggregates.image_position_arcsec?.rmse ?? "not available"}\n- Incomplete-topology targets: ${rows.reduce((sum, row) => sum + row.observation_incomplete_topology_targets, 0)}\n\nThis deterministic briefing distinguishes immutable field execution, separately cached massive-tracer and photon observation evaluation, and every score channel. Ensemble percentiles are prediction spread under declared baryonic priors, not likelihood-derived posterior intervals. It must not describe an unscored channel as validated.\n`,
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
      "ensemble_summary.csv",
      "ensemble_summary.json",
      "failures.csv",
      "llm_briefing.md",
      "model.json",
      "observation_predictions.csv",
      "observation_velocity_field_predictions.csv",
      "observation_multiple_image_predictions.csv",
      "observation_multiple_image_families.csv",
      "per_galaxy.csv",
      "per_realization.csv",
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
      schemaVersion: "sigma-batch-manifest/2",
      batchId: id,
      state: failureRows.length ? "completed_with_failures" : "succeeded",
      scientificResultSha256,
      modelSha256: record.preflight.modelSha256,
      parameterPolicy: record.parameterPolicy,
      systemCount: rows.length,
      artifactIndexSha256: digest(await readFile(resolve(artifacts, "artifact_index.json"))),
      fieldWorkerSourceSha256: this.fieldService.workerSourceSha256,
      observationWorkerSourceSha256: this.fieldService.observationWorkerSourceSha256,
      galaxyEnsembleMaterializerSourceSha256: this.fieldService.galaxyEnsembleMaterializerSourceSha256,
      parentSystemCount,
      ensembleRealizationCount: aggregate.withinSystemUncertainty.ensembleRealizationCount,
      reportScope: aggregate.observationScoresAvailable
        ? "composed_field_execution_and_massive_tracer_observation_scoring"
        : "numerical_execution_not_observation_validation",
    };
    const manifest = { ...manifestCore, manifestSha256: sha256(manifestCore) };
    await atomicWrite(resolve(artifacts, "manifest.json"), manifest);
    record.state = manifest.state;
    record.phase = "complete";
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
