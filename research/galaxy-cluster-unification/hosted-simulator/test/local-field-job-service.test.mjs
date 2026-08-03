import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { mkdtemp, mkdir, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { resolve } from "node:path";
import test from "node:test";
import { sha256 } from "../lib/canonical.mjs";
import { validateFieldModel } from "../lib/field-model.mjs";
import { LocalFieldJobService, LocalServiceError } from "../lib/local-field-job-service.mjs";

function digest(value) {
  return createHash("sha256").update(value).digest("hex");
}

function model() {
  const manifest = {
    schemaVersion: "sigma-field-model/1",
    name: "Local service manufactured model",
    modelClass: "stationary_elliptic",
    source: { format: "plain_text", text: "laplacian(u) = forcing", confirmedCanonical: false },
    geometry: { coordinateSystem: "cartesian_2d", dimensions: 2, domain: { lengthUnit: "m", boundaryExtent: "unit square" } },
    fields: {
      forcing: { rank: "scalar", role: "source", unit: "1/s^2", datasetKey: "forcing" },
      u: { rank: "scalar", role: "solved", unit: "m^2/s^2", boundary: { type: "dirichlet", value: 0 } },
    },
    parameters: {},
    equations: [{ id: "manufactured", kind: "equality", lhs: { op: "laplacian", args: [{ field: "u" }] }, rhs: { field: "forcing" } }],
    observables: [{ id: "gradient", target: "diagnostic", rank: "vector", unit: "m/s^2", expression: { op: "gradient", args: [{ field: "u" }] } }],
    dataRequirements: [{ key: "forcing", rank: "scalar", unit: "1/s^2" }],
    solver: { family: "finite_volume_elliptic", relativeTolerance: 1e-8, maxIterations: 8, damping: 1 },
    parameterPolicy: { mode: "universal_fixed", perObjectParameters: [] },
  };
  return bindConfirmation(manifest);
}

function bindConfirmation(manifest) {
  manifest.source.confirmedCanonical = false;
  delete manifest.source.confirmedModelSha256;
  manifest.source.confirmedCanonical = true;
  manifest.source.confirmedModelSha256 = validateFieldModel(manifest).modelSha256;
  return manifest;
}

function bundle() {
  const core = {
    schemaVersion: "sigma-array-bundle/1",
    geometry: { coordinateSystem: "cartesian_2d", dimensions: 2, spacing: [0.0625, 0.0625], lengthUnit: "m" },
    arrays: [{ key: "forcing", npzKey: "raw_forcing", unit: "1/s^2", rank: "scalar", role: "source", dtype: "<f8", shape: [17, 17], elementCount: 289, contentSha256: "1".repeat(64) }],
    provenance: { kind: "test_fixture" },
    license: { id: "CC0-1.0", redistributionAllowed: true },
  };
  return { ...core, bundleSha256: sha256(core) };
}

function request() {
  return {
    schemaVersion: "sigma-field-job-request/1",
    spacing: [0.0625, 0.0625],
    boundaryFields: { u: { value: 0 } },
    requestedObservables: ["gradient"],
    seed: 7,
  };
}

function galaxyBundle() {
  const core = {
    schemaVersion: "sigma-array-bundle/1",
    geometry: { coordinateSystem: "cartesian_2d", dimensions: 2, spacing: [0.2, 0.2], lengthUnit: "kpc" },
    arrays: ["gas_surface_density", "stellar_surface_density"].map((key, index) => ({
      key, npzKey: key, unit: "M_sun/kpc^2", rank: "scalar", role: "source", dtype: "<f8",
      shape: [17, 17], elementCount: 289, contentSha256: String(index + 1).repeat(64),
    })),
    provenance: { kind: "test_fixture" },
    license: { id: "CC0-1.0", redistributionAllowed: true },
  };
  return { ...core, bundleSha256: sha256(core) };
}

function inverseBundle() {
  const arrays = [];
  for (const index of [1, 2]) {
    arrays.push(
      {
        key: `baryons_${index}`, npzKey: `baryons_${index}`, unit: "kg/m^2",
        rank: "scalar", role: "source", scientificRole: "baryonic_input",
        dtype: "<f8", shape: [17, 17], elementCount: 289, contentSha256: String(index).repeat(64),
      },
      {
        key: `response_${index}`, npzKey: `response_${index}`, unit: "kg/m^2",
        rank: "scalar", role: "auxiliary", scientificRole: "model_derived_discovery_target",
        dtype: "<f8", shape: [17, 17], elementCount: 289, contentSha256: String(index + 2).repeat(64),
      },
      {
        key: `uncertainty_${index}`, npzKey: `uncertainty_${index}`, unit: "kg/m^2",
        rank: "scalar", role: "uncertainty", scientificRole: "nuisance_or_calibration",
        dtype: "<f8", shape: [17, 17], elementCount: 289, contentSha256: String(index + 4).repeat(64),
      },
    );
  }
  const core = {
    schemaVersion: "sigma-array-bundle/1",
    geometry: { coordinateSystem: "cartesian_2d", dimensions: 2, spacing: [1, 1], lengthUnit: "kpc" },
    arrays,
    provenance: { kind: "synthetic_inverse_fixture" },
    license: { id: "CC0-1.0", redistributionAllowed: true },
  };
  return { ...core, bundleSha256: sha256(core) };
}

function inverseSubmission(dataUploadId) {
  return {
    schemaVersion: "sigma-inverse-response-job-submit/1",
    dataUploadId,
    systems: [1, 2].map((index) => ({
      id: `SYNTH-${index}`,
      sourceKey: `baryons_${index}`,
      targetKey: `response_${index}`,
      uncertaintyKey: `uncertainty_${index}`,
    })),
    kernel: { shape: [5, 5], ridge: 1e-10, smoothness: 1e-8, nonnegative: true },
    uncertainty: { ensembleSize: 20, seed: 17 },
    nullControls: { kind: "source_radial_angle_shuffle", count: 19, seed: 23 },
    outputLicense: { id: "CC-BY-4.0", redistributionAllowed: true },
  };
}

async function successfulRunner({ jobDirectory }) {
  const root = resolve(jobDirectory, "artifacts");
  await mkdir(root, { recursive: true });
  const scientific = Buffer.from('{"state":"succeeded"}\n');
  await writeFile(resolve(root, "scientific_result.json"), scientific);
  const artifactIndex = {
    schemaVersion: "sigma-field-artifact-index/1",
    jobId: "fieldjob_test",
    artifacts: [{ path: "scientific_result.json", bytes: scientific.length, sha256: digest(scientific) }],
  };
  const indexContent = Buffer.from(`${JSON.stringify(artifactIndex)}\n`);
  await writeFile(resolve(root, "artifact_index.json"), indexContent);
  const manifestCore = {
    schemaVersion: "sigma-field-run-manifest/1",
    state: "succeeded",
    jobId: "fieldjob_test",
    scientificResultSha256: "2".repeat(64),
    artifactIndexSha256: digest(indexContent),
  };
  const manifest = { ...manifestCore, manifestSha256: sha256(manifestCore) };
  await writeFile(resolve(root, "manifest.json"), `${JSON.stringify(manifest)}\n`);
  return { exitCode: 0, exitSignal: null, timedOut: false, stdout: "ok", stderr: "" };
}

function publicationRunner({ payloadBytes = 64, addUnindexedFile = false } = {}) {
  return async ({ jobDirectory }) => {
    const root = resolve(jobDirectory, "artifacts");
    await mkdir(root, { recursive: true });
    const scientific = Buffer.alloc(payloadBytes, 7);
    await writeFile(resolve(root, "scientific_result.json"), scientific);
    const artifactIndex = {
      schemaVersion: "sigma-field-artifact-index/1",
      jobId: "fieldjob_quota_test",
      artifacts: [{ path: "scientific_result.json", bytes: scientific.length, sha256: digest(scientific) }],
    };
    const indexContent = Buffer.from(`${JSON.stringify(artifactIndex)}\n`);
    await writeFile(resolve(root, "artifact_index.json"), indexContent);
    const manifestCore = {
      schemaVersion: "sigma-field-run-manifest/1",
      state: "succeeded",
      jobId: "fieldjob_quota_test",
      scientificResultSha256: digest(scientific),
      artifactIndexSha256: digest(indexContent),
    };
    const manifest = { ...manifestCore, manifestSha256: sha256(manifestCore) };
    await writeFile(resolve(root, "manifest.json"), `${JSON.stringify(manifest)}\n`);
    if (addUnindexedFile) await writeFile(resolve(root, "unindexed.bin"), Buffer.from("not declared"));
    return { exitCode: 0, exitSignal: null, timedOut: false, stdout: "ok", stderr: "" };
  };
}

function observationFieldModel() {
  const value = model();
  value.observables[0] = {
    ...value.observables[0],
    id: "acceleration",
    target: "massive_tracers",
  };
  return bindConfirmation(value);
}

function observationFieldRequest() {
  const value = request();
  value.requestedObservables = ["acceleration"];
  return value;
}

function circularTarget(radius = 1) {
  return {
    schemaVersion: "sigma-observation-target/1",
    id: "decoupled-curve",
    kind: "circular_speed_curve",
    observable: "acceleration",
    centerM: [0, 0],
    planeAxes: [0, 1],
    radiiM: [radius],
    provenance: { kind: "P0732 service fixture" },
    license: { id: "CC0-1.0", redistributionAllowed: true },
  };
}

async function successfulSourceFieldRunner({ jobDirectory }) {
  const root = resolve(jobDirectory, "artifacts");
  await mkdir(root, { recursive: true });
  const fieldModel = JSON.parse(await readFile(resolve(jobDirectory, "model.json"), "utf8"));
  const validation = validateFieldModel(fieldModel);
  const fieldJobSha256 = "4".repeat(64);
  const fieldJob = {
    schemaVersion: "sigma-field-job/1",
    id: `fieldjob_${fieldJobSha256.slice(0, 24)}`,
    jobSha256: fieldJobSha256,
    modelSha256: validation.modelSha256,
    geometry: { coordinateSystem: "cartesian_2d", dimensions: 2, spacing: [0.0625, 0.0625], origin: [-0.5, -0.5], lengthUnit: "m" },
  };
  const observable = Buffer.from("fixture-observable-archive");
  const resultSha256 = "5".repeat(64);
  const scientificResult = {
    schemaVersion: "sigma-field-result/1",
    state: "succeeded",
    converged: true,
    jobId: fieldJob.id,
    jobSha256: fieldJobSha256,
    resultSha256,
    observables: [
      { key: "acceleration__axis0", dtype: "<f8", shape: [17, 17], contentSha256: "6".repeat(64) },
      { key: "acceleration__axis1", dtype: "<f8", shape: [17, 17], contentSha256: "7".repeat(64) },
    ],
  };
  const contents = new Map([
    ["model.json", Buffer.from(`${JSON.stringify(fieldModel)}\n`)],
    ["job.json", Buffer.from(`${JSON.stringify(fieldJob)}\n`)],
    ["scientific_result.json", Buffer.from(`${JSON.stringify(scientificResult)}\n`)],
    ["observables.npz", observable],
  ]);
  await Promise.all([...contents].map(([name, content]) => writeFile(resolve(root, name), content)));
  const artifactIndex = {
    schemaVersion: "sigma-field-artifact-index/1",
    jobId: fieldJob.id,
    artifacts: [...contents].map(([path, content]) => ({ path, bytes: content.length, sha256: digest(content) })),
  };
  const indexContent = Buffer.from(`${JSON.stringify(artifactIndex)}\n`);
  await writeFile(resolve(root, "artifact_index.json"), indexContent);
  const manifestCore = {
    schemaVersion: "sigma-field-run-manifest/1",
    state: "succeeded",
    jobId: fieldJob.id,
    scientificResultSha256: resultSha256,
    artifactIndexSha256: digest(indexContent),
  };
  const manifest = { ...manifestCore, manifestSha256: sha256(manifestCore) };
  await writeFile(resolve(root, "manifest.json"), `${JSON.stringify(manifest)}\n`);
  return { exitCode: 0, exitSignal: null, timedOut: false, stdout: "ok", stderr: "" };
}

async function prepareObservationSubmission(service) {
  const fieldUpload = await readyUpload(service, Buffer.from("p0732-field-source"));
  const source = await service.createFieldJob({
    schemaVersion: "sigma-field-job-submit/1",
    model: observationFieldModel(),
    dataUploadId: fieldUpload.id,
    request: observationFieldRequest(),
  });
  await service.waitForIdle();
  const observationUpload = await readyUpload(service, Buffer.from("p0732-observation-data"));
  return {
    source,
    submission: {
      schemaVersion: "sigma-observation-evaluation-job-submit/1",
      fieldJobId: source.id,
      dataUploadId: observationUpload.id,
      observationTargets: [circularTarget()],
    },
  };
}

async function fixture(t, options = {}) {
  const root = await mkdtemp(resolve(tmpdir(), "sigma-local-service-"));
  t.after(async () => rm(root, { recursive: true, force: true }));
  const service = new LocalFieldJobService({
    root,
    projectRoot: resolve(import.meta.dirname, "..", ".."),
    runner: successfulRunner,
    ...options,
  });
  await service.initialize();
  t.after(async () => service.close());
  return service;
}

async function readyUpload(service, archive = Buffer.from("npz-test-archive")) {
  const ticket = await service.createUpload({
    schemaVersion: "sigma-data-upload-request/1",
    inputBundle: bundle(),
    archive: { sha256: digest(archive), bytes: archive.length },
  });
  await service.putUploadContent(ticket.id, archive);
  return ticket;
}

test("upload tickets enforce byte hashes and become immutable", async (t) => {
  const service = await fixture(t);
  const archive = Buffer.from("npz-test-archive");
  const ticket = await service.createUpload({ schemaVersion: "sigma-data-upload-request/1", inputBundle: bundle(), archive: { sha256: digest(archive), bytes: archive.length } });
  assert.equal(ticket.state, "awaiting_content");
  await assert.rejects(() => service.putUploadContent(ticket.id, Buffer.from("wrong-size")), /byte count/);
  const ready = await service.putUploadContent(ticket.id, archive);
  assert.equal(ready.state, "ready");
  assert.equal(ready.integrity.archiveSha256Verified, true);
  const replay = await service.putUploadContent(ticket.id, archive);
  assert.equal(replay.state, "ready");
});

test("queued field jobs expose status, events, immutable artifacts, and duplicate identity", async (t) => {
  const service = await fixture(t);
  const upload = await readyUpload(service);
  const submission = await service.createFieldJob({ schemaVersion: "sigma-field-job-submit/1", model: model(), dataUploadId: upload.id, request: request() });
  assert.match(submission.id, /^job_[0-9a-f]{24}$/);
  assert.equal(submission.duplicate, false);
  assert.equal(submission.parameterAccounting.perObject, 0);
  await service.waitForIdle();
  const completed = await service.getFieldJob(submission.id);
  assert.equal(completed.state, "succeeded");
  assert.equal(completed.scientificJobId, "fieldjob_test");
  const events = await service.getEvents(submission.id);
  assert.deepEqual(events.items.map((event) => event.state), ["queued", "running", "succeeded"]);
  const artifacts = await service.getArtifacts(submission.id);
  assert.equal(artifacts.items.length, 1);
  const artifact = await service.getArtifact(submission.id, "scientific_result.json");
  assert.equal(digest(artifact.content), artifact.record.sha256);
  await assert.rejects(() => service.getArtifact(submission.id, "..%2Frecord.json"), /unknown artifact/);
  const duplicate = await service.createFieldJob({ schemaVersion: "sigma-field-job-submit/1", model: model(), dataUploadId: upload.id, request: request() });
  assert.equal(duplicate.id, submission.id);
  assert.equal(duplicate.duplicate, true);
});

test("queued galaxy jobs use separate routes and zero gravity parameters", async (t) => {
  const service = await fixture(t);
  const archive = Buffer.from("galaxy-npz-test-archive");
  const ticket = await service.createUpload({
    schemaVersion: "sigma-data-upload-request/1",
    inputBundle: galaxyBundle(),
    archive: { sha256: digest(archive), bytes: archive.length },
  });
  await service.putUploadContent(ticket.id, archive);
  const submission = await service.createGalaxyJob({
    schemaVersion: "sigma-galaxy-job-submit/1",
    operation: "extract_roundtrip",
    dataUploadId: ticket.id,
    galaxy: "FIXTURE",
    vertical: { enabled: true, realizations: 2, zCells: 17, seed: 4 },
    outputLicense: { id: "CC0-1.0", redistributionAllowed: true },
  });
  assert.equal(submission.jobType, "galaxy");
  assert.match(submission.links.self, /^\/api\/v1\/galaxy-jobs\//);
  assert.equal(submission.parameterAccounting.gravityPerObject, 0);
  await service.waitForIdle();
  const completed = await service.getGalaxyJob(submission.id);
  assert.equal(completed.state, "succeeded");
  const artifacts = await service.getArtifacts(submission.id);
  assert.equal(artifacts.schemaVersion, "sigma-galaxy-job-artifact-response/1");
  assert.match(artifacts.items[0].url, /^\/api\/v1\/galaxy-jobs\//);
  await assert.rejects(() => service.getFieldJob(submission.id), /unknown field job/);
});

test("queued inverse response jobs retain discovery roles and separate routes", async (t) => {
  const service = await fixture(t);
  const archive = Buffer.from("inverse-response-npz-test-archive");
  const ticket = await service.createUpload({
    schemaVersion: "sigma-data-upload-request/1",
    inputBundle: inverseBundle(),
    archive: { sha256: digest(archive), bytes: archive.length },
  });
  await service.putUploadContent(ticket.id, archive);
  const submission = await service.createInverseResponseJob(inverseSubmission(ticket.id));
  assert.equal(submission.jobType, "inverse_response");
  assert.match(submission.links.self, /^\/api\/v1\/inverse-response-jobs\//);
  assert.equal(submission.parameterAccounting.fittedPerSystemGravityParameters, 0);
  assert.equal(submission.preflight.dataRoleAudit[0].targetRole, "model_derived_discovery_target");
  await service.waitForIdle();
  const completed = await service.getInverseResponseJob(submission.id);
  assert.equal(completed.state, "succeeded");
  const artifacts = await service.getArtifacts(submission.id);
  assert.equal(artifacts.schemaVersion, "sigma-inverse-response-job-artifact-response/1");
  assert.match(artifacts.items[0].url, /^\/api\/v1\/inverse-response-jobs\//);
  const listed = await service.listInverseResponseJobs();
  assert.equal(listed.items.length, 1);
  await assert.rejects(() => service.getFieldJob(submission.id), /unknown field job/);
});

test("decoupled observation jobs reuse one immutable field and have independent identity", async (t) => {
  const calls = { field: 0, observation: 0 };
  const runner = async (argumentsValue) => {
    if (argumentsValue.jobType === "observation_evaluation") {
      calls.observation += 1;
      return successfulRunner(argumentsValue);
    }
    calls.field += 1;
    return successfulSourceFieldRunner(argumentsValue);
  };
  const service = await fixture(t, { runner });
  const { source, submission } = await prepareObservationSubmission(service);
  assert.equal((await service.getFieldJob(source.id)).state, "succeeded");
  assert.equal(calls.field, 1);
  const queued = await service.createObservationEvaluationJob(submission);
  assert.equal(queued.jobType, "observation_evaluation");
  assert.equal(queued.fieldJobId, source.id);
  assert.equal(queued.evaluationAddedGravityParameters, 0);
  assert.match(queued.links.self, /^\/api\/v1\/observation-evaluation-jobs\//);
  await service.waitForIdle();
  assert.equal(calls.field, 1);
  assert.equal(calls.observation, 1);
  const completed = await service.getObservationEvaluationJob(queued.id);
  assert.equal(completed.state, "succeeded");
  const duplicate = await service.createObservationEvaluationJob(submission);
  assert.equal(duplicate.id, queued.id);
  assert.equal(duplicate.duplicate, true);
  assert.equal(calls.observation, 1);
  const changed = await service.createObservationEvaluationJob({
    ...submission,
    observationTargets: [circularTarget(2)],
  });
  assert.notEqual(changed.id, queued.id);
  await service.waitForIdle();
  assert.equal(calls.field, 1);
  assert.equal(calls.observation, 2);
  const listed = await service.listObservationEvaluationJobs();
  assert.equal(listed.items.length, 2);
  const events = await service.getEvents(queued.id);
  assert.equal(events.schemaVersion, "sigma-observation-evaluation-job-events/1");
  const artifacts = await service.getArtifacts(queued.id);
  assert.equal(artifacts.schemaVersion, "sigma-observation-evaluation-job-artifact-response/1");
  assert.match(artifacts.items[0].url, /^\/api\/v1\/observation-evaluation-jobs\//);
  await assert.rejects(() => service.getFieldJob(queued.id), /unknown field job/);
});

test("running observation evaluation can be cancelled without publishing artifacts", async (t) => {
  const runner = (argumentsValue) => {
    if (argumentsValue.jobType !== "observation_evaluation") return successfulSourceFieldRunner(argumentsValue);
    return new Promise((resolvePromise) => {
      argumentsValue.signal.addEventListener(
        "abort",
        () => resolvePromise({ exitCode: null, exitSignal: "SIGTERM", timedOut: false, stdout: "", stderr: "" }),
        { once: true },
      );
    });
  };
  const service = await fixture(t, { runner });
  const { submission } = await prepareObservationSubmission(service);
  const queued = await service.createObservationEvaluationJob(submission);
  for (let attempt = 0; attempt < 100 && (await service.getObservationEvaluationJob(queued.id)).state !== "running"; attempt++) {
    await new Promise((resolvePromise) => setTimeout(resolvePromise, 5));
  }
  const cancelled = await service.cancelObservationEvaluationJob(queued.id);
  assert.equal(cancelled.state, "cancelled");
  await service.waitForIdle();
  await assert.rejects(
    () => service.getArtifacts(queued.id),
    (error) => error instanceof LocalServiceError && error.code === "artifacts_not_ready",
  );
});

test("completed observation evaluation recovers without reevaluating after restart", async (t) => {
  const runner = (argumentsValue) => argumentsValue.jobType === "observation_evaluation"
    ? successfulRunner(argumentsValue)
    : successfulSourceFieldRunner(argumentsValue);
  const service = await fixture(t, { runner });
  const { submission } = await prepareObservationSubmission(service);
  const queued = await service.createObservationEvaluationJob(submission);
  await service.waitForIdle();
  const recordPath = resolve(service.root, "jobs", queued.id, "record.json");
  const record = JSON.parse(await readFile(recordPath, "utf8"));
  record.state = "running";
  await writeFile(recordPath, `${JSON.stringify(record, null, 2)}\n`);
  const recovered = new LocalFieldJobService({
    root: service.root,
    projectRoot: service.projectRoot,
    runner: async () => { throw new Error("completed observation job must not reevaluate"); },
  });
  await recovered.initialize();
  t.after(async () => recovered.close());
  const recoveredRecord = await recovered.getObservationEvaluationJob(queued.id);
  assert.equal(recoveredRecord.state, "succeeded");
  assert.equal(recoveredRecord.recoveredAfterRestart, true);
});

test("artifact mutation is rejected and completed manifests recover after restart", async (t) => {
  const service = await fixture(t);
  const upload = await readyUpload(service, Buffer.from("recovery-test"));
  const submission = await service.createFieldJob({ schemaVersion: "sigma-field-job-submit/1", model: model(), dataUploadId: upload.id, request: request() });
  await service.waitForIdle();
  const artifactPath = resolve(service.root, "jobs", submission.id, "artifacts", "scientific_result.json");
  await writeFile(artifactPath, "changed after publication");
  await assert.rejects(
    () => service.getArtifact(submission.id, "scientific_result.json"),
    (error) => error instanceof LocalServiceError && error.code === "artifact_integrity_failed",
  );
  await successfulRunner({ jobDirectory: resolve(service.root, "jobs", submission.id) });
  const recordPath = resolve(service.root, "jobs", submission.id, "record.json");
  const record = JSON.parse(await readFile(recordPath, "utf8"));
  record.state = "running";
  await writeFile(recordPath, `${JSON.stringify(record, null, 2)}\n`);
  const recovered = new LocalFieldJobService({
    root: service.root,
    projectRoot: service.projectRoot,
    runner: async () => { throw new Error("recovered jobs must not rerun"); },
  });
  await recovered.initialize();
  t.after(async () => recovered.close());
  const recoveredRecord = await recovered.getFieldJob(submission.id);
  assert.equal(recoveredRecord.state, "succeeded");
  assert.equal(recoveredRecord.recoveredAfterRestart, true);
});

test("worker publication rejects artifact bytes beyond the configured quota", async (t) => {
  const service = await fixture(t, {
    runner: publicationRunner({ payloadBytes: 4096 }),
    maxArtifactBytes: 1024,
  });
  const upload = await readyUpload(service, Buffer.from("artifact-quota-test"));
  const submission = await service.createFieldJob({
    schemaVersion: "sigma-field-job-submit/1",
    model: model(),
    dataUploadId: upload.id,
    request: request(),
  });
  await service.waitForIdle();
  const failed = await service.getFieldJob(submission.id);
  assert.equal(failed.state, "infrastructure_failed");
  assert.equal(failed.infrastructureFailure.code, "artifact_quota_exceeded");
  await assert.rejects(
    () => service.getArtifacts(submission.id),
    (error) => error instanceof LocalServiceError && error.code === "artifacts_not_ready",
  );
});

test("worker publication rejects files omitted from the immutable artifact index", async (t) => {
  const service = await fixture(t, {
    runner: publicationRunner({ addUnindexedFile: true }),
  });
  const upload = await readyUpload(service, Buffer.from("unindexed-artifact-test"));
  const submission = await service.createFieldJob({
    schemaVersion: "sigma-field-job-submit/1",
    model: model(),
    dataUploadId: upload.id,
    request: request(),
  });
  await service.waitForIdle();
  const failed = await service.getFieldJob(submission.id);
  assert.equal(failed.state, "infrastructure_failed");
  assert.equal(failed.infrastructureFailure.code, "artifact_integrity_failed");
});

test("restart recovery refuses a completed manifest whose artifacts were mutated", async (t) => {
  const service = await fixture(t);
  const upload = await readyUpload(service, Buffer.from("restart-integrity-test"));
  const submission = await service.createFieldJob({
    schemaVersion: "sigma-field-job-submit/1",
    model: model(),
    dataUploadId: upload.id,
    request: request(),
  });
  await service.waitForIdle();
  const jobDirectory = resolve(service.root, "jobs", submission.id);
  await writeFile(resolve(jobDirectory, "artifacts", "scientific_result.json"), "mutated");
  const recordPath = resolve(jobDirectory, "record.json");
  const record = JSON.parse(await readFile(recordPath, "utf8"));
  record.state = "running";
  await writeFile(recordPath, `${JSON.stringify(record, null, 2)}\n`);
  const recovered = new LocalFieldJobService({
    root: service.root,
    projectRoot: service.projectRoot,
    runner: async () => { throw new Error("corrupt completed output must not rerun or publish"); },
  });
  await recovered.initialize();
  t.after(async () => recovered.close());
  const failed = await recovered.getFieldJob(submission.id);
  assert.equal(failed.state, "infrastructure_failed");
  assert.equal(failed.recoveredAfterRestart, true);
  assert.equal(failed.infrastructureFailure.code, "artifact_integrity_failed");
});

test("running jobs can be cancelled without publishing scientific artifacts", async (t) => {
  const runner = ({ signal }) => new Promise((resolvePromise) => {
    signal.addEventListener("abort", () => resolvePromise({ exitCode: null, exitSignal: "SIGTERM", timedOut: false, stdout: "", stderr: "" }), { once: true });
  });
  const service = await fixture(t, { runner });
  const upload = await readyUpload(service, Buffer.from("cancel-test"));
  const submission = await service.createFieldJob({ schemaVersion: "sigma-field-job-submit/1", model: model(), dataUploadId: upload.id, request: request() });
  for (let attempt = 0; attempt < 100 && (await service.getFieldJob(submission.id)).state !== "running"; attempt++) {
    await new Promise((resolvePromise) => setTimeout(resolvePromise, 5));
  }
  const cancelled = await service.cancelFieldJob(submission.id);
  assert.equal(cancelled.state, "cancelled");
  await service.waitForIdle();
  await assert.rejects(() => service.getArtifacts(submission.id), (error) => error instanceof LocalServiceError && error.code === "artifacts_not_ready");
});

test("worker input rejection is not misreported as infrastructure failure", async (t) => {
  const runner = async () => ({
    exitCode: 2,
    exitSignal: null,
    timedOut: false,
    stdout: "",
    stderr: `${JSON.stringify({ schemaVersion: "sigma-field-job-cli-error/1", state: "rejected_input", errorType: "ValueError", message: "array content hash mismatch" })}\n`,
  });
  const service = await fixture(t, { runner });
  const upload = await readyUpload(service, Buffer.from("internally-invalid-npz"));
  const submission = await service.createFieldJob({ schemaVersion: "sigma-field-job-submit/1", model: model(), dataUploadId: upload.id, request: request() });
  await service.waitForIdle();
  const rejected = await service.getFieldJob(submission.id);
  assert.equal(rejected.state, "rejected_input");
  assert.equal(rejected.inputFailure.message, "array content hash mismatch");
  assert.equal("infrastructureFailure" in rejected, false);
});

test("graceful shutdown leaves interrupted work resumable", async (t) => {
  const blockingRunner = ({ signal }) => new Promise((resolvePromise) => {
    const interrupted = () => resolvePromise({ exitCode: null, exitSignal: "SIGTERM", timedOut: false, stdout: "", stderr: "" });
    if (signal.aborted) interrupted();
    else signal.addEventListener("abort", interrupted, { once: true });
  });
  const service = await fixture(t, { runner: blockingRunner });
  const upload = await readyUpload(service, Buffer.from("restartable-job"));
  const submission = await service.createFieldJob({ schemaVersion: "sigma-field-job-submit/1", model: model(), dataUploadId: upload.id, request: request() });
  for (let attempt = 0; attempt < 100 && (await service.getFieldJob(submission.id)).state !== "running"; attempt++) {
    await new Promise((resolvePromise) => setTimeout(resolvePromise, 5));
  }
  await service.close();
  assert.equal((await service.getFieldJob(submission.id)).state, "queued");
  const restarted = new LocalFieldJobService({
    root: service.root,
    projectRoot: service.projectRoot,
    runner: successfulRunner,
  });
  await restarted.initialize();
  t.after(async () => restarted.close());
  await restarted.waitForIdle();
  assert.equal((await restarted.getFieldJob(submission.id)).state, "succeeded");
});
