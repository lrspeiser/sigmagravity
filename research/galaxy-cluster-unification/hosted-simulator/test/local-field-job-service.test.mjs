import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { mkdtemp, mkdir, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { resolve } from "node:path";
import test from "node:test";
import { sha256 } from "../lib/canonical.mjs";
import { LocalFieldJobService, LocalServiceError } from "../lib/local-field-job-service.mjs";

function digest(value) {
  return createHash("sha256").update(value).digest("hex");
}

function model() {
  return {
    schemaVersion: "sigma-field-model/1",
    name: "Local service manufactured model",
    modelClass: "stationary_elliptic",
    source: { format: "plain_text", text: "laplacian(u) = forcing", confirmedCanonical: true },
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
  const manifest = {
    schemaVersion: "sigma-field-run-manifest/1",
    state: "succeeded",
    jobId: "fieldjob_test",
    scientificResultSha256: "2".repeat(64),
    artifactIndexSha256: digest(indexContent),
    manifestSha256: "3".repeat(64),
  };
  await writeFile(resolve(root, "manifest.json"), `${JSON.stringify(manifest)}\n`);
  return { exitCode: 0, exitSignal: null, timedOut: false, stdout: "ok", stderr: "" };
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
    signal.addEventListener("abort", () => resolvePromise({ exitCode: null, exitSignal: "SIGTERM", timedOut: false, stdout: "", stderr: "" }), { once: true });
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
