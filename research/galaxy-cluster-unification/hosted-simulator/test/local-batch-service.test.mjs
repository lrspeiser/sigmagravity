import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { mkdir, mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { basename, resolve } from "node:path";
import test from "node:test";
import { sha256 } from "../lib/canonical.mjs";
import { LocalBatchService } from "../lib/local-batch-service.mjs";
import { LocalFieldJobService, LocalServiceError } from "../lib/local-field-job-service.mjs";

function digest(value) {
  return createHash("sha256").update(value).digest("hex");
}

function model() {
  return {
    schemaVersion: "sigma-field-model/1",
    name: "Batch manufactured model",
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
    parameterPolicy: { mode: "published_fixed", perObjectParameters: [] },
  };
}

function bundle(label) {
  const core = {
    schemaVersion: "sigma-array-bundle/1",
    geometry: { coordinateSystem: "cartesian_2d", dimensions: 2, spacing: [0.1, 0.1], lengthUnit: "m" },
    arrays: [{
      key: "forcing", npzKey: "forcing", unit: "1/s^2", rank: "scalar", role: "source",
      dtype: "<f8", shape: [17, 17], elementCount: 289, contentSha256: sha256({ label }),
    }],
    provenance: { kind: "batch_service_fixture", label },
    license: { id: "CC0-1.0", redistributionAllowed: true },
  };
  return { ...core, bundleSha256: sha256(core) };
}

function velocityBundle(label) {
  const base = bundle(label);
  const core = Object.fromEntries(Object.entries(base).filter(([key]) => key !== "bundleSha256"));
  const records = [
    ["major", "m", "auxiliary"],
    ["minor", "m", "auxiliary"],
    ["observed_velocity", "m/s", "auxiliary"],
    ["velocity_uncertainty", "m/s", "uncertainty"],
  ].map(([key, unit, role], index) => ({
    key, npzKey: key, unit, rank: "scalar", role, dtype: "<f8",
    shape: [17, 17], elementCount: 289, contentSha256: String(index + 2).repeat(64),
  }));
  core.arrays.push(...records);
  return { ...core, bundleSha256: sha256(core) };
}

async function successfulRunner({ jobDirectory }) {
  const root = resolve(jobDirectory, "artifacts");
  await mkdir(root, { recursive: true });
  const childId = basename(jobDirectory);
  const requestEnvelope = JSON.parse(await readFile(resolve(jobDirectory, "request.json"), "utf8"));
  const observationTargets = requestEnvelope.request.observationTargets ?? [];
  const targetKinds = [...new Set(observationTargets.map((target) => target.kind))].sort();
  const observationEvaluation = observationTargets.length ? {
    schemaVersion: "sigma-observation-evaluation/1",
    targetKinds,
    targetCount: 1,
    scoredTargetCount: 1,
    totalPoints: 1,
    validScoredPoints: 1,
    sumSquaredResidualM2PerS2: 4,
    rmseMPerS: 2,
    inverseVarianceWeightedSquaredResidual: 1,
    inverseVarianceWeightSum: 0.25,
    inverseVarianceWeightedRmseMPerS: 2,
    chiSquare: 1,
    degreesFreedom: 1,
    reducedChiSquare: 1,
    targets: [],
  } : null;
  const scientific = {
    schemaVersion: "sigma-field-result/1",
    jobId: `fieldjob_${childId.slice(-12)}`,
    state: "succeeded",
    converged: true,
    iterations: 4,
    maximumRelativeUpdate: 1e-9,
    equationResiduals: { manufactured: { relativeL2: 2e-8 } },
    observationEvaluation,
    parameterAccounting: { universal: 0, perObject: 0 },
  };
  const scientificContent = Buffer.from(`${JSON.stringify(scientific)}\n`);
  await writeFile(resolve(root, "scientific_result.json"), scientificContent);
  const predictionContent = Buffer.from(
    "target_id,point_index,radius_m,predicted_speed_m_s,observed_speed_m_s,uncertainty_m_s,residual_m_s,azimuthal_coverage,mean_inward_acceleration_m_s2\nfixture,0,1,8,10,2,-2,1,64\n",
  );
  const velocityPredictionContent = Buffer.from(
    "target_id,point_index,row_index,column_index,disk_major_coordinate_m,disk_minor_coordinate_m,circular_radius_m,predicted_circular_speed_m_s,predicted_velocity_m_s,observed_velocity_m_s,uncertainty_m_s,residual_m_s,declared_weight,inward_acceleration_m_s2\nfixture-map,0,0,0,1,0,1,8,8,10,2,-2,0.25,64\n",
  );
  if (targetKinds.includes("circular_speed_curve")) {
    await writeFile(resolve(root, "observation_predictions.csv"), predictionContent);
  }
  if (targetKinds.includes("line_of_sight_velocity_field")) {
    await writeFile(resolve(root, "observation_velocity_field_predictions.csv"), velocityPredictionContent);
  }
  const artifactRecords = [
    { path: "scientific_result.json", bytes: scientificContent.length, sha256: digest(scientificContent) },
  ];
  if (targetKinds.includes("circular_speed_curve")) {
    artifactRecords.push({
      path: "observation_predictions.csv",
      bytes: predictionContent.length,
      sha256: digest(predictionContent),
    });
  }
  if (targetKinds.includes("line_of_sight_velocity_field")) {
    artifactRecords.push({
      path: "observation_velocity_field_predictions.csv",
      bytes: velocityPredictionContent.length,
      sha256: digest(velocityPredictionContent),
    });
  }
  const artifactIndex = {
    schemaVersion: "sigma-field-artifact-index/1",
    jobId: scientific.jobId,
    artifacts: artifactRecords,
  };
  const indexContent = Buffer.from(`${JSON.stringify(artifactIndex)}\n`);
  await writeFile(resolve(root, "artifact_index.json"), indexContent);
  const manifest = {
    schemaVersion: "sigma-field-run-manifest/1",
    state: "succeeded",
    jobId: scientific.jobId,
    scientificResultSha256: sha256(scientific),
    artifactIndexSha256: digest(indexContent),
    manifestSha256: sha256({ childId }),
  };
  await writeFile(resolve(root, "manifest.json"), `${JSON.stringify(manifest)}\n`);
  return { exitCode: 0, exitSignal: null, timedOut: false, stdout: "ok", stderr: "" };
}

async function fixture(t) {
  const root = await mkdtemp(resolve(tmpdir(), "sigma-batch-service-"));
  t.after(async () => rm(root, { recursive: true, force: true }));
  const projectRoot = resolve(import.meta.dirname, "..", "..");
  const fieldService = new LocalFieldJobService({ root, projectRoot, runner: successfulRunner });
  await fieldService.initialize();
  const batchService = new LocalBatchService({ root, fieldService, pollMilliseconds: 5 });
  await batchService.initialize();
  t.after(async () => {
    await batchService.close();
    await fieldService.close();
  });
  return { fieldService, batchService };
}

async function upload(fieldService, label, inputBundle = bundle(label)) {
  const archive = Buffer.from(`npz-${label}`);
  const ticket = await fieldService.createUpload({
    schemaVersion: "sigma-data-upload-request/1",
    inputBundle,
    archive: { sha256: digest(archive), bytes: archive.length },
  });
  await fieldService.putUploadContent(ticket.id, archive);
  return ticket;
}

test("one fixed model produces deterministic multi-system batch reports", async (t) => {
  const { fieldService, batchService } = await fixture(t);
  const [first, second] = await Promise.all([upload(fieldService, "A"), upload(fieldService, "B")]);
  const payload = {
    schemaVersion: "sigma-batch-submit/1",
    model: model(),
    systems: [
      { id: "GALAXY-A", dataUploadId: first.id },
      { id: "GALAXY-B", dataUploadId: second.id },
    ],
    fieldRequest: {
      schemaVersion: "sigma-field-job-request/1",
      boundaryFields: { u: { value: 0 } },
      requestedObservables: ["gradient"],
      seed: 9,
    },
    parameterPolicy: { mode: "published_fixed", perObjectParameters: [] },
  };
  const submission = await batchService.createBatch(payload);
  assert.match(submission.id, /^batch_[0-9a-f]{24}$/);
  assert.equal(submission.systemCount, 2);
  await fieldService.waitForIdle();
  await batchService.waitForIdle();
  const completed = await batchService.getBatch(submission.id);
  assert.equal(completed.state, "succeeded");
  assert.equal(completed.successfulChildren, 2);
  const response = await batchService.getArtifacts(submission.id);
  assert.deepEqual(
    response.items.map((item) => item.path).sort(),
    [
      "aggregate_scores.json", "batch.json", "child_jobs.json", "failures.csv",
      "llm_briefing.md", "model.json", "observation_predictions.csv",
      "observation_velocity_field_predictions.csv", "per_galaxy.csv", "report.html",
      "reproduction_command.txt",
    ],
  );
  const aggregate = JSON.parse((await batchService.getArtifact(submission.id, "aggregate_scores.json")).content.toString("utf8"));
  assert.equal(aggregate.systemCount, 2);
  assert.equal(aggregate.convergenceFraction, 1);
  assert.equal(aggregate.parameterPolicy.mode, "published_fixed");
  assert.equal(aggregate.perObjectGravityParameters, 0);
  assert.equal(aggregate.observationScoresAvailable, false);
  const report = (await batchService.getArtifact(submission.id, "report.html")).content.toString("utf8");
  assert.match(report, /numerical execution and convergence only/);
  const duplicate = await batchService.createBatch(payload);
  assert.equal(duplicate.id, submission.id);
  assert.equal(duplicate.duplicate, true);
});

test("unsupported fitting policies are rejected before child execution", async (t) => {
  const { fieldService, batchService } = await fixture(t);
  const source = await upload(fieldService, "FIT");
  const fittedModel = model();
  fittedModel.parameterPolicy.mode = "universal_fit";
  await assert.rejects(
    () => batchService.createBatch({
      schemaVersion: "sigma-batch-submit/1",
      model: fittedModel,
      systems: [{ id: "GALAXY-FIT", dataUploadId: source.id }],
      fieldRequest: { schemaVersion: "sigma-field-job-request/1", requestedObservables: ["gradient"] },
      parameterPolicy: { mode: "universal_fit", perObjectParameters: [] },
    }),
    (error) => error instanceof LocalServiceError && error.code === "parameter_policy_not_executable",
  );
  assert.equal((await fieldService.listFieldJobs()).items.length, 0);
});

test("batch aggregates post-solve circular-speed scores and predictions", async (t) => {
  const { fieldService, batchService } = await fixture(t);
  const source = await upload(fieldService, "OBSERVATION");
  const observationModel = model();
  observationModel.observables[0].target = "massive_tracers";
  const target = {
    schemaVersion: "sigma-observation-target/1",
    id: "GALAXY-O-rotation",
    kind: "circular_speed_curve",
    observable: "gradient",
    gridOriginM: [0, 0],
    centerM: [0, 0],
    radiiM: [1],
    observedSpeedsMPerS: [10],
    uncertaintiesMPerS: [2],
    provenance: { kind: "test fixture" },
    license: { id: "CC0-1.0", redistributionAllowed: true },
  };
  const submission = await batchService.createBatch({
    schemaVersion: "sigma-batch-submit/1",
    model: observationModel,
    systems: [{ id: "GALAXY-O", dataUploadId: source.id, observationTargets: [target] }],
    fieldRequest: { schemaVersion: "sigma-field-job-request/1", requestedObservables: ["gradient"] },
    parameterPolicy: { mode: "published_fixed", perObjectParameters: [] },
  });
  await fieldService.waitForIdle();
  await batchService.waitForIdle();
  const aggregate = JSON.parse((await batchService.getArtifact(submission.id, "aggregate_scores.json")).content.toString("utf8"));
  assert.equal(aggregate.observationScoresAvailable, true);
  assert.equal(aggregate.scoredObservationTargets, 1);
  assert.equal(aggregate.validObservationPoints, 1);
  assert.equal(aggregate.observationRmseMPerS, 2);
  const predictions = (await batchService.getArtifact(submission.id, "observation_predictions.csv")).content.toString("utf8");
  assert.match(predictions, /GALAXY-O,fixture,0,1,8,10,2,-2,1,64/);
});

test("batch aggregates resolved velocity-field scores and prediction pixels", async (t) => {
  const { fieldService, batchService } = await fixture(t);
  const source = await upload(fieldService, "VELOCITY", velocityBundle("VELOCITY"));
  const observationModel = model();
  observationModel.observables[0].target = "massive_tracers";
  const target = {
    schemaVersion: "sigma-observation-target/1",
    id: "GALAXY-V-map",
    kind: "line_of_sight_velocity_field",
    observable: "gradient",
    centerM: [0, 0],
    inclinationDeg: 45,
    handedness: 1,
    majorCoordinateArrayKey: "major",
    minorCoordinateArrayKey: "minor",
    observedVelocityArrayKey: "observed_velocity",
    uncertaintyArrayKey: "velocity_uncertainty",
    minimumValidPixels: 100,
    provenance: { kind: "test velocity map" },
    license: { id: "CC0-1.0", redistributionAllowed: true },
  };
  const submission = await batchService.createBatch({
    schemaVersion: "sigma-batch-submit/1",
    model: observationModel,
    systems: [{ id: "GALAXY-V", dataUploadId: source.id, observationTargets: [target] }],
    fieldRequest: { schemaVersion: "sigma-field-job-request/1", requestedObservables: ["gradient"] },
    parameterPolicy: { mode: "published_fixed", perObjectParameters: [] },
  });
  await fieldService.waitForIdle();
  await batchService.waitForIdle();
  const aggregate = JSON.parse((await batchService.getArtifact(submission.id, "aggregate_scores.json")).content.toString("utf8"));
  assert.equal(aggregate.observationScoresAvailable, true);
  assert.equal(aggregate.scoredObservationTargets, 1);
  const predictions = (await batchService.getArtifact(
    submission.id,
    "observation_velocity_field_predictions.csv",
  )).content.toString("utf8");
  assert.match(predictions, /GALAXY-V,fixture-map,0,0,0,1,0,1,8,8,10,2,-2,0.25,64/);
  const report = (await batchService.getArtifact(submission.id, "report.html")).content.toString("utf8");
  assert.match(report, /line_of_sight_velocity_field/);
});

test("batch report artifacts reject traversal and mutation", async (t) => {
  const { fieldService, batchService } = await fixture(t);
  const source = await upload(fieldService, "INTEGRITY");
  const submission = await batchService.createBatch({
    schemaVersion: "sigma-batch-submit/1",
    model: model(),
    systems: [{ id: "GALAXY-I", dataUploadId: source.id }],
    fieldRequest: { schemaVersion: "sigma-field-job-request/1", requestedObservables: ["gradient"] },
    parameterPolicy: { mode: "published_fixed", perObjectParameters: [] },
  });
  await fieldService.waitForIdle();
  await batchService.waitForIdle();
  await assert.rejects(() => batchService.getArtifact(submission.id, "..%2Frecord.json"), /unknown batch artifact/);
  const artifactPath = resolve(batchService.root, "batches", submission.id, "artifacts", "report.html");
  await writeFile(artifactPath, "mutated");
  await assert.rejects(
    () => batchService.getArtifact(submission.id, "report.html"),
    (error) => error instanceof LocalServiceError && error.code === "artifact_integrity_failed",
  );
  assert.ok((await readFile(resolve(batchService.root, "batches", submission.id, "record.json"))).length > 0);
});
