import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { mkdir, mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { basename, resolve } from "node:path";
import test from "node:test";
import { sha256 } from "../lib/canonical.mjs";
import { validateFieldModel } from "../lib/field-model.mjs";
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

function velocityObservationBundle(label) {
  const source = velocityBundle(label);
  const core = {
    ...Object.fromEntries(Object.entries(source).filter(([key]) => key !== "bundleSha256")),
    arrays: source.arrays.filter((record) => record.key !== "forcing"),
    provenance: { kind: "batch_observation_fixture", label },
  };
  return { ...core, bundleSha256: sha256(core) };
}

function photonModel() {
  const result = model();
  result.name = "Batch 3D photon model";
  result.geometry = {
    coordinateSystem: "cartesian_3d",
    dimensions: 3,
    domain: { lengthUnit: "m", boundaryExtent: "fixture cube" },
  };
  result.observables[0].target = "photons";
  return result;
}

function photonBundle(label, { observations = false } = {}) {
  const core = {
    schemaVersion: "sigma-array-bundle/1",
    geometry: { coordinateSystem: "cartesian_3d", dimensions: 3, spacing: [0.1, 0.1, 0.1], lengthUnit: "m" },
    arrays: observations
      ? [
        ["alpha_east", "arcsec"], ["alpha_north", "arcsec"], ["alpha_sigma", "arcsec"],
        ["reduced_g1", "1"], ["reduced_g2", "1"], ["reduced_g_sigma", "1"], ["score_mask", "1"],
      ].map(([key, unit], index) => ({
        key, npzKey: key, unit, rank: "scalar", role: "observation", dtype: "<f8",
        shape: [17, 17], elementCount: 289, contentSha256: String(index + 2).repeat(64),
      }))
      : [{
        key: "forcing", npzKey: "forcing", unit: "1/s^2", rank: "scalar", role: "source",
        dtype: "<f8", shape: [17, 17, 17], elementCount: 4913, contentSha256: sha256({ label }),
      }],
    provenance: { kind: "batch_photon_fixture", label },
    license: { id: "CC0-1.0", redistributionAllowed: true },
  };
  return { ...core, bundleSha256: sha256(core) };
}

async function successfulFieldRunner({ jobDirectory }) {
  const root = resolve(jobDirectory, "artifacts");
  await mkdir(root, { recursive: true });
  const childId = basename(jobDirectory);
  const fieldModel = JSON.parse(await readFile(resolve(jobDirectory, "model.json"), "utf8"));
  const validation = validateFieldModel(fieldModel);
  const dimensions = fieldModel.geometry.dimensions;
  const shape = Array(dimensions).fill(17);
  const fieldJobSha256 = sha256({ childId, modelSha256: validation.modelSha256 });
  const fieldJob = {
    schemaVersion: "sigma-field-job/1",
    id: `fieldjob_${fieldJobSha256.slice(0, 24)}`,
    jobSha256: fieldJobSha256,
    modelSha256: validation.modelSha256,
    geometry: {
      coordinateSystem: fieldModel.geometry.coordinateSystem, dimensions,
      spacing: Array(dimensions).fill(0.1), origin: Array(dimensions).fill(-0.8), lengthUnit: "m",
    },
  };
  const scientificCore = {
    schemaVersion: "sigma-field-result/1",
    jobId: fieldJob.id,
    jobSha256: fieldJob.jobSha256,
    state: "succeeded",
    converged: true,
    iterations: 4,
    maximumRelativeUpdate: 1e-9,
    equationResiduals: { manufactured: { relativeL2: 2e-8 } },
    observables: Array.from({ length: dimensions }, (_, axis) => ({
      key: `gradient__axis${axis}`, dtype: "<f8", shape, contentSha256: String(axis + 6).repeat(64),
    })),
    parameterAccounting: { universal: 0, perObject: 0 },
  };
  const scientific = { ...scientificCore, resultSha256: sha256(scientificCore) };
  const contents = new Map([
    ["model.json", Buffer.from(`${JSON.stringify(fieldModel)}\n`)],
    ["job.json", Buffer.from(`${JSON.stringify(fieldJob)}\n`)],
    ["scientific_result.json", Buffer.from(`${JSON.stringify(scientific)}\n`)],
    ["observables.npz", Buffer.from("fixture-observable-archive")],
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
    scientificResultSha256: scientific.resultSha256,
    artifactIndexSha256: digest(indexContent),
  };
  await writeFile(
    resolve(root, "manifest.json"),
    `${JSON.stringify({ ...manifestCore, manifestSha256: sha256(manifestCore) })}\n`,
  );
  return { exitCode: 0, exitSignal: null, timedOut: false, stdout: "ok", stderr: "" };
}

async function successfulObservationRunner({ jobDirectory }) {
  const root = resolve(jobDirectory, "artifacts");
  await mkdir(root, { recursive: true });
  const childId = basename(jobDirectory);
  const requestEnvelope = JSON.parse(await readFile(resolve(jobDirectory, "request.json"), "utf8"));
  const observationTargets = requestEnvelope.request.observationTargets ?? [];
  const targetKinds = [...new Set(observationTargets.map((target) => target.kind))].sort();
  const photonScored = targetKinds.includes("photon_lensing_map");
  const rawImageScored = targetKinds.includes("multiple_image_systems");
  const nonVelocityScored = photonScored || rawImageScored;
  const channelAggregates = rawImageScored
    ? {
      image_position_arcsec: {
        channel: "image_position_arcsec", unit: "arcsec", scoredTargetCount: 1, validPoints: 4,
        fittedNuisanceParameters: 2, sumSquaredResidual: 0.04,
        rmse: 0.1, inverseVarianceWeightedSquaredResidual: 4,
        inverseVarianceWeightSum: 400, inverseVarianceWeightedRmse: 0.1,
        chiSquare: 4, degreesFreedom: 2, reducedChiSquare: 2, gaussianLogLikelihood: 4,
      },
    }
    : photonScored
    ? {
      deflection_arcsec: {
        channel: "deflection_arcsec", unit: "arcsec", scoredTargetCount: 1, validPoints: 4,
        fittedNuisanceParameters: 0, sumSquaredResidual: 16,
        rmse: 2, inverseVarianceWeightedSquaredResidual: 4,
        inverseVarianceWeightSum: 1, inverseVarianceWeightedRmse: 2,
        chiSquare: 4, degreesFreedom: 4, reducedChiSquare: 1, gaussianLogLikelihood: -8,
      },
      reduced_shear_dimensionless: {
        channel: "reduced_shear_dimensionless", unit: "1", scoredTargetCount: 1, validPoints: 4,
        fittedNuisanceParameters: 0, sumSquaredResidual: 0.04,
        rmse: 0.1, inverseVarianceWeightedSquaredResidual: 4,
        inverseVarianceWeightSum: 400, inverseVarianceWeightedRmse: 0.1,
        chiSquare: 4, degreesFreedom: 4, reducedChiSquare: 1, gaussianLogLikelihood: 4,
      },
    }
    : {
      velocity_m_s: {
        channel: "velocity_m_s", unit: "m/s", scoredTargetCount: observationTargets.length,
        validPoints: 1, fittedNuisanceParameters: 0, sumSquaredResidual: 4, rmse: 2,
        inverseVarianceWeightedSquaredResidual: 1, inverseVarianceWeightSum: 0.25,
        inverseVarianceWeightedRmse: 2, chiSquare: 1, degreesFreedom: 1,
        reducedChiSquare: 1, gaussianLogLikelihood: -2,
      },
    };
  const observationEvaluation = {
    schemaVersion: "sigma-observation-evaluation/1",
    targetKinds,
    targetCount: observationTargets.length,
    scoredTargetCount: observationTargets.length,
    totalPoints: photonScored ? 8 : rawImageScored ? 4 : 1,
    validScoredPoints: photonScored ? 8 : rawImageScored ? 4 : 1,
    channelAggregates,
    sumSquaredResidualM2PerS2: nonVelocityScored ? null : 4,
    rmseMPerS: nonVelocityScored ? null : 2,
    inverseVarianceWeightedSquaredResidual: nonVelocityScored ? null : 1,
    inverseVarianceWeightSum: nonVelocityScored ? null : 0.25,
    inverseVarianceWeightedRmseMPerS: nonVelocityScored ? null : 2,
    chiSquare: nonVelocityScored ? null : 1,
    degreesFreedom: nonVelocityScored ? null : 1,
    reducedChiSquare: nonVelocityScored ? null : 1,
    targets: rawImageScored ? [{ state: "scored" }] : [],
  };
  const scientificCore = {
    schemaVersion: "sigma-observation-evaluation-result/1",
    jobId: `observationjob_${childId.slice(-12)}`,
    state: "succeeded",
    observationEvaluation,
    parameterAccounting: { universal: 0, perObject: 0 },
    evaluationAddedGravityParameters: 0,
  };
  const scientific = { ...scientificCore, resultSha256: sha256(scientificCore) };
  const scientificContent = Buffer.from(`${JSON.stringify(scientific)}\n`);
  const scoreContent = Buffer.from(`${JSON.stringify(observationEvaluation)}\n`);
  await writeFile(resolve(root, "scientific_result.json"), scientificContent);
  await writeFile(resolve(root, "observation_scores.json"), scoreContent);
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
  const photonMapContent = Buffer.from("deterministic-photon-map-fixture");
  if (photonScored) await writeFile(resolve(root, "observation_photon_lensing_maps.npz"), photonMapContent);
  const rawPredictionContent = Buffer.from(
    "target_id,family_id,family_index,image_index,assignment_state,observed_east_arcsec,observed_north_arcsec,position_uncertainty_arcsec,predicted_root_index,predicted_east_arcsec,predicted_north_arcsec,residual_east_arcsec,residual_north_arcsec,separation_arcsec,root_closure_arcsec,root_absolute_magnification\nraw,source-a,0,0,matched,-1,0,0.05,0,-0.9,0,0.1,0,0.1,0.00001,2\n",
  );
  const rawFamilyContent = Buffer.from(
    "target_id,family_id,family_index,distance_ratio,profiled_source_east_arcsec,profiled_source_north_arcsec,observed_images,predicted_roots,matched_images,complete_observed_assignment,excess_predicted_roots,critical_curve_points,state,image_plane_rms_arcsec,matched_subset_diagnostic_rms_arcsec,chi_square,degrees_freedom,fitted_observation_nuisance_parameters,gravity_parameters_added\nraw,source-a,0,0.7,0.2,0,2,2,2,True,0,0,scored,0.1,,4,2,2,0\n",
  );
  const rawRootContent = Buffer.from("deterministic-raw-root-fixture");
  if (rawImageScored) {
    await writeFile(resolve(root, "observation_multiple_image_predictions.csv"), rawPredictionContent);
    await writeFile(resolve(root, "observation_multiple_image_families.csv"), rawFamilyContent);
    await writeFile(resolve(root, "observation_multiple_image_roots.npz"), rawRootContent);
  }
  const artifactRecords = [
    { path: "scientific_result.json", bytes: scientificContent.length, sha256: digest(scientificContent) },
    { path: "observation_scores.json", bytes: scoreContent.length, sha256: digest(scoreContent) },
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
  if (photonScored) {
    artifactRecords.push({
      path: "observation_photon_lensing_maps.npz",
      bytes: photonMapContent.length,
      sha256: digest(photonMapContent),
    });
  }
  if (rawImageScored) {
    for (const [path, content] of [
      ["observation_multiple_image_predictions.csv", rawPredictionContent],
      ["observation_multiple_image_families.csv", rawFamilyContent],
      ["observation_multiple_image_roots.npz", rawRootContent],
    ]) artifactRecords.push({ path, bytes: content.length, sha256: digest(content) });
  }
  const artifactIndex = {
    schemaVersion: "sigma-observation-evaluation-artifact-index/1",
    jobId: scientific.jobId,
    artifacts: artifactRecords,
  };
  const indexContent = Buffer.from(`${JSON.stringify(artifactIndex)}\n`);
  await writeFile(resolve(root, "artifact_index.json"), indexContent);
  const manifestCore = {
    schemaVersion: "sigma-observation-evaluation-run-manifest/1",
    state: "succeeded",
    jobId: scientific.jobId,
    scientificResultSha256: scientific.resultSha256,
    artifactIndexSha256: digest(indexContent),
  };
  await writeFile(
    resolve(root, "manifest.json"),
    `${JSON.stringify({ ...manifestCore, manifestSha256: sha256(manifestCore) })}\n`,
  );
  return { exitCode: 0, exitSignal: null, timedOut: false, stdout: "ok", stderr: "" };
}

async function successfulRunner(argumentsValue) {
  return argumentsValue.jobType === "observation_evaluation"
    ? successfulObservationRunner(argumentsValue)
    : successfulFieldRunner(argumentsValue);
}

async function fixture(t, options = {}) {
  const root = await mkdtemp(resolve(tmpdir(), "sigma-batch-service-"));
  t.after(async () => rm(root, { recursive: true, force: true }));
  const projectRoot = resolve(import.meta.dirname, "..", "..");
  const fieldService = new LocalFieldJobService({
    root,
    projectRoot,
    runner: options.runner ?? successfulRunner,
  });
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
  assert.equal(completed.phase, "complete");
  assert.equal(completed.successfulChildren, 2);
  const response = await batchService.getArtifacts(submission.id);
  assert.deepEqual(
    response.items.map((item) => item.path).sort(),
    [
      "aggregate_scores.json", "batch.json", "child_jobs.json", "failures.csv",
      "llm_briefing.md", "model.json", "observation_multiple_image_families.csv",
      "observation_multiple_image_predictions.csv", "observation_predictions.csv",
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
  const children = JSON.parse((await batchService.getArtifact(
    submission.id,
    "child_jobs.json",
  )).content.toString("utf8")).items;
  assert.ok(children.every((child) => child.observationEvaluationJobId === null));
  assert.equal((await fieldService.listObservationEvaluationJobs()).items.length, 0);
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
  const failureDetails = (await batchService.getArtifact(submission.id, "failures.csv")).content.toString("utf8");
  assert.equal(aggregate.observationScoresAvailable, true, failureDetails);
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
  assert.equal(aggregate.observationScoresAvailable, true, JSON.stringify(aggregate));
  assert.equal(aggregate.scoredObservationTargets, 1);
  const predictions = (await batchService.getArtifact(
    submission.id,
    "observation_velocity_field_predictions.csv",
  )).content.toString("utf8");
  assert.match(predictions, /GALAXY-V,fixture-map,0,0,0,1,0,1,8,8,10,2,-2,0.25,64/);
  const report = (await batchService.getArtifact(submission.id, "report.html")).content.toString("utf8");
  assert.match(report, /line_of_sight_velocity_field/);
});

test("batch retains photon deflection and reduced-shear channels without a velocity alias", async (t) => {
  const { fieldService, batchService } = await fixture(t);
  const fieldUpload = await upload(fieldService, "PHOTON-FIELD", photonBundle("PHOTON-FIELD"));
  const observationUpload = await upload(
    fieldService,
    "PHOTON-OBSERVATION",
    photonBundle("PHOTON-OBSERVATION", { observations: true }),
  );
  const photonTarget = {
    schemaVersion: "sigma-observation-target/1",
    id: "CLUSTER-PHOTON-MAP",
    kind: "photon_lensing_map",
    observable: "gradient",
    northAxis: 0,
    eastAxis: 1,
    lineOfSightAxis: 2,
    distanceRatio: 0.72,
    lensAngularDiameterDistanceM: 3.0e25,
    observedAlphaEastArcsecArrayKey: "alpha_east",
    observedAlphaNorthArcsecArrayKey: "alpha_north",
    deflectionUncertaintyArcsecArrayKey: "alpha_sigma",
    observedReducedShear1ArrayKey: "reduced_g1",
    observedReducedShear2ArrayKey: "reduced_g2",
    reducedShearUncertaintyArrayKey: "reduced_g_sigma",
    scoreMaskArrayKey: "score_mask",
    minimumValidPixels: 25,
    provenance: { kind: "typed photon batch fixture" },
    license: { id: "CC0-1.0", redistributionAllowed: true },
  };
  const submission = await batchService.createBatch({
    schemaVersion: "sigma-batch-submit/1",
    model: photonModel(),
    systems: [{
      id: "CLUSTER-P",
      dataUploadId: fieldUpload.id,
      observationDataUploadId: observationUpload.id,
      observationTargets: [photonTarget],
    }],
    fieldRequest: { schemaVersion: "sigma-field-job-request/1", requestedObservables: ["gradient"] },
    parameterPolicy: { mode: "published_fixed", perObjectParameters: [] },
  });
  await fieldService.waitForIdle();
  await batchService.waitForIdle();
  const aggregate = JSON.parse((await batchService.getArtifact(
    submission.id,
    "aggregate_scores.json",
  )).content.toString("utf8"));
  assert.equal(aggregate.observationRmseMPerS, null);
  assert.equal(aggregate.observationChannelAggregates.deflection_arcsec.rmse, 2);
  assert.equal(aggregate.observationChannelAggregates.reduced_shear_dimensionless.rmse, 0.1);
  assert.equal(aggregate.validObservationPoints, 8);
  const children = JSON.parse((await batchService.getArtifact(
    submission.id,
    "child_jobs.json",
  )).content.toString("utf8")).items;
  assert.ok(children[0].observationArtifacts.some((item) => item.path === "observation_photon_lensing_maps.npz"));
  const report = (await batchService.getArtifact(submission.id, "report.html")).content.toString("utf8");
  assert.match(report, /photon_lensing_map/);
  assert.match(report, /Deflection RMSE \(arcsec\)/);
});

test("batch retains raw image-position scores and family topology artifacts", async (t) => {
  const { fieldService, batchService } = await fixture(t);
  const fieldUpload = await upload(fieldService, "RAW-IMAGE-FIELD", photonBundle("RAW-IMAGE-FIELD"));
  const rawTarget = {
    schemaVersion: "sigma-observation-target/1",
    id: "CLUSTER-RAW-IMAGES",
    kind: "multiple_image_systems",
    observable: "gradient",
    northAxis: 0,
    eastAxis: 1,
    lineOfSightAxis: 2,
    lensAngularDiameterDistanceM: 1.0e3,
    skyCenterM: [0, 0, 0],
    rootSearchBoundArcsec: 10,
    families: [{
      id: "source-a",
      distanceRatio: 0.7,
      observedImagesArcsec: [[-1, 0], [1, 0]],
      positionUncertaintiesArcsec: [0.05, 0.05],
    }],
    provenance: { kind: "raw multiple-image batch fixture" },
    license: { id: "CC0-1.0", redistributionAllowed: true },
  };
  const submission = await batchService.createBatch({
    schemaVersion: "sigma-batch-submit/1",
    model: photonModel(),
    systems: [{
      id: "CLUSTER-RAW",
      dataUploadId: fieldUpload.id,
      observationTargets: [rawTarget],
    }],
    fieldRequest: { schemaVersion: "sigma-field-job-request/1", requestedObservables: ["gradient"] },
    parameterPolicy: { mode: "published_fixed", perObjectParameters: [] },
  });
  await fieldService.waitForIdle();
  await batchService.waitForIdle();
  const aggregate = JSON.parse((await batchService.getArtifact(
    submission.id,
    "aggregate_scores.json",
  )).content.toString("utf8"));
  assert.equal(aggregate.observationRmseMPerS, null);
  assert.equal(aggregate.observationChannelAggregates.image_position_arcsec.rmse, 0.1);
  assert.equal(aggregate.observationChannelAggregates.image_position_arcsec.fittedNuisanceParameters, 2);
  const predictions = (await batchService.getArtifact(
    submission.id,
    "observation_multiple_image_predictions.csv",
  )).content.toString("utf8");
  assert.match(predictions, /CLUSTER-RAW,raw,source-a,0,0,matched/);
  const families = (await batchService.getArtifact(
    submission.id,
    "observation_multiple_image_families.csv",
  )).content.toString("utf8");
  assert.match(families, /CLUSTER-RAW,raw,source-a,0,0.7/);
  const report = (await batchService.getArtifact(submission.id, "report.html")).content.toString("utf8");
  assert.match(report, /multiple_image_systems/);
  assert.match(report, /Image-position coordinate RMSE \(arcsec\)/);
});

test("changed observation data reuses the field child and changes only the observation child", async (t) => {
  const calls = { field: 0, observation: 0 };
  const runner = async (argumentsValue) => {
    if (argumentsValue.jobType === "observation_evaluation") {
      calls.observation += 1;
      return successfulObservationRunner(argumentsValue);
    }
    calls.field += 1;
    return successfulFieldRunner(argumentsValue);
  };
  const { fieldService, batchService } = await fixture(t, { runner });
  const fieldUpload = await upload(fieldService, "SHARED-FIELD");
  const firstObservationUpload = await upload(
    fieldService,
    "OBSERVATION-A",
    velocityObservationBundle("OBSERVATION-A"),
  );
  const secondObservationUpload = await upload(
    fieldService,
    "OBSERVATION-B",
    velocityObservationBundle("OBSERVATION-B"),
  );
  const observationModel = model();
  observationModel.observables[0].target = "massive_tracers";
  const target = {
    schemaVersion: "sigma-observation-target/1",
    id: "GALAXY-COMPOSED-map",
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
    provenance: { kind: "composed batch fixture" },
    license: { id: "CC0-1.0", redistributionAllowed: true },
  };
  const payload = (observationDataUploadId) => ({
    schemaVersion: "sigma-batch-submit/1",
    model: observationModel,
    systems: [{
      id: "GALAXY-COMPOSED",
      dataUploadId: fieldUpload.id,
      observationDataUploadId,
      observationTargets: [target],
    }],
    fieldRequest: { schemaVersion: "sigma-field-job-request/1", requestedObservables: ["gradient"] },
    parameterPolicy: { mode: "published_fixed", perObjectParameters: [] },
  });

  const first = await batchService.createBatch(payload(firstObservationUpload.id));
  await batchService.waitForIdle();
  const firstChildren = JSON.parse((await batchService.getArtifact(
    first.id,
    "child_jobs.json",
  )).content.toString("utf8")).items;
  const second = await batchService.createBatch(payload(secondObservationUpload.id));
  await batchService.waitForIdle();
  const secondChildren = JSON.parse((await batchService.getArtifact(
    second.id,
    "child_jobs.json",
  )).content.toString("utf8")).items;

  assert.notEqual(first.id, second.id);
  assert.equal(firstChildren[0].fieldJobId, secondChildren[0].fieldJobId);
  assert.notEqual(
    firstChildren[0].observationEvaluationJobId,
    secondChildren[0].observationEvaluationJobId,
  );
  assert.equal(calls.field, 1);
  assert.equal(calls.observation, 2);
  const fieldEnvelope = JSON.parse(await readFile(resolve(
    fieldService.root,
    "jobs",
    firstChildren[0].fieldJobId,
    "request.json",
  ), "utf8"));
  assert.equal((fieldEnvelope.request.observationTargets ?? []).length, 0);
  const duplicate = await batchService.createBatch(payload(secondObservationUpload.id));
  assert.equal(duplicate.id, second.id);
  assert.equal(duplicate.duplicate, true);
  assert.equal(calls.field, 1);
  assert.equal(calls.observation, 2);

  const observationId = secondChildren[0].observationEvaluationJobId;
  const standalonePrediction = await fieldService.getArtifact(
    observationId,
    "observation_velocity_field_predictions.csv",
  );
  const recordedPrediction = secondChildren[0].observationArtifacts.find(
    (artifact) => artifact.path === "observation_velocity_field_predictions.csv",
  );
  assert.equal(recordedPrediction.sha256, standalonePrediction.record.sha256);
  const recordedScores = secondChildren[0].observationArtifacts.find(
    (artifact) => artifact.path === "observation_scores.json",
  );
  const standaloneScores = await fieldService.getArtifact(observationId, "observation_scores.json");
  assert.equal(recordedScores.sha256, standaloneScores.record.sha256);
  const standaloneLines = standalonePrediction.content.toString("utf8").trimEnd().split("\n");
  const expectedAggregatePrediction = `system_id,${standaloneLines[0]}\n${standaloneLines
    .slice(1)
    .map((line) => `GALAXY-COMPOSED,${line}`)
    .join("\n")}\n`;
  const aggregatePrediction = (await batchService.getArtifact(
    second.id,
    "observation_velocity_field_predictions.csv",
  )).content.toString("utf8");
  assert.equal(aggregatePrediction, expectedAggregatePrediction);
  const aggregate = JSON.parse((await batchService.getArtifact(
    second.id,
    "aggregate_scores.json",
  )).content.toString("utf8"));
  assert.equal(aggregate.observationAddedGravityParameters, 0);
  assert.equal(aggregate.observationSucceededSystems, 1);
  const perGalaxy = (await batchService.getArtifact(second.id, "per_galaxy.csv")).content.toString("utf8");
  assert.match(perGalaxy, new RegExp(firstChildren[0].fieldJobId));
  assert.match(perGalaxy, new RegExp(observationId));
});

test("batch cancellation reaches a running observation child without cancelling its completed field", async (t) => {
  const runner = (argumentsValue) => {
    if (argumentsValue.jobType !== "observation_evaluation") {
      return successfulFieldRunner(argumentsValue);
    }
    return new Promise((resolvePromise) => {
      argumentsValue.signal.addEventListener(
        "abort",
        () => resolvePromise({
          exitCode: null, exitSignal: "SIGTERM", timedOut: false, stdout: "", stderr: "",
        }),
        { once: true },
      );
    });
  };
  const { fieldService, batchService } = await fixture(t, { runner });
  const source = await upload(fieldService, "CANCEL-OBSERVATION");
  const observationModel = model();
  observationModel.observables[0].target = "massive_tracers";
  const target = {
    schemaVersion: "sigma-observation-target/1",
    id: "CANCEL-curve",
    kind: "circular_speed_curve",
    observable: "gradient",
    centerM: [0, 0],
    radiiM: [1],
    observedSpeedsMPerS: [10],
    uncertaintiesMPerS: [2],
    provenance: { kind: "cancellation fixture" },
    license: { id: "CC0-1.0", redistributionAllowed: true },
  };
  const submission = await batchService.createBatch({
    schemaVersion: "sigma-batch-submit/1",
    model: observationModel,
    systems: [{ id: "CANCEL", dataUploadId: source.id, observationTargets: [target] }],
    fieldRequest: { schemaVersion: "sigma-field-job-request/1", requestedObservables: ["gradient"] },
    parameterPolicy: { mode: "published_fixed", perObjectParameters: [] },
  });
  let record;
  for (let attempt = 0; attempt < 200; attempt += 1) {
    record = await batchService.getBatch(submission.id);
    const observationId = record.childJobs[0].observationEvaluationJobId;
    if (observationId && (await fieldService.getObservationEvaluationJob(observationId)).state === "running") break;
    await new Promise((resolvePromise) => setTimeout(resolvePromise, 5));
  }
  const observationId = record.childJobs[0].observationEvaluationJobId;
  assert.ok(observationId);
  const cancelled = await batchService.cancelBatch(submission.id);
  assert.equal(cancelled.state, "cancelled");
  await fieldService.waitForIdle();
  assert.equal((await fieldService.getFieldJob(record.childJobs[0].fieldJobId)).state, "succeeded");
  assert.equal((await fieldService.getObservationEvaluationJob(observationId)).state, "cancelled");
});

test("batch restart rebuilds reporting from completed field and observation children without rerunning", async (t) => {
  const calls = { field: 0, observation: 0 };
  const runner = async (argumentsValue) => {
    if (argumentsValue.jobType === "observation_evaluation") {
      calls.observation += 1;
      return successfulObservationRunner(argumentsValue);
    }
    calls.field += 1;
    return successfulFieldRunner(argumentsValue);
  };
  const { fieldService, batchService } = await fixture(t, { runner });
  const source = await upload(fieldService, "RECOVERY");
  const observationModel = model();
  observationModel.observables[0].target = "massive_tracers";
  const target = {
    schemaVersion: "sigma-observation-target/1",
    id: "RECOVERY-curve",
    kind: "circular_speed_curve",
    observable: "gradient",
    centerM: [0, 0],
    radiiM: [1],
    observedSpeedsMPerS: [10],
    uncertaintiesMPerS: [2],
    provenance: { kind: "recovery fixture" },
    license: { id: "CC0-1.0", redistributionAllowed: true },
  };
  const submission = await batchService.createBatch({
    schemaVersion: "sigma-batch-submit/1",
    model: observationModel,
    systems: [{ id: "RECOVERY", dataUploadId: source.id, observationTargets: [target] }],
    fieldRequest: { schemaVersion: "sigma-field-job-request/1", requestedObservables: ["gradient"] },
    parameterPolicy: { mode: "published_fixed", perObjectParameters: [] },
  });
  await batchService.waitForIdle();
  assert.deepEqual(calls, { field: 1, observation: 1 });
  await batchService.close();
  const batchDirectory = resolve(batchService.root, "batches", submission.id);
  const recordPath = resolve(batchDirectory, "record.json");
  const record = JSON.parse(await readFile(recordPath, "utf8"));
  record.state = "running";
  record.phase = "observation";
  record.completedChildren = 0;
  record.successfulChildren = 0;
  await writeFile(recordPath, `${JSON.stringify(record, null, 2)}\n`);
  await rm(resolve(batchDirectory, "artifacts"), { recursive: true, force: true });
  const recovered = new LocalBatchService({
    root: batchService.root,
    fieldService,
    pollMilliseconds: 5,
  });
  await recovered.initialize();
  t.after(async () => recovered.close());
  await recovered.waitForIdle();
  assert.equal((await recovered.getBatch(submission.id)).state, "succeeded");
  assert.deepEqual(calls, { field: 1, observation: 1 });
});

test("a rejected field creates no observation child and an observation failure is scored separately", async (t) => {
  const rejected = async (argumentsValue) => {
    if (argumentsValue.jobType === "field") {
      return {
        exitCode: 2,
        exitSignal: null,
        timedOut: false,
        stdout: "",
        stderr: JSON.stringify({
          schemaVersion: "sigma-field-job-cli-error/1",
          errorType: "ValueError",
          message: "rejected field fixture",
        }),
      };
    }
    throw new Error("observation runner must not be called after a rejected field");
  };
  const rejectedFixture = await fixture(t, { runner: rejected });
  const rejectedSource = await upload(rejectedFixture.fieldService, "REJECTED-FIELD");
  const rejectedModel = model();
  rejectedModel.observables[0].target = "massive_tracers";
  const rejectedTarget = {
    schemaVersion: "sigma-observation-target/1",
    id: "REJECTED-FIELD-curve",
    kind: "circular_speed_curve",
    observable: "gradient",
    centerM: [0, 0],
    radiiM: [1],
    observedSpeedsMPerS: [10],
    uncertaintiesMPerS: [2],
    provenance: { kind: "rejected field observation fixture" },
    license: { id: "CC0-1.0", redistributionAllowed: true },
  };
  const rejectedBatch = await rejectedFixture.batchService.createBatch({
    schemaVersion: "sigma-batch-submit/1",
    model: rejectedModel,
    systems: [{
      id: "REJECTED",
      dataUploadId: rejectedSource.id,
      observationTargets: [rejectedTarget],
    }],
    fieldRequest: { schemaVersion: "sigma-field-job-request/1", requestedObservables: ["gradient"] },
    parameterPolicy: { mode: "published_fixed", perObjectParameters: [] },
  });
  await rejectedFixture.batchService.waitForIdle();
  assert.equal((await rejectedFixture.batchService.getBatch(rejectedBatch.id)).state, "completed_with_failures");
  assert.equal((await rejectedFixture.fieldService.listObservationEvaluationJobs()).items.length, 0);

  const observationFailureRunner = async (argumentsValue) => {
    if (argumentsValue.jobType === "field") return successfulFieldRunner(argumentsValue);
    return {
      exitCode: 2,
      exitSignal: null,
      timedOut: false,
      stdout: "",
      stderr: JSON.stringify({
        schemaVersion: "sigma-observation-evaluation-job-cli-error/1",
        errorType: "ValueError",
        message: "rejected observation fixture",
      }),
    };
  };
  const failedObservationFixture = await fixture(t, { runner: observationFailureRunner });
  const source = await upload(failedObservationFixture.fieldService, "REJECTED-OBSERVATION");
  const observationModel = model();
  observationModel.observables[0].target = "massive_tracers";
  const target = {
    schemaVersion: "sigma-observation-target/1",
    id: "REJECTED-OBSERVATION-curve",
    kind: "circular_speed_curve",
    observable: "gradient",
    centerM: [0, 0],
    radiiM: [1],
    observedSpeedsMPerS: [10],
    uncertaintiesMPerS: [2],
    provenance: { kind: "observation failure fixture" },
    license: { id: "CC0-1.0", redistributionAllowed: true },
  };
  const failedBatch = await failedObservationFixture.batchService.createBatch({
    schemaVersion: "sigma-batch-submit/1",
    model: observationModel,
    systems: [{ id: "REJECTED-OBSERVATION", dataUploadId: source.id, observationTargets: [target] }],
    fieldRequest: { schemaVersion: "sigma-field-job-request/1", requestedObservables: ["gradient"] },
    parameterPolicy: { mode: "published_fixed", perObjectParameters: [] },
  });
  await failedObservationFixture.batchService.waitForIdle();
  const aggregate = JSON.parse((await failedObservationFixture.batchService.getArtifact(
    failedBatch.id,
    "aggregate_scores.json",
  )).content.toString("utf8"));
  assert.equal(aggregate.fieldSucceededSystems, 1);
  assert.equal(aggregate.observationSucceededSystems, 0);
  assert.equal(aggregate.observationScoresAvailable, false);
  const failures = (await failedObservationFixture.batchService.getArtifact(
    failedBatch.id,
    "failures.csv",
  )).content.toString("utf8");
  assert.match(failures, /observation_execution_failure/);
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
