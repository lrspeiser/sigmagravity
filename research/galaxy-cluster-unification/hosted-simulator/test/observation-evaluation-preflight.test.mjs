import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { resolve } from "node:path";
import test from "node:test";
import { sha256 } from "../lib/canonical.mjs";
import { prepareObservationEvaluationJob } from "../lib/observation-evaluation-preflight.mjs";
import { validateFieldModel } from "../lib/field-model.mjs";

async function model() {
  return JSON.parse(await readFile(resolve(import.meta.dirname, "../examples/models/newtonian-poisson.json"), "utf8"));
}

function observationBundle() {
  const arrays = [
    ["major", "m"],
    ["minor", "m"],
    ["observed", "m/s"],
    ["uncertainty", "m/s"],
    ["score_mask", "1"],
  ].map(([key, unit], index) => ({
    key,
    npzKey: key,
    unit,
    rank: "scalar",
    role: "observation",
    dtype: "<f8",
    shape: [17, 17],
    elementCount: 289,
    contentSha256: String(index + 1).repeat(64),
  }));
  const core = {
    schemaVersion: "sigma-array-bundle/1",
    geometry: { coordinateSystem: "sky_plane", dimensions: 2, spacing: [1, 1], lengthUnit: "pixel" },
    arrays,
    provenance: { kind: "P0732 preflight fixture" },
    license: { id: "CC0-1.0", redistributionAllowed: true },
  };
  return { ...core, bundleSha256: sha256(core) };
}

function target() {
  return {
    schemaVersion: "sigma-observation-target/1",
    id: "resolved-map",
    kind: "line_of_sight_velocity_field",
    observable: "massive_tracer_acceleration",
    centerM: [0, 0, 0],
    planeAxes: [0, 1],
    inclinationDeg: 60,
    handedness: 1,
    majorCoordinateArrayKey: "major",
    minorCoordinateArrayKey: "minor",
    observedVelocityArrayKey: "observed",
    uncertaintyArrayKey: "uncertainty",
    scoreMaskArrayKey: "score_mask",
    minimumValidPixels: 25,
    provenance: { kind: "P0732 preflight fixture" },
    license: { id: "CC0-1.0", redistributionAllowed: true },
  };
}

async function fixture() {
  const fieldModel = await model();
  const modelSha256 = validateFieldModel(fieldModel).modelSha256;
  const jobSha256 = "a".repeat(64);
  const fieldJob = {
    schemaVersion: "sigma-field-job/1",
    id: `fieldjob_${jobSha256.slice(0, 24)}`,
    jobSha256,
    modelSha256,
    geometry: { coordinateSystem: "cartesian_3d", dimensions: 3, spacing: [1, 1, 1], origin: [-8, -8, -8], lengthUnit: "m" },
  };
  const resultSha256 = "b".repeat(64);
  const observables = [0, 1, 2].map((axis) => ({
    key: `massive_tracer_acceleration__axis${axis}`,
    dtype: "<f8",
    shape: [17, 17, 17],
    contentSha256: String(axis + 3).repeat(64),
  }));
  const fieldManifestCore = {
    schemaVersion: "sigma-field-run-manifest/1",
    state: "succeeded",
    jobId: fieldJob.id,
    scientificResultSha256: resultSha256,
  };
  return {
    gatewayFieldJob: { id: "job_111111111111111111111111", jobType: "field", state: "succeeded" },
    fieldManifest: {
      ...fieldManifestCore,
      manifestSha256: sha256(fieldManifestCore),
    },
    fieldJob,
    model: fieldModel,
    scientificResult: {
      schemaVersion: "sigma-field-result/1",
      state: "succeeded",
      converged: true,
      jobId: fieldJob.id,
      jobSha256,
      resultSha256,
      observables,
    },
    observationBundle: observationBundle(),
    observationTargets: [target()],
    fieldArtifactHashes: {
      model: "d".repeat(64),
      job: "e".repeat(64),
      scientificResult: "f".repeat(64),
      observables: "0".repeat(64),
    },
  };
}

test("observation preflight binds immutable 3D field and 2D observation identities", async () => {
  const prepared = prepareObservationEvaluationJob(await fixture());
  assert.equal(prepared.valid, true);
  assert.equal(prepared.state, "ready_for_local_worker");
  assert.equal(prepared.field.observableIds[0], "massive_tracer_acceleration");
  assert.equal(prepared.observationTargets[0].kind, "line_of_sight_velocity_field");
  assert.equal(prepared.evaluationAddedGravityParameters, 0);
  assert.equal(prepared.parameterAccounting.perObject, 0);
  assert.match(prepared.preflightSha256, /^[0-9a-f]{64}$/);
});

test("observation preflight rejects a nonconverged source field", async () => {
  const payload = await fixture();
  payload.scientificResult.converged = false;
  assert.throws(() => prepareObservationEvaluationJob(payload), /converged and succeeded/);
});

test("changed observation data changes the evaluation preflight identity", async () => {
  const first = await fixture();
  const second = await fixture();
  second.observationBundle.arrays[0].contentSha256 = "9".repeat(64);
  const core = Object.fromEntries(Object.entries(second.observationBundle).filter(([key]) => key !== "bundleSha256"));
  second.observationBundle.bundleSha256 = sha256(core);
  assert.notEqual(
    prepareObservationEvaluationJob(first).preflightSha256,
    prepareObservationEvaluationJob(second).preflightSha256,
  );
});
