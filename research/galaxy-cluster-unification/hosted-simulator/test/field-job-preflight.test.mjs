import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { sha256 } from "../lib/canonical.mjs";
import { prepareFieldJob } from "../lib/field-job-preflight.mjs";
import { validateFieldModel } from "../lib/field-model.mjs";

const model = JSON.parse(readFileSync(new URL("../examples/models/refracted-gravity.json", import.meta.url), "utf8"));
const twoPotentialModel = JSON.parse(readFileSync(new URL("../examples/models/two-potential.json", import.meta.url), "utf8"));

function bindConfirmation(manifest) {
  manifest.source.confirmedCanonical = false;
  delete manifest.source.confirmedModelSha256;
  manifest.source.confirmedCanonical = true;
  manifest.source.confirmedModelSha256 = validateFieldModel(manifest).modelSha256;
}

function inputBundle() {
  const core = {
    schemaVersion: "sigma-array-bundle/1",
    geometry: {
      coordinateSystem: "cartesian_3d",
      dimensions: 3,
      spacing: [3.085677581491367e19, 3.085677581491367e19, 3.085677581491367e19],
      origin: [-4.937084130386187e20, -4.937084130386187e20, -4.937084130386187e20],
      lengthUnit: "m",
      axisOrder: ["x", "y", "z"],
      referenceFrame: "barycentric_cartesian",
    },
    arrays: [{
      key: "baryon_density",
      npzKey: "baryon_density",
      unit: "kg/m^3",
      rank: "scalar",
      role: "source",
      dtype: "<f8",
      shape: [33, 33, 33],
      elementCount: 35937,
      contentSha256: "1".repeat(64),
    }],
    provenance: { kind: "test_fixture", citation: "repository conformance test" },
    license: { id: "CC0-1.0", redistributionAllowed: true },
  };
  return { ...core, bundleSha256: sha256(core) };
}

function payload() {
  return {
    model: structuredClone(model),
    inputBundle: inputBundle(),
    request: {
      schemaVersion: "sigma-field-job-request/1",
      requestedObservables: ["massive_tracer_acceleration"],
      boundaryFields: {},
      seed: 0,
    },
  };
}

test("field preflight binds a generic model to content-hashed 3D data", () => {
  const first = prepareFieldJob(payload());
  const second = prepareFieldJob(payload());
  assert.equal(first.valid, true);
  assert.equal(first.state, "worker_not_connected");
  assert.equal(first.preflightSha256, second.preflightSha256);
  assert.equal(first.parameterAccounting.perObject, 0);
  assert.equal(first.confirmation.confirmed, true);
  assert.equal(first.resourceEstimate.cellCount, 33 ** 3);
  assert.deepEqual(first.executionReadiness.blockers, [
    "array_bytes_not_uploaded",
    "generic_scientific_worker_not_connected",
  ]);
});

test("field preflight binds the exact axisymmetric radial convention", () => {
  const request = payload();
  request.model.geometry.coordinateSystem = "axisymmetric_cylindrical";
  request.model.geometry.dimensions = 2;
  bindConfirmation(request.model);
  request.inputBundle.geometry.coordinateSystem = "axisymmetric_cylindrical";
  request.inputBundle.geometry.dimensions = 2;
  request.inputBundle.geometry.spacing = [1, 1];
  request.inputBundle.geometry.origin = [0, -16];
  request.inputBundle.geometry.axisOrder = ["r", "z"];
  request.inputBundle.arrays[0].shape = [33, 33];
  request.inputBundle.arrays[0].elementCount = 33 ** 2;
  const { bundleSha256: _oldHash, ...core } = request.inputBundle;
  request.inputBundle = { ...core, bundleSha256: sha256(core) };

  const result = prepareFieldJob(request);
  assert.equal(result.valid, true);
  assert.deepEqual(result.geometry.axisOrder, ["r", "z"]);
  assert.deepEqual(result.geometry.origin, [0, -16]);
});

test("axisymmetric preflight rejects a fabricated wall or swapped axes", () => {
  const request = payload();
  request.model.geometry.coordinateSystem = "axisymmetric_cylindrical";
  request.model.geometry.dimensions = 2;
  bindConfirmation(request.model);
  request.inputBundle.geometry.coordinateSystem = "axisymmetric_cylindrical";
  request.inputBundle.geometry.dimensions = 2;
  request.inputBundle.geometry.axisOrder = ["z", "r"];
  request.inputBundle.geometry.origin = [0, -16];
  request.inputBundle.arrays[0].shape = [33, 33];
  request.inputBundle.arrays[0].elementCount = 33 ** 2;
  let { bundleSha256: _oldHash, ...core } = request.inputBundle;
  request.inputBundle = { ...core, bundleSha256: sha256(core) };
  assert.throws(() => prepareFieldJob(request), /axisOrder/);

  request.inputBundle.geometry.axisOrder = ["r", "z"];
  request.inputBundle.geometry.origin = [1, -16];
  ({ bundleSha256: _oldHash, ...core } = request.inputBundle);
  request.inputBundle = { ...core, bundleSha256: sha256(core) };
  assert.throws(() => prepareFieldJob(request), /origin/);
});

test("field preflight refuses a structurally valid but unconfirmed model", () => {
  const request = payload();
  request.model.source.confirmedCanonical = false;
  delete request.model.source.confirmedModelSha256;
  const result = prepareFieldJob(request);
  assert.equal(result.valid, false);
  assert.equal(result.state, "unconfirmed_model");
  assert.equal(result.confirmation.confirmed, false);
  assert.match(result.errors.join(" "), /exact canonical model hash/);
});

test("field preflight rejects data whose units differ from the model", () => {
  const request = payload();
  request.inputBundle.arrays[0].unit = "kg/m^2";
  const core = Object.fromEntries(Object.entries(request.inputBundle).filter(([key]) => key !== "bundleSha256"));
  request.inputBundle.bundleSha256 = sha256(core);
  assert.throws(() => prepareFieldJob(request), /rank or unit does not match/);
});

test("field preflight rejects modified bundle metadata", () => {
  const request = payload();
  request.inputBundle.arrays[0].shape = [65, 65, 65];
  assert.throws(() => prepareFieldJob(request), /manifest hash mismatch/);
});

test("field preflight binds an uncertainty-aware circular-speed target after the solve", () => {
  const request = payload();
  request.request.observationTargets = [{
    schemaVersion: "sigma-observation-target/1",
    id: "DDO-test-rotation",
    kind: "circular_speed_curve",
    observable: "massive_tracer_acceleration",
    centerM: [0, 0, 0],
    planeAxes: [0, 1],
    radiiM: [3.085677581491367e19, 6.171355162982734e19],
    observedSpeedsMPerS: [20_000, 25_000],
    uncertaintiesMPerS: [2_000, 2_500],
    fittedNuisanceParameters: 0,
    provenance: { kind: "published rotation curve fixture" },
    license: { id: "CC-BY-4.0", redistributionAllowed: true },
  }];
  const result = prepareFieldJob(request);
  assert.equal(result.observationTargets.length, 1);
  assert.equal(result.observationTargets[0].scored, true);
  assert.equal(result.observationTargets[0].pointCount, 2);
  assert.match(result.observationTargets[0].targetSha256, /^[0-9a-f]{64}$/);
});

test("a circular-speed target cannot silently use a photon observable", () => {
  const request = payload();
  request.model.observables[0].target = "photons";
  bindConfirmation(request.model);
  request.request.observationTargets = [{
    schemaVersion: "sigma-observation-target/1",
    id: "wrong-channel",
    kind: "circular_speed_curve",
    observable: "massive_tracer_acceleration",
    centerM: [0, 0, 0],
    radiiM: [3.085677581491367e19],
    provenance: { kind: "negative fixture" },
    license: { id: "CC0-1.0", redistributionAllowed: true },
  }];
  assert.throws(() => prepareFieldJob(request), /massive_tracers vector/);
});

test("field preflight binds content-hashed resolved velocity maps and beam data", () => {
  const request = payload();
  const observationRecords = [
    ["major", "m", "auxiliary", [17, 17]],
    ["minor", "m", "auxiliary", [17, 17]],
    ["observed_velocity", "m/s", "auxiliary", [17, 17]],
    ["velocity_uncertainty", "m/s", "uncertainty", [17, 17]],
    ["intensity", "1", "auxiliary", [17, 17]],
    ["valid_mask", "1", "mask", [17, 17]],
    ["beam", "1", "auxiliary", [7, 7]],
  ].map(([key, unit, role, shape], index) => ({
    key,
    npzKey: key,
    unit,
    rank: "scalar",
    role,
    dtype: "<f8",
    shape,
    elementCount: shape[0] * shape[1],
    contentSha256: String(index + 2).repeat(64),
  }));
  request.inputBundle.arrays.push(...observationRecords);
  const core = Object.fromEntries(Object.entries(request.inputBundle).filter(([key]) => key !== "bundleSha256"));
  request.inputBundle.bundleSha256 = sha256(core);
  request.request.observationTargets = [{
    schemaVersion: "sigma-observation-target/1",
    id: "DDO-test-velocity-map",
    kind: "line_of_sight_velocity_field",
    observable: "massive_tracer_acceleration",
    centerM: [0, 0, 0],
    planeAxes: [0, 1],
    inclinationDeg: 45,
    handedness: 1,
    majorCoordinateArrayKey: "major",
    minorCoordinateArrayKey: "minor",
    observedVelocityArrayKey: "observed_velocity",
    uncertaintyArrayKey: "velocity_uncertainty",
    intensityWeightArrayKey: "intensity",
    maskArrayKey: "valid_mask",
    beamKernelArrayKey: "beam",
    weighting: "intensity_inverse_variance",
    minimumValidPixels: 100,
    provenance: { kind: "resolved velocity fixture" },
    license: { id: "CC-BY-4.0", redistributionAllowed: true },
  }];
  const result = prepareFieldJob(request);
  assert.equal(result.observationTargets[0].kind, "line_of_sight_velocity_field");
  assert.equal(result.observationTargets[0].pointCount, 289);
  assert.equal(result.observationTargets[0].scored, true);
});

test("resolved velocity preflight rejects a unit-mismatched coordinate map", () => {
  const request = payload();
  request.inputBundle.arrays.push({
    key: "major", npzKey: "major", unit: "km/s", rank: "scalar", role: "auxiliary",
    dtype: "<f8", shape: [17, 17], elementCount: 289, contentSha256: "2".repeat(64),
  });
  request.inputBundle.arrays.push({
    key: "minor", npzKey: "minor", unit: "m", rank: "scalar", role: "auxiliary",
    dtype: "<f8", shape: [17, 17], elementCount: 289, contentSha256: "3".repeat(64),
  });
  const core = Object.fromEntries(Object.entries(request.inputBundle).filter(([key]) => key !== "bundleSha256"));
  request.inputBundle.bundleSha256 = sha256(core);
  request.request.observationTargets = [{
    schemaVersion: "sigma-observation-target/1",
    id: "bad-map",
    kind: "line_of_sight_velocity_field",
    observable: "massive_tracer_acceleration",
    centerM: [0, 0, 0],
    inclinationDeg: 45,
    handedness: 1,
    majorCoordinateArrayKey: "major",
    minorCoordinateArrayKey: "minor",
    provenance: { kind: "negative fixture" },
    license: { id: "CC0-1.0", redistributionAllowed: true },
  }];
  assert.throws(() => prepareFieldJob(request), /majorCoordinateArrayKey must reference a scalar m array/);
});

test("resolved velocity preflight separates emission and score masks", () => {
  const request = payload();
  const shape = request.inputBundle.arrays[0].shape;
  for (const [key, unit] of [
    ["major", "m"], ["minor", "m"], ["score_mask", "1"], ["emission_mask", "1"],
  ]) {
    request.inputBundle.arrays.push({
      key,
      npzKey: key,
      unit,
      rank: "scalar",
      role: "auxiliary",
      dtype: "<f8",
      shape: shape.slice(0, 2),
      elementCount: shape[0] * shape[1],
      contentSha256: `${key.length}`.repeat(64).slice(0, 64),
    });
  }
  const { bundleSha256: _oldHash, ...bundleCore } = request.inputBundle;
  request.inputBundle = { ...bundleCore, bundleSha256: sha256(bundleCore) };
  request.model.observables[0].target = "massive_tracers";
  request.model.observables[0].unit = "m/s^2";
  request.model.observables[0].rank = "vector";
  bindConfirmation(request.model);
  request.request.observationTargets = [{
    schemaVersion: "sigma-observation-target/1",
    id: "separate-masks",
    kind: "line_of_sight_velocity_field",
    observable: request.model.observables[0].id,
    centerM: [0, 0, 0],
    inclinationDeg: 45,
    handedness: 1,
    majorCoordinateArrayKey: "major",
    minorCoordinateArrayKey: "minor",
    scoreMaskArrayKey: "score_mask",
    emissionMaskArrayKey: "emission_mask",
    minimumValidPixels: 25,
    provenance: { kind: "test" },
    license: { id: "CC0-1.0", redistributionAllowed: true },
  }];
  const result = prepareFieldJob(request);
  assert.equal(result.valid, true);
  request.request.observationTargets[0].maskArrayKey = "score_mask";
  assert.throws(() => prepareFieldJob(request), /either maskArrayKey or scoreMaskArrayKey/);
});

test("field preflight binds typed photon maps and two independent score channels", () => {
  const request = payload();
  request.model = structuredClone(twoPotentialModel);
  request.request.requestedObservables = ["photon_lensing_acceleration"];
  const shape = [33, 33];
  for (const [index, [key, unit]] of [
    ["alpha_east", "arcsec"],
    ["alpha_north", "arcsec"],
    ["alpha_sigma", "arcsec"],
    ["reduced_g1", "1"],
    ["reduced_g2", "1"],
    ["reduced_g_sigma", "1"],
    ["score_mask", "1"],
  ].entries()) {
    request.inputBundle.arrays.push({
      key,
      npzKey: key,
      unit,
      rank: "scalar",
      role: "auxiliary",
      dtype: "<f8",
      shape,
      elementCount: shape[0] * shape[1],
      contentSha256: String(index + 2).repeat(64),
    });
  }
  const { bundleSha256: _oldHash, ...bundleCore } = request.inputBundle;
  request.inputBundle = { ...bundleCore, bundleSha256: sha256(bundleCore) };
  request.request.observationTargets = [{
    schemaVersion: "sigma-observation-target/1",
    id: "typed-photon-map",
    kind: "photon_lensing_map",
    observable: "photon_lensing_acceleration",
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
    minimumValidPixels: 100,
    provenance: { kind: "synthetic photon fixture" },
    license: { id: "CC0-1.0", redistributionAllowed: true },
  }];
  const result = prepareFieldJob(request);
  assert.equal(result.observationTargets[0].kind, "photon_lensing_map");
  assert.equal(result.observationTargets[0].scored, true);
  assert.equal(result.observationTargets[0].pointCount, 2 * 33 * 33);

  request.request.observationTargets[0].eastAxis = 0;
  assert.throws(() => prepareFieldJob(request), /permutation/);
});

test("photon preflight rejects massive-only observables and incomplete map triples", () => {
  const request = payload();
  request.request.observationTargets = [{
    schemaVersion: "sigma-observation-target/1",
    id: "wrong-photon-channel",
    kind: "photon_lensing_map",
    observable: "massive_tracer_acceleration",
    northAxis: 0,
    eastAxis: 1,
    lineOfSightAxis: 2,
    distanceRatio: 0.72,
    lensAngularDiameterDistanceM: 3.0e25,
    provenance: { kind: "negative fixture" },
    license: { id: "CC0-1.0", redistributionAllowed: true },
  }];
  assert.throws(() => prepareFieldJob(request), /photons or both vector/);

  request.model = structuredClone(twoPotentialModel);
  request.request.requestedObservables = ["photon_lensing_acceleration"];
  request.request.observationTargets[0].observable = "photon_lensing_acceleration";
  request.request.observationTargets[0].observedAlphaEastArcsecArrayKey = "only-east";
  assert.throws(() => prepareFieldJob(request), /both observed component maps/);
});

test("raw multiple-image preflight counts source nuisances separately", () => {
  const request = payload();
  request.model = structuredClone(twoPotentialModel);
  request.request.requestedObservables = ["photon_lensing_acceleration"];
  request.request.observationTargets = [{
    schemaVersion: "sigma-observation-target/1",
    id: "raw-image-families",
    kind: "multiple_image_systems",
    observable: "photon_lensing_acceleration",
    northAxis: 0,
    eastAxis: 1,
    lineOfSightAxis: 2,
    lensAngularDiameterDistanceM: 3.0e25,
    skyCenterM: [0, 0, 0],
    rootSearchBoundArcsec: 40,
    rootGridPoints: 161,
    families: [
      {
        id: "source-a",
        distanceRatio: 0.72,
        observedImagesArcsec: [[-8, 2], [11, 3]],
        positionUncertaintiesArcsec: [0.05, 0.05],
      },
      {
        id: "source-b",
        distanceRatio: 0.64,
        observedImagesArcsec: [[-5, -4], [6, -3], [1, 8]],
        positionUncertaintiesArcsec: [0.08, 0.08, 0.08],
      },
    ],
    provenance: { kind: "synthetic root fixture" },
    license: { id: "CC0-1.0", redistributionAllowed: true },
  }];
  const result = prepareFieldJob(request);
  assert.equal(result.observationTargets[0].kind, "multiple_image_systems");
  assert.equal(result.observationTargets[0].scored, true);
  assert.equal(result.observationTargets[0].pointCount, 10);
  assert.equal(result.observationTargets[0].fittedNuisanceParameters, 4);

  request.request.observationTargets[0].fittedNuisanceParameters = 2;
  assert.throws(() => prepareFieldJob(request), /two source coordinates per family/);
  delete request.request.observationTargets[0].fittedNuisanceParameters;
  request.request.observationTargets[0].families[0].positionUncertaintiesArcsec = [0.05];
  assert.throws(() => prepareFieldJob(request), /must contain 2 values/);
});
