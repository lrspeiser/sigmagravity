import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { sha256 } from "../lib/canonical.mjs";
import { prepareFieldJob } from "../lib/field-job-preflight.mjs";

const model = JSON.parse(readFileSync(new URL("../examples/models/refracted-gravity.json", import.meta.url), "utf8"));

function inputBundle() {
  const core = {
    schemaVersion: "sigma-array-bundle/1",
    geometry: {
      coordinateSystem: "cartesian_3d",
      dimensions: 3,
      spacing: [3.085677581491367e19, 3.085677581491367e19, 3.085677581491367e19],
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
    model,
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
  assert.equal(first.resourceEstimate.cellCount, 33 ** 3);
  assert.deepEqual(first.executionReadiness.blockers, [
    "array_bytes_not_uploaded",
    "generic_scientific_worker_not_connected",
  ]);
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
