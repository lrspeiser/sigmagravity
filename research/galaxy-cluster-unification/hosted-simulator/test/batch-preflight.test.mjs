import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { prepareBatch } from "../lib/batch-preflight.mjs";
import { sha256 } from "../lib/canonical.mjs";

const model = JSON.parse(
  readFileSync(new URL("../examples/models/newtonian-poisson.json", import.meta.url), "utf8"),
);

function bundle(label, cells = 17) {
  const core = {
    schemaVersion: "sigma-array-bundle/1",
    geometry: {
      coordinateSystem: "cartesian_3d",
      dimensions: 3,
      spacing: [1e19, 1e19, 1e19],
      origin: [-8e19, -8e19, -8e19],
      lengthUnit: "m",
      axisOrder: ["x", "y", "z"],
      referenceFrame: label,
    },
    arrays: [
      {
        key: "baryon_density",
        npzKey: "baryon_density",
        unit: "kg/m^3",
        rank: "scalar",
        role: "source",
        dtype: "<f8",
        shape: [cells, cells, cells],
        elementCount: cells ** 3,
        contentSha256: sha256({ label, cells }),
      },
    ],
    provenance: { kind: "batch_test", label },
    license: { id: "CC0-1.0", redistributionAllowed: true },
  };
  return { ...core, bundleSha256: sha256(core) };
}

function submission(policy = { mode: "published_fixed", perObjectParameters: [] }) {
  return {
    schemaVersion: "sigma-batch-submit/1",
    model,
    systems: [
      { id: "GALAXY-A", dataUploadId: `upload_${"a".repeat(24)}` },
      { id: "GALAXY-B", dataUploadId: `upload_${"b".repeat(24)}` },
    ],
    fieldRequest: {
      schemaVersion: "sigma-field-job-request/1",
      requestedObservables: ["massive_tracer_acceleration"],
      seed: 17,
    },
    parameterPolicy: policy,
  };
}

function resolved() {
  return [
    {
      id: "GALAXY-A",
      source: { kind: "data_upload", id: `upload_${"a".repeat(24)}` },
      inputBundle: bundle("A"),
    },
    {
      id: "GALAXY-B",
      source: { kind: "data_upload", id: `upload_${"b".repeat(24)}` },
      inputBundle: bundle("B", 21),
    },
  ];
}

test("batch preflight binds one fixed model to multiple resolved systems", () => {
  const result = prepareBatch({ submission: submission(), resolvedSystems: resolved() });
  assert.equal(result.valid, true);
  assert.equal(result.systemCount, 2);
  assert.equal(result.executionReadiness.executable, true);
  assert.equal(result.parameterPolicy.mode, "published_fixed");
  assert.equal(result.parameterPolicy.perObjectParameters.length, 0);
  assert.equal(result.systems.length, 2);
  assert.ok(result.resourceEstimate.totalEstimatedMemoryBytes > 0);
});

test("batch preflight supports more than the old 25-system interactive limit", () => {
  const systems = Array.from({ length: 30 }, (_, index) => {
    const id = `GALAXY-${String(index).padStart(2, "0")}`;
    return {
      id,
      source: { kind: "data_upload", id: `upload_${index.toString(16).padStart(24, "0")}` },
      inputBundle: bundle(id),
    };
  });
  const request = submission();
  request.systems = systems.map((item) => ({ id: item.id, dataUploadId: item.source.id }));
  const result = prepareBatch({ submission: request, resolvedSystems: systems });
  assert.equal(result.systemCount, 30);
});

test("batch preflight keeps system-specific observation targets out of the shared model", () => {
  const request = submission();
  const systems = resolved();
  const target = {
    schemaVersion: "sigma-observation-target/1",
    id: "GALAXY-A-rotation",
    kind: "circular_speed_curve",
    observable: "massive_tracer_acceleration",
    centerM: [0, 0, 0],
    radiiM: [1e19, 2e19],
    observedSpeedsMPerS: [20_000, 25_000],
    uncertaintiesMPerS: [2_000, 2_500],
    provenance: { kind: "published fixture" },
    license: { id: "CC-BY-4.0", redistributionAllowed: true },
  };
  request.systems[0].observationTargets = [target];
  systems[0].observationTargets = [target];
  const result = prepareBatch({ submission: request, resolvedSystems: systems });
  assert.equal(result.scoredObservationTargets, 1);
  assert.equal(result.systems[0].observationTargetSummary[0].scored, true);
  assert.equal(result.systems[1].observationTargets.length, 0);
  assert.match(result.claimBoundary[0], /separately content-addressed observation jobs/);
  const changedTarget = { ...target, uncertaintiesMPerS: [3_000, 3_500] };
  request.systems[0].observationTargets = [changedTarget];
  systems[0].observationTargets = [changedTarget];
  const changed = prepareBatch({ submission: request, resolvedSystems: systems });
  assert.equal(changed.systems[0].fieldPreflightSha256, result.systems[0].fieldPreflightSha256);
  assert.notEqual(changed.systems[0].observationBindingSha256, result.systems[0].observationBindingSha256);
  assert.notEqual(changed.preflightSha256, result.preflightSha256);
});

test("fitted and per-galaxy policies are explicit but not silently executed", () => {
  const fittedModel = structuredClone(model);
  fittedModel.parameterPolicy.mode = "universal_fit";
  const request = submission({ mode: "universal_fit", perObjectParameters: [] });
  request.model = fittedModel;
  const result = prepareBatch({ submission: request, resolvedSystems: resolved() });
  assert.equal(result.executionReadiness.executable, false);
  assert.match(result.executionReadiness.blockers[0], /not implemented/);

  assert.throws(
    () => prepareBatch({
      submission: submission({ mode: "published_fixed", perObjectParameters: ["halo_mass"] }),
      resolvedSystems: resolved(),
    }),
    /disclosures differ|cannot declare/,
  );
});

test("train-validation-holdout policy requires complete non-overlapping splits", () => {
  const splitModel = structuredClone(model);
  splitModel.parameterPolicy.mode = "train_validation_holdout";
  const systems = [...resolved(), {
    id: "GALAXY-C",
    source: { kind: "data_upload", id: `upload_${"c".repeat(24)}` },
    inputBundle: bundle("C"),
  }];
  const request = submission({
    mode: "train_validation_holdout",
    perObjectParameters: [],
    splits: { train: ["GALAXY-A"], validation: ["GALAXY-B"], holdout: ["GALAXY-C"] },
  });
  request.model = splitModel;
  request.systems.push({ id: "GALAXY-C", dataUploadId: `upload_${"c".repeat(24)}` });
  const result = prepareBatch({ submission: request, resolvedSystems: systems });
  assert.equal(result.executionReadiness.executable, false);
  assert.deepEqual(result.parameterPolicy.splits.holdout, ["GALAXY-C"]);
});
