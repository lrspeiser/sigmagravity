import test from "node:test";
import assert from "node:assert/strict";
import { sha256 } from "../lib/canonical.mjs";
import { prepareInverseResponseJob } from "../lib/inverse-response-preflight.mjs";

function bundle(targetScientificRole = "model_derived_discovery_target") {
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
        rank: "scalar", role: "auxiliary", scientificRole: targetScientificRole,
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
    provenance: { kind: "synthetic_injected_kernel_fixture" },
    license: { id: "CC0-1.0", redistributionAllowed: true },
  };
  return { ...core, bundleSha256: sha256(core) };
}

function submission() {
  return {
    schemaVersion: "sigma-inverse-response-job-submit/1",
    dataUploadId: `upload_${"1".repeat(24)}`,
    systems: [1, 2].map((index) => ({
      id: `SYNTH-${index}`,
      sourceKey: `baryons_${index}`,
      targetKey: `response_${index}`,
      uncertaintyKey: `uncertainty_${index}`,
    })),
    kernel: {
      shape: [5, 5], ridge: 1e-10, smoothness: 1e-8, nonnegative: true,
      regularizationMultipliers: [0.1, 1, 10],
    },
    uncertainty: { ensembleSize: 20, seed: 17 },
    nullControls: { kind: "source_radial_angle_shuffle", count: 19, seed: 23 },
    outputLicense: { id: "CC-BY-4.0", redistributionAllowed: true },
  };
}

test("inverse response preflight binds roles, uncertainty, nulls, and parameter counts", () => {
  const first = prepareInverseResponseJob({ submission: submission(), inputBundle: bundle() });
  const second = prepareInverseResponseJob({ submission: submission(), inputBundle: bundle() });
  assert.equal(first.valid, true);
  assert.equal(first.preflightSha256, second.preflightSha256);
  assert.equal(first.systemCount, 2);
  assert.equal(first.parameterAccounting.fittedDiscoveryKernelCells, 25);
  assert.equal(first.parameterAccounting.fittedUniversalResponseAmplitudes, 1);
  assert.equal(first.parameterAccounting.fittedPerSystemGravityParameters, 0);
  assert.ok(first.resourceEstimate.estimatedMemoryBytes > 0);
  assert.ok(first.dataRoleAudit.every((row) => row.heldOutRawObservationsUsed === false));
  assert.equal(first.workerRequest.nullControls.families.length, 1);
  assert.equal(first.workerRequest.nullControls.families[0].count, 19);
  assert.equal(first.workerRequest.nullControls.combinationRule, "all_declared_families");
});

test("inverse response preflight normalizes a deterministic multi-null suite", () => {
  const value = submission();
  value.nullControls = {
    combinationRule: "all_declared_families",
    families: [
      { kind: "source_radial_angle_shuffle", count: 19, seed: 31 },
      { kind: "source_phase_scramble", count: 23, seed: 32 },
      { kind: "target_system_permutation", count: 29, seed: 33 },
      { kind: "target_radial_angle_shuffle", count: 31, seed: 34 },
      { kind: "source_missing_baryon_dropout", count: 37, seed: 35, dropoutFraction: 0.2 },
    ],
  };
  const result = prepareInverseResponseJob({ submission: value, inputBundle: bundle() });
  assert.equal(result.workerRequest.nullControls.families.length, 5);
  assert.equal(result.workerRequest.nullControls.families[4].dropoutFraction, 0.2);
  assert.equal(result.resourceEstimate.estimatedFits, 1 + 20 + 139 + 3);
});

test("inverse preflight rejects a raw observation as an inverse target", () => {
  assert.throws(
    () => prepareInverseResponseJob({ submission: submission(), inputBundle: bundle("raw_observation") }),
    /scientificRole=model_derived_discovery_target/,
  );
});

test("inverse preflight rejects an even kernel and weak null ensemble", () => {
  const value = submission();
  value.kernel.shape = [4, 5];
  assert.throws(
    () => prepareInverseResponseJob({ submission: value, inputBundle: bundle() }),
    /odd integers/,
  );
  value.kernel.shape = [5, 5];
  value.nullControls.count = 18;
  assert.throws(
    () => prepareInverseResponseJob({ submission: value, inputBundle: bundle() }),
    /19 to 999/,
  );
  value.nullControls.count = 19;
  value.kernel.nonnegative = "false";
  assert.throws(
    () => prepareInverseResponseJob({ submission: value, inputBundle: bundle() }),
    /must be a boolean/,
  );
});

test("inverse preflight rejects malformed null suites", () => {
  const mixed = submission();
  mixed.nullControls = {
    kind: "source_radial_angle_shuffle",
    count: 19,
    seed: 1,
    families: [{ kind: "source_phase_scramble", count: 19, seed: 2 }],
  };
  assert.throws(
    () => prepareInverseResponseJob({ submission: mixed, inputBundle: bundle() }),
    /either legacy.*or families/,
  );
  const badDropout = submission();
  badDropout.nullControls = {
    families: [{ kind: "source_phase_scramble", count: 19, seed: 2, dropoutFraction: 0.2 }],
  };
  assert.throws(
    () => prepareInverseResponseJob({ submission: badDropout, inputBundle: bundle() }),
    /dropoutFraction is only valid/,
  );
  const duplicate = submission();
  duplicate.nullControls = {
    families: [
      { kind: "source_phase_scramble", count: 19, seed: 2 },
      { kind: "source_phase_scramble", count: 19, seed: 3 },
    ],
  };
  assert.throws(
    () => prepareInverseResponseJob({ submission: duplicate, inputBundle: bundle() }),
    /family kinds must be unique/,
  );
});
