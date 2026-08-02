import assert from "node:assert/strict";
import test from "node:test";
import { sha256 } from "../lib/canonical.mjs";
import { prepareGalaxyJob } from "../lib/galaxy-job-preflight.mjs";

function bundle() {
  const core = {
    schemaVersion: "sigma-array-bundle/1",
    geometry: {
      coordinateSystem: "cartesian_2d",
      dimensions: 2,
      spacing: [0.2, 0.2],
      lengthUnit: "kpc",
      axisOrder: ["x", "y"],
    },
    arrays: ["gas_surface_density", "stellar_surface_density"].map((key, index) => ({
      key,
      npzKey: key,
      unit: "M_sun/kpc^2",
      rank: "scalar",
      role: "source",
      dtype: "<f8",
      shape: [65, 65],
      elementCount: 4225,
      contentSha256: String(index + 1).repeat(64),
    })),
    provenance: { kind: "test_fixture" },
    license: { id: "CC0-1.0", redistributionAllowed: true },
  };
  return { ...core, bundleSha256: sha256(core) };
}

function packageFixture() {
  const core = {
    schemaVersion: "1.0.0",
    generator: "radial-fourier-sparse-residual",
    galaxy: "FIXTURE",
    grid: { cellsPerAxis: 65, minimumKpc: -6.4, maximumKpc: 6.4, spacingKpc: 0.2 },
    components: { gas: { fixture: true }, stars: { fixture: true } },
    gravityParameters: {},
    velocityTargetsUsed: false,
  };
  return { ...core, contentSha256: sha256(core) };
}

test("extract preflight binds resolved maps without gravity parameters", () => {
  const result = prepareGalaxyJob({
    submission: {
      schemaVersion: "sigma-galaxy-job-submit/1",
      operation: "extract_roundtrip",
      dataUploadId: `upload_${"1".repeat(24)}`,
      galaxy: "FIXTURE",
      extractionControls: { radialBins: 20, maximumFourierMode: 4, residualFeatureCountPerComponent: 32 },
      vertical: { enabled: true, realizations: 3, zCells: 33, seed: 8 },
      outputLicense: { id: "CC0-1.0", redistributionAllowed: true },
    },
    inputBundle: bundle(),
  });
  assert.equal(result.valid, true);
  assert.deepEqual(result.gridShape, [65, 65]);
  assert.equal(result.parameterAccounting.gravityUniversal, 0);
  assert.equal(result.parameterAccounting.gravityPerObject, 0);
  assert.equal(result.workerRequest.parameterPackage, null);
});

test("generate preflight accepts content-hashed parameters and normalizes controls", () => {
  const result = prepareGalaxyJob({
    submission: {
      schemaVersion: "sigma-galaxy-job-submit/1",
      operation: "generate",
      parameterPackage: packageFixture(),
      generationControls: {
        gas: { massScale: 1.5, radialScale: 0.8, centerOffsetKpc: [0.1, -0.2] },
      },
      outputGrid: { cellsPerAxis: 25, extentScale: 1.5 },
      vertical: { enabled: false },
      outputLicense: { id: "CC-BY-4.0", redistributionAllowed: true },
    },
  });
  assert.equal(result.operation, "generate");
  assert.equal(result.workerRequest.generationControls.gas.mass_scale, 1.5);
  assert.equal(result.workerRequest.generationControls.gas.radial_scale, 0.8);
  assert.deepEqual(result.workerRequest.generationControls.gas.center_offset_kpc, [0.1, -0.2]);
  assert.deepEqual(result.gridShape, [25, 25]);
  assert.deepEqual(result.workerRequest.outputGrid, { cellsPerAxis: 25, extentScale: 1.5 });
});

test("galaxy preflight rejects hidden gravity state and incompatible data", () => {
  const parameters = packageFixture();
  parameters.gravityParameters = { fitted: 1 };
  const core = Object.fromEntries(Object.entries(parameters).filter(([key]) => key !== "contentSha256"));
  parameters.contentSha256 = sha256(core);
  assert.throws(
    () => prepareGalaxyJob({
      submission: {
        schemaVersion: "sigma-galaxy-job-submit/1",
        operation: "generate",
        parameterPackage: parameters,
        outputLicense: { id: "CC0-1.0", redistributionAllowed: true },
      },
    }),
    /gravity-independent/,
  );
  const incompatible = bundle();
  incompatible.geometry.lengthUnit = "m";
  const bundleCore = Object.fromEntries(Object.entries(incompatible).filter(([key]) => key !== "bundleSha256"));
  incompatible.bundleSha256 = sha256(bundleCore);
  assert.throws(
    () => prepareGalaxyJob({
      submission: {
        schemaVersion: "sigma-galaxy-job-submit/1",
        operation: "extract_roundtrip",
        outputLicense: { id: "CC0-1.0", redistributionAllowed: true },
      },
      inputBundle: incompatible,
    }),
    /geometry in kpc/,
  );
});

test("generate preflight rejects shrinking or unbounded output boxes", () => {
  const submission = {
    schemaVersion: "sigma-galaxy-job-submit/1",
    operation: "generate",
    parameterPackage: packageFixture(),
    outputGrid: { cellsPerAxis: 25, extentScale: 0.9 },
    outputLicense: { id: "CC0-1.0", redistributionAllowed: true },
  };
  assert.throws(
    () => prepareGalaxyJob({ submission }),
    /extentScale must be finite and between 1 and 4/,
  );
  submission.outputGrid.extentScale = 4.1;
  assert.throws(
    () => prepareGalaxyJob({ submission }),
    /extentScale must be finite and between 1 and 4/,
  );
});
