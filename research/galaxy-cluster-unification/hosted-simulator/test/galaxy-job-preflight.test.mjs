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

function conditionedBundle() {
  const value = bundle();
  value.arrays.push(
    {
      key: "gas_surface_density_uncertainty",
      npzKey: "gas_sigma",
      unit: "M_sun/kpc^2",
      rank: "scalar",
      role: "uncertainty",
      dtype: "<f8",
      shape: [65, 65],
      elementCount: 4225,
      contentSha256: "3".repeat(64),
    },
    {
      key: "stellar_surface_density_uncertainty",
      npzKey: "stars_sigma",
      unit: "M_sun/kpc^2",
      rank: "scalar",
      role: "uncertainty",
      dtype: "<f8",
      shape: [65, 65],
      elementCount: 4225,
      contentSha256: "4".repeat(64),
    },
    {
      key: "baryonic_conditioning_mask",
      npzKey: "conditioning_mask",
      unit: "1",
      rank: "scalar",
      role: "mask",
      dtype: "<f8",
      shape: [65, 65],
      elementCount: 4225,
      contentSha256: "5".repeat(64),
    },
  );
  const core = Object.fromEntries(Object.entries(value).filter(([key]) => key !== "bundleSha256"));
  value.bundleSha256 = sha256(core);
  return value;
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

test("preflight normalizes a bounded gravity-independent baryonic uncertainty ensemble", () => {
  const result = prepareGalaxyJob({
    submission: {
      schemaVersion: "sigma-galaxy-job-submit/1",
      operation: "extract_roundtrip",
      dataUploadId: `upload_${"2".repeat(24)}`,
      sourceObservables: { inclinationDeg: 47 },
      uncertaintyEnsemble: {
        enabled: true,
        realizations: 4,
        seed: 19,
        priors: {
          gasMassLnSigma: 0.1,
          stellarMassLnSigma: 0.2,
          distanceScaleLnSigma: 0.04,
          inclinationSigmaDeg: 3,
          warpSigmaDeg: 2,
          coSpatialUnseenBaryonFractionMax: 0.08,
        },
      },
      vertical: { enabled: true, realizations: 2, zCells: 17, seed: 20 },
      outputLicense: { id: "CC0-1.0", redistributionAllowed: true },
    },
    inputBundle: bundle(),
  });
  assert.equal(result.workerRequest.uncertaintyEnsemble.realizations, 4);
  assert.equal(result.workerRequest.uncertaintyEnsemble.priors.gas_mass_ln_sigma, 0.1);
  assert.equal(result.workerRequest.uncertaintyEnsemble.priors.reference_inclination_deg, 47);
  assert.equal(result.resourceEstimate.surfaceRealizations, 4);
  assert.equal(result.resourceEstimate.verticalRealizationsPerSurface, 2);
  assert.ok(result.resourceEstimate.ensembleRawArrayBytes > 0);
  assert.equal(result.parameterAccounting.gravityPerObject, 0);
  assert.match(result.warnings.join(" "), /not a likelihood-derived posterior/);
});

test("uncertainty preflight requires an inclination reference and enforces resource limits", () => {
  const missingReference = {
    schemaVersion: "sigma-galaxy-job-submit/1",
    operation: "extract_roundtrip",
    uncertaintyEnsemble: { enabled: true, priors: { inclinationSigmaDeg: 2 } },
    outputLicense: { id: "CC0-1.0", redistributionAllowed: true },
  };
  assert.throws(
    () => prepareGalaxyJob({ submission: missingReference, inputBundle: bundle() }),
    /requires referenceInclinationDeg/,
  );
  const oversizedBundle = bundle();
  for (const record of oversizedBundle.arrays) {
    record.shape = [513, 513];
    record.elementCount = 513 * 513;
  }
  const core = Object.fromEntries(Object.entries(oversizedBundle).filter(([key]) => key !== "bundleSha256"));
  oversizedBundle.bundleSha256 = sha256(core);
  assert.throws(
    () => prepareGalaxyJob({
      submission: {
        schemaVersion: "sigma-galaxy-job-submit/1",
        operation: "extract_roundtrip",
        uncertaintyEnsemble: { enabled: true, realizations: 16 },
        vertical: { enabled: true, realizations: 8, zCells: 129 },
        outputLicense: { id: "CC0-1.0", redistributionAllowed: true },
      },
      inputBundle: oversizedBundle,
    }),
    /256 MiB raw-array limit/,
  );
});

test("preflight binds a gravity-independent baryonic image likelihood", () => {
  const result = prepareGalaxyJob({
    submission: {
      schemaVersion: "sigma-galaxy-job-submit/1",
      operation: "extract_roundtrip",
      dataUploadId: `upload_${"6".repeat(24)}`,
      uncertaintyEnsemble: {
        enabled: true,
        realizations: 6,
        seed: 99,
        conditioning: {
          enabled: true,
          likelihood: "diagonal_gaussian_surface_density",
          useMask: true,
          minimumValidPixelsPerComponent: 100,
          correlationAreaPixels: 6.5,
        },
      },
      outputLicense: { id: "CC0-1.0", redistributionAllowed: true },
    },
    inputBundle: conditionedBundle(),
  });
  assert.equal(result.workerRequest.uncertaintyEnsemble.conditioning.enabled, true);
  assert.equal(result.workerRequest.uncertaintyEnsemble.conditioning.use_mask, true);
  assert.equal(result.workerRequest.uncertaintyEnsemble.conditioning.correlation_area_pixels, 6.5);
  assert.equal(result.parameterAccounting.gravityPerObject, 0);
  assert.match(result.warnings.join(" "), /Velocity, lensing, and gravity-field targets are forbidden/);
});

test("conditioning refuses missing uncertainty data and generated-only packages", () => {
  const extraction = {
    schemaVersion: "sigma-galaxy-job-submit/1",
    operation: "extract_roundtrip",
    uncertaintyEnsemble: { enabled: true, conditioning: { enabled: true } },
    outputLicense: { id: "CC0-1.0", redistributionAllowed: true },
  };
  assert.throws(
    () => prepareGalaxyJob({ submission: extraction, inputBundle: bundle() }),
    /requires gas_surface_density_uncertainty/,
  );
  assert.throws(
    () => prepareGalaxyJob({
      submission: {
        schemaVersion: "sigma-galaxy-job-submit/1",
        operation: "generate",
        parameterPackage: packageFixture(),
        uncertaintyEnsemble: { enabled: true, conditioning: { enabled: true } },
        outputLicense: { id: "CC0-1.0", redistributionAllowed: true },
      },
    }),
    /available only for extract_roundtrip/,
  );
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
