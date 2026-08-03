import assert from "node:assert/strict";
import { FIXED_MOND_FORMULA } from "../lib/formula.mjs";
import { sha256 } from "../lib/canonical.mjs";

const base = process.env.SIMULATOR_URL ?? "http://127.0.0.1:4173";

async function request(path, options) {
  const response = await fetch(`${base}${path}`, options);
  const type = response.headers.get("content-type") ?? "";
  const body = type.includes("json") ? await response.json() : await response.text();
  if (!response.ok) throw new Error(`${path} returned ${response.status}: ${JSON.stringify(body)}`);
  return body;
}

const page = await request("/");
assert.match(page, /Put a gravity formula in front of real galaxies/);
assert.match(page, /Create a synthetic radial galaxy/);
assert.match(page, /held-out twin/i);
assert.match(page, /Resolved fake-galaxy evidence/i);
assert.match(page, /Describe a full 2D or 3D theory/);

const guide = await request("/guide.html");
assert.match(guide, /Know what the simulator tested/);
assert.match(guide, /Inputs, outputs, and meaning/);
assert.match(guide, /Use halo maps only for discovery/);
assert.match(guide, /Inverse baryon-to-response discovery/);
assert.match(guide, /hypothesis_generator_not_forward_theory_fit/);
assert.match(guide, /Couple distinct photon and matter potentials/);
assert.match(guide, /bcc7c218/);
assert.match(guide, /A genuinely useful result is a prediction, not a reconstruction/);

const health = await request("/api/v1/health");
assert.equal(health.status, "ok");
assert.equal(health.version, "0.20.0-preview");
assert.equal(health.capabilities.researcherGuide, "available");
assert.equal(health.capabilities.localNonlocalConvolution, "available_in_dev_server");
assert.equal(health.capabilities.localInverseHaloResponseDiscovery, "available_in_dev_server");
assert.equal(health.capabilities.localCoupledTwoPotentialPhotonMatter, "available_in_dev_server");
const inverseSchema = await request("/schemas/inverse-response-job-submit-v1.schema.json");
assert.equal(inverseSchema.properties.schemaVersion.const, "sigma-inverse-response-job-submit/1");
const datasets = await request("/api/v1/datasets");
assert.equal(datasets.items[0].systemCount, 175);
const systems = await request("/api/v1/systems?q=DDO&limit=3");
assert.equal(systems.items.length, 3);
const system = await request("/api/v1/systems/DDO154");
assert.equal(system.points.length, 12);

const post = (body) => ({
  method: "POST",
  headers: { "content-type": "application/json" },
  body: JSON.stringify(body),
});
const validation = await request("/api/v1/formulas/validate", post(FIXED_MOND_FORMULA));
assert.equal(validation.valid, true);
const fieldModel = await request("/examples/models/refracted-gravity.json");
const fieldValidation = await request("/api/v1/models/validate", post(fieldModel));
assert.equal(fieldValidation.valid, true);
assert.equal(fieldValidation.executionReadiness.state, "worker_not_connected");
const nonlocalModel = await request("/examples/models/nonlocal-response.json");
const nonlocalValidation = await request("/api/v1/models/validate", post(nonlocalModel));
assert.equal(nonlocalValidation.valid, true);
assert.ok(nonlocalValidation.requiredCapabilities.operators.includes("convolution"));
const bundleCore = {
  schemaVersion: "sigma-array-bundle/1",
  geometry: { coordinateSystem: "cartesian_3d", dimensions: 3, spacing: [1, 1, 1], lengthUnit: "m" },
  arrays: [{ key: "baryon_density", npzKey: "baryon_density", unit: "kg/m^3", rank: "scalar", role: "source", dtype: "<f8", shape: [17, 17, 17], elementCount: 4913, contentSha256: "3".repeat(64) }],
  provenance: { kind: "smoke_fixture" },
  license: { id: "CC0-1.0", redistributionAllowed: true },
};
const preflight = await request("/api/v1/field-jobs/prepare", post({
  model: fieldModel,
  inputBundle: { ...bundleCore, bundleSha256: sha256(bundleCore) },
  request: { schemaVersion: "sigma-field-job-request/1", requestedObservables: ["massive_tracer_acceleration"] },
}));
assert.equal(preflight.valid, true);
assert.equal(preflight.state, "worker_not_connected");
const synthetic = await request("/api/v1/synthetic-galaxies", post({
  seed: 42,
  physical: { baryonicMassMsolar: 2e10, gasFraction: 0.35, bulgeFraction: 0.1, diskScaleKpc: 2.4 },
  observation: { pointCount: 24, noiseKmS: 1.5 },
}));
assert.equal(synthetic.points.length, 24);
const run = await request("/api/v1/runs", post({
  systemIds: ["DDO154"], tests: ["rotation_curve"], formula: FIXED_MOND_FORMULA,
}));
assert.equal(run.state, "succeeded");
assert.equal(run.results[0].predictions.length, 12);
assert.equal(run.manifest.parameterAccounting.perObject, 0);
const twinRun = await request("/api/v1/twin-runs", post({
  systemId: "DDO154", formula: FIXED_MOND_FORMULA,
}));
assert.equal(twinRun.state, "succeeded");
assert.equal(twinRun.manifest.twinProtocol.velocityTargetsUsedInExtraction, false);
assert.equal(twinRun.predictions.length, 12);
const resolvedEvidence = await request("/api/v1/resolved-twin-evidence?galaxy=NGC3198");
assert.equal(resolvedEvidence.evidenceClass, "precomputed_development_validation_and_final_holdout_result");
assert.equal(resolvedEvidence.systems.length, 1);
assert.equal(resolvedEvidence.systems[0].models.fixed_simple_mond.twinVersusObserved.classification, "consistent");

console.log(JSON.stringify({
  base,
  systems: datasets.items[0].systemCount,
  formulaSha256: validation.formulaSha256,
  runId: run.id,
  manifestSha256: run.manifestSha256,
  submittedRmseKmS: run.scores.submitted.meanSystemRmseKmS,
  fixedMondRmseKmS: run.scores.fixedMond.meanSystemRmseKmS,
  newtonianRmseKmS: run.scores.newtonian.meanSystemRmseKmS,
  twinRunId: twinRun.id,
  twinSourceGBarNormalizedRmse: twinRun.metrics.sourceReconstruction.gBarNormalizedRmse,
  twinFormulaRmseKmS: twinRun.metrics.formulaOnGeneratedTwin.rmseKmS,
  resolvedEvidenceSha256: resolvedEvidence.evidenceSha256,
}, null, 2));
