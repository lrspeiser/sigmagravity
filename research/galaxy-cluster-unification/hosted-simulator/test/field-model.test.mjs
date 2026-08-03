import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import {
  confirmFieldModel,
  MODEL_CONFIRMATION_ACKNOWLEDGEMENT,
  validateFieldModel,
} from "../lib/field-model.mjs";

const examples = new URL("../examples/models/", import.meta.url);
const load = (name) => JSON.parse(readFileSync(new URL(name, examples), "utf8"));

for (const name of ["newtonian-poisson.json", "aqual.json", "qumond.json", "refracted-gravity.json", "nonlocal-response.json", "two-potential.json"]) {
  test(`${name} is represented by the same generic manifest`, () => {
    const first = validateFieldModel(load(name));
    const second = validateFieldModel(load(name));
    assert.equal(first.valid, true, first.errors.join("; "));
    assert.equal(first.modelSha256, second.modelSha256);
    assert.equal(first.executionReadiness.state, "worker_not_connected");
    assert.equal(first.confirmation.confirmed, true);
    assert.equal(first.parameterAccounting.perObject, 0);
  });
}

test("a structurally valid draft cannot execute until its exact hash is confirmed", () => {
  const draft = load("newtonian-poisson.json");
  draft.source.confirmedCanonical = false;
  delete draft.source.confirmedModelSha256;
  const validation = validateFieldModel(draft);
  assert.equal(validation.valid, true);
  assert.equal(validation.confirmation.confirmed, false);
  assert.equal(validation.executionReadiness.state, "awaiting_researcher_confirmation");
  assert.throws(
    () => confirmFieldModel(draft, {
      expectedModelSha256: "f".repeat(64),
      acknowledgement: MODEL_CONFIRMATION_ACKNOWLEDGEMENT,
    }),
    /does not match/,
  );
  const receipt = confirmFieldModel(draft, {
    expectedModelSha256: validation.modelSha256,
    acknowledgement: MODEL_CONFIRMATION_ACKNOWLEDGEMENT,
  });
  assert.equal(receipt.modelSha256, validation.modelSha256);
  assert.equal(receipt.confirmedModel.source.confirmedCanonical, true);
  assert.equal(validateFieldModel(receipt.confirmedModel).confirmation.confirmed, true);
});

test("QUMOND declares its zero-vector flux limit without a theory-specific route", () => {
  const result = validateFieldModel(load("qumond.json"));
  assert.equal(result.valid, true, result.errors.join("; "));
  assert.ok(result.requiredCapabilities.operators.includes("multiply_zero_vector_limit"));
});

test("nonlocal convolution requires an explicit, non-periodic physical integral convention", () => {
  const manifest = load("nonlocal-response.json");
  let result = validateFieldModel(manifest);
  assert.equal(result.valid, true, result.errors.join("; "));
  assert.ok(result.requiredCapabilities.operators.includes("convolution"));
  assert.match(result.warnings.join(" "), /never normalized automatically/);

  delete manifest.solver.nonlocalBoundary;
  result = validateFieldModel(manifest);
  assert.equal(result.valid, false);
  assert.match(result.errors.join(" "), /nonlocalBoundary=zero_padded/);
});

test("dimensionally invalid field equation is rejected", () => {
  const manifest = load("newtonian-poisson.json");
  manifest.equations[0].rhs = { field: "rho_b" };
  const result = validateFieldModel(manifest);
  assert.equal(result.valid, false);
  assert.match(result.errors.join(" "), /equation poisson mismatch/);
});

test("solved fields require explicit boundary conditions", () => {
  const manifest = load("newtonian-poisson.json");
  delete manifest.fields.Phi.boundary;
  const result = validateFieldModel(manifest);
  assert.equal(result.valid, false);
  assert.match(result.errors.join(" "), /requires a supported boundary/);
});

test("per-object parameters are permitted only when explicitly disclosed", () => {
  const manifest = load("refracted-gravity.json");
  manifest.parameters.epsilon0.scope = "per_object";
  let result = validateFieldModel(manifest);
  assert.equal(result.valid, false);
  assert.match(result.errors.join(" "), /must exactly match/);
  manifest.parameterPolicy.mode = "per_object";
  manifest.parameterPolicy.perObjectParameters = ["epsilon0"];
  result = validateFieldModel(manifest);
  assert.equal(result.valid, true, result.errors.join("; "));
  assert.equal(result.parameterAccounting.perObject, 1);
  assert.match(result.warnings.join(" "), /disclosed separately/);
});

test("source arrays must match their declared data rank and units", () => {
  const manifest = load("newtonian-poisson.json");
  manifest.dataRequirements[0].unit = "kg/m^2";
  const result = validateFieldModel(manifest);
  assert.equal(result.valid, false);
  assert.match(result.errors.join(" "), /does not match data requirement/);
});

test("published JSON schema identifies the same manifest version", () => {
  const schema = JSON.parse(readFileSync(new URL("../schemas/model-manifest-v1.schema.json", import.meta.url), "utf8"));
  assert.equal(schema.properties.schemaVersion.const, "sigma-field-model/1");
  assert.equal(schema.$schema, "https://json-schema.org/draft/2020-12/schema");
  assert.deepEqual(schema.$defs.solver.properties.nonlocalBoundary.enum, ["zero_padded"]);
  assert.deepEqual(schema.$defs.solver.properties.convolutionMeasure.enum, ["physical_volume"]);
});

test("generic nonlinear solver controls are validated and disclosed", () => {
  const manifest = load("aqual.json");
  manifest.solver.initialization = "linearized_unit_coefficient";
  manifest.solver.residualTolerance = 1e-8;
  manifest.solver.nonlinearMethod = "anderson";
  manifest.solver.andersonHistory = 5;
  manifest.solver.picardWarmupIterations = 20;
  manifest.solver.picardWarmupDamping = 0.2;
  let result = validateFieldModel(manifest);
  assert.equal(result.valid, true, result.errors.join("; "));
  assert.match(result.warnings.join(" "), /at most 200 nonlinear iterations/);
  manifest.solver.initialization = "formula_specific_magic";
  result = validateFieldModel(manifest);
  assert.equal(result.valid, false);
  assert.match(result.errors.join(" "), /initialization is unsupported/);
  manifest.solver.initialization = "linearized_unit_coefficient";
  manifest.solver.nonlinearMethod = "formula_specific_magic";
  result = validateFieldModel(manifest);
  assert.equal(result.valid, false);
  assert.match(result.errors.join(" "), /nonlinearMethod is unsupported/);
});
