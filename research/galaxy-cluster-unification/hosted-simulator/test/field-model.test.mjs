import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { validateFieldModel } from "../lib/field-model.mjs";

const examples = new URL("../examples/models/", import.meta.url);
const load = (name) => JSON.parse(readFileSync(new URL(name, examples), "utf8"));

for (const name of ["newtonian-poisson.json", "aqual.json", "qumond.json", "refracted-gravity.json", "two-potential.json"]) {
  test(`${name} is represented by the same generic manifest`, () => {
    const first = validateFieldModel(load(name));
    const second = validateFieldModel(load(name));
    assert.equal(first.valid, true, first.errors.join("; "));
    assert.equal(first.modelSha256, second.modelSha256);
    assert.equal(first.executionReadiness.state, "worker_not_connected");
    assert.equal(first.parameterAccounting.perObject, 0);
  });
}

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
});
