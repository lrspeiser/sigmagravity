import test from "node:test";
import assert from "node:assert/strict";
import { FIXED_MOND_FORMULA, validateFormula } from "../lib/formula.mjs";

test("fixed MOND formula is dimensionally valid and executable", () => {
  const validation = validateFormula(FIXED_MOND_FORMULA);
  assert.equal(validation.valid, true, validation.errors.join("; "));
  assert.equal(validation.dimensionAudit.output, "L^1 T^-2");
  assert.deepEqual(validation.parameterAccounting, { universal: 1, perObject: 0, referenced: ["a0"] });
  assert.ok(validation.evaluate({ g_bar: 1e-12 }) > 1e-12);
});

test("dimension mismatch is rejected", () => {
  const validation = validateFormula({
    name: "bad",
    outputUnit: "m/s^2",
    parameters: {},
    parameterPolicy: { universal: true, perObjectParameters: 0 },
    expression: { op: "add", args: [{ input: "g_bar" }, { input: "radius" }] },
  });
  assert.equal(validation.valid, false);
  assert.match(validation.errors.join(" "), /matching dimensions/);
});

test("per-object gravity parameters are rejected", () => {
  const validation = validateFormula({
    ...FIXED_MOND_FORMULA,
    parameterPolicy: { universal: false, perObjectParameters: 1 },
  });
  assert.equal(validation.valid, false);
  assert.match(validation.errors.join(" "), /per-object gravity parameters/);
});
