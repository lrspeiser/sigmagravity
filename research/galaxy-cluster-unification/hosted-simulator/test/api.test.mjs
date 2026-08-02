import test from "node:test";
import assert from "node:assert/strict";
import datasets from "../api/v1/datasets.mjs";
import systems from "../api/v1/systems.mjs";
import system from "../api/v1/system.mjs";
import validate from "../api/v1/formulas/validate.mjs";
import runs from "../api/v1/runs.mjs";
import { FIXED_MOND_FORMULA } from "../lib/formula.mjs";

function response() {
  return {
    headers: {},
    statusCode: 200,
    body: null,
    setHeader(name, value) { this.headers[name] = value; },
    status(code) { this.statusCode = code; return this; },
    json(body) { this.body = body; return this; },
    end() { return this; },
  };
}

function call(handler, { method = "GET", query = {}, body = undefined } = {}) {
  const output = response();
  handler({ method, query, body }, output);
  return output;
}

test("catalog API lists and retrieves systems", () => {
  const datasetResponse = call(datasets);
  assert.equal(datasetResponse.statusCode, 200);
  assert.equal(datasetResponse.body.items[0].systemCount, 175);

  const listResponse = call(systems, { query: { q: "DDO", limit: "4" } });
  assert.equal(listResponse.statusCode, 200);
  assert.equal(listResponse.body.items.length, 4);
  assert.ok(listResponse.body.page.total > 4);

  const detailResponse = call(system, { query: { id: "DDO154" } });
  assert.equal(detailResponse.statusCode, 200);
  assert.equal(detailResponse.body.points.length, 12);
});

test("formula API returns canonical safety audit", () => {
  const output = call(validate, { method: "POST", body: FIXED_MOND_FORMULA });
  assert.equal(output.statusCode, 200);
  assert.equal(output.body.valid, true);
  assert.equal(output.body.safetyAudit.arbitraryCodeExecuted, false);
  assert.equal("evaluate" in output.body, false);
});

test("run API produces a content-addressed comparator result", () => {
  const output = call(runs, {
    method: "POST",
    body: { systemIds: ["DDO154"], tests: ["rotation_curve"], formula: FIXED_MOND_FORMULA },
  });
  assert.equal(output.statusCode, 200);
  assert.match(output.body.id, /^run_[0-9a-f]{24}$/);
  assert.equal(output.body.state, "succeeded");
  assert.equal(output.body.manifest.parameterAccounting.perObject, 0);
});

test("heavy solver requests are never replaced with radial proxies", () => {
  const output = call(runs, {
    method: "POST",
    body: { systemIds: ["DDO154"], tests: ["raw_lensing_roots"], formula: FIXED_MOND_FORMULA },
  });
  assert.equal(output.statusCode, 503);
  assert.equal(output.body.error, "worker_not_connected");
});
