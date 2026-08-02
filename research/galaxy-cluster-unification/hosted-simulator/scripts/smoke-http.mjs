import assert from "node:assert/strict";
import { FIXED_MOND_FORMULA } from "../lib/formula.mjs";

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

const health = await request("/api/v1/health");
assert.equal(health.status, "ok");
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

console.log(JSON.stringify({
  base,
  systems: datasets.items[0].systemCount,
  formulaSha256: validation.formulaSha256,
  runId: run.id,
  manifestSha256: run.manifestSha256,
  submittedRmseKmS: run.scores.submitted.meanSystemRmseKmS,
  fixedMondRmseKmS: run.scores.fixedMond.meanSystemRmseKmS,
  newtonianRmseKmS: run.scores.newtonian.meanSystemRmseKmS,
}, null, 2));
