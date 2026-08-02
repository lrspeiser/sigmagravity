import test from "node:test";
import assert from "node:assert/strict";
import { catalog, getSystem } from "../lib/catalog.mjs";
import { FIXED_MOND_FORMULA } from "../lib/formula.mjs";
import { createSyntheticGalaxy, runRotationCurveBenchmark } from "../lib/simulator.mjs";

test("catalog packages all published SPARC galaxies", () => {
  assert.equal(catalog.systems.length, 175);
  assert.equal(getSystem("DDO154").points.length, 12);
  assert.match(catalog.dataset.sha256, /^[0-9a-f]{64}$/);
});

test("rotation benchmark is deterministic and compares MOND and Newtonian", () => {
  const first = runRotationCurveBenchmark({ systems: [getSystem("DDO154")], formula: FIXED_MOND_FORMULA });
  const second = runRotationCurveBenchmark({ systems: [getSystem("DDO154")], formula: FIXED_MOND_FORMULA });
  assert.equal(first.id, second.id);
  assert.deepEqual(first.scores.submitted, first.scores.fixedMond);
  assert.ok(first.scores.submitted.meanSystemRmseKmS < first.scores.newtonian.meanSystemRmseKmS);
  assert.equal(first.manifest.parameterAccounting.perObject, 0);
});

test("synthetic galaxy generation is seeded and scoreable", () => {
  const request = {
    seed: 42,
    physical: { baryonicMassMsolar: 2e10, gasFraction: 0.35, bulgeFraction: 0.1, diskScaleKpc: 2.4 },
    observation: { pointCount: 24, noiseKmS: 1.5 },
  };
  const first = createSyntheticGalaxy(request);
  const second = createSyntheticGalaxy(request);
  assert.deepEqual(first, second);
  assert.equal(first.points.length, 24);
  const result = runRotationCurveBenchmark({ systems: [first], formula: FIXED_MOND_FORMULA });
  assert.equal(result.state, "succeeded");
});
