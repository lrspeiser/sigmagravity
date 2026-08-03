import test from "node:test";
import assert from "node:assert/strict";
import { catalog, getSystem } from "../lib/catalog.mjs";
import { FIXED_MOND_FORMULA } from "../lib/formula.mjs";
import {
  createObservedGalaxyRadialTwin,
  createSyntheticGalaxy,
  runHeldoutTwinBenchmark,
  runRotationCurveBenchmark,
} from "../lib/simulator.mjs";

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

test("observed-galaxy twin extraction is deterministic and blind to measured speeds", () => {
  const system = getSystem("DDO154");
  const first = createObservedGalaxyRadialTwin(system);
  const altered = {
    ...system,
    vFlatKmS: 999,
    points: system.points.map((point, index) => ({
      ...point,
      vObsKmS: 900 + index,
      eVObsKmS: 100 + index,
    })),
  };
  const second = createObservedGalaxyRadialTwin(altered);
  assert.equal(first.parameterPackage.contentSha256, second.parameterPackage.contentSha256);
  assert.deepEqual(first.sourcePoints, second.sourcePoints);
  assert.equal(first.parameterPackage.velocityTargetsUsed, false);
  assert.deepEqual(first.parameterPackage.gravityParameters, []);
  assert.ok(first.reconstruction.gBarNormalizedRmse < 0.25);
  assert.equal("vObsKmS" in first.sourcePoints[0], false);
});

test("held-out twin run separates source, formula, and transport errors", () => {
  const result = runHeldoutTwinBenchmark({ system: getSystem("DDO154"), formula: FIXED_MOND_FORMULA });
  assert.equal(result.state, "succeeded");
  assert.equal(result.manifest.twinProtocol.velocityTargetsUsedInExtraction, false);
  assert.equal(result.manifest.parameterAccounting.perObject, 0);
  assert.equal(result.predictions.length, 12);
  assert.ok(Number.isFinite(result.metrics.sourceReconstruction.gBarNormalizedRmse));
  assert.ok(Number.isFinite(result.metrics.formulaOnGeneratedTwin.rmseKmS));
  assert.ok(Number.isFinite(result.metrics.transport.predictionRmseKmS));
  assert.equal(result.metrics.formulaOnGeneratedTwin.pointCount, 12);
});
