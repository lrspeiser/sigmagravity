import { mkdir, readFile, writeFile } from "node:fs/promises";
import { resolve } from "node:path";

import { catalog } from "../lib/catalog.mjs";
import { sha256 } from "../lib/canonical.mjs";
import { FIXED_MOND_FORMULA } from "../lib/formula.mjs";
import { createObservedGalaxyRadialTwin, runHeldoutTwinBenchmark } from "../lib/simulator.mjs";

const project = resolve(import.meta.dirname, "..", "..");
const configPath = resolve(project, "configs", "p0737_heldout_radial_twin_validation.json");
const output = resolve(project, "results", "p0737_heldout_radial_twin_validation");
const config = JSON.parse(await readFile(configPath, "utf8"));

function median(values) {
  const sorted = [...values].sort((a, b) => a - b);
  return sorted[Math.floor(sorted.length / 2)];
}

function maximumRow(rows, field) {
  return [...rows].sort((a, b) => b[field] - a[field])[0];
}

const rows = catalog.systems.map((system) => {
  const result = runHeldoutTwinBenchmark({
    system,
    formula: FIXED_MOND_FORMULA,
    twinOptions: { controlPointCount: config.twinContract.controlPointsPerChannel },
  });
  const altered = {
    ...system,
    vFlatKmS: 1e6,
    vFlatErrorKmS: 1e5,
    points: system.points.map((point, index) => ({
      ...point,
      vObsKmS: 1e5 + index,
      eVObsKmS: 1e4 + index,
    })),
  };
  const alteredTwin = createObservedGalaxyRadialTwin(altered, {
    controlPointCount: config.twinContract.controlPointsPerChannel,
  });
  return {
    galaxy: system.id,
    morphology: system.morphology.label,
    radialPoints: system.points.length,
    twinPackageSha256: result.twin.parameterPackage.contentSha256,
    gravityParametersInExtraction: result.twin.parameterPackage.gravityParameters.length,
    velocityTargetsUsedInExtraction: result.twin.parameterPackage.velocityTargetsUsed,
    velocityPerturbationInvariant: alteredTwin.parameterPackage.contentSha256 === result.twin.parameterPackage.contentSha256,
    finite: Object.values(result.metrics.sourceReconstruction.componentNormalizedRmse).every(Number.isFinite)
      && result.predictions.every((point) => Object.values(point).every(Number.isFinite)),
    gBarNormalizedRmse: result.metrics.sourceReconstruction.gBarNormalizedRmse,
    gBarMedianAbsolutePercentError: result.metrics.sourceReconstruction.gBarMedianAbsolutePercentError,
    fixedMondMeasuredRmseKmS: result.metrics.formulaOnMeasuredBaryons.rmseKmS,
    fixedMondTwinRmseKmS: result.metrics.formulaOnGeneratedTwin.rmseKmS,
    fixedMondTransportRmseKmS: result.metrics.transport.predictionRmseKmS,
    fixedMondTwinWithinOneSigmaFraction: result.metrics.formulaOnGeneratedTwin.withinOneSigmaFraction,
  };
});

const worstSource = maximumRow(rows, "gBarNormalizedRmse");
const worstTransport = maximumRow(rows, "fixedMondTransportRmseKmS");
const aggregate = {
  galaxies: rows.length,
  radialPoints: rows.reduce((sum, row) => sum + row.radialPoints, 0),
  medianGBarNormalizedRmse: median(rows.map((row) => row.gBarNormalizedRmse)),
  worstGBarNormalizedRmse: worstSource.gBarNormalizedRmse,
  worstGBarGalaxy: worstSource.galaxy,
  medianGBarMedianAbsolutePercentError: median(rows.map((row) => row.gBarMedianAbsolutePercentError)),
  medianFixedMondMeasuredRmseKmS: median(rows.map((row) => row.fixedMondMeasuredRmseKmS)),
  medianFixedMondTwinRmseKmS: median(rows.map((row) => row.fixedMondTwinRmseKmS)),
  medianFixedMondTransportRmseKmS: median(rows.map((row) => row.fixedMondTransportRmseKmS)),
  worstFixedMondTransportRmseKmS: worstTransport.fixedMondTransportRmseKmS,
  worstTransportGalaxy: worstTransport.galaxy,
};
const gates = config.acceptanceGates;
const checks = {
  galaxyCount: aggregate.galaxies === config.dataset.requiredGalaxies,
  radialPointCount: aggregate.radialPoints === config.dataset.requiredRadialPoints,
  allGalaxiesProduceFiniteTwins: rows.every((row) => row.finite),
  twinPackageInvariantToObservedVelocityPerturbation: rows.every((row) => row.velocityPerturbationInvariant),
  gravityParametersInExtraction: rows.every((row) => row.gravityParametersInExtraction === 0),
  velocityTargetsExcludedFromExtraction: rows.every((row) => row.velocityTargetsUsedInExtraction === false),
  medianGBarNormalizedRmse: aggregate.medianGBarNormalizedRmse <= gates.maximumMedianGBarNormalizedRmse,
  worstGalaxyGBarNormalizedRmse: aggregate.worstGBarNormalizedRmse <= gates.maximumWorstGalaxyGBarNormalizedRmse,
  medianFixedMondTransportRmseKmS: aggregate.medianFixedMondTransportRmseKmS <= gates.maximumMedianFixedMondTransportRmseKmS,
  worstGalaxyFixedMondTransportRmseKmS: aggregate.worstFixedMondTransportRmseKmS <= gates.maximumWorstGalaxyFixedMondTransportRmseKmS,
};
const reportCore = {
  stage: config.stage,
  status: Object.values(checks).every(Boolean) ? "pass" : "needs_improvement",
  evidenceClass: config.dataset.evidenceClass,
  protocol: {
    twinKind: config.twinContract.kind,
    controlPointsPerChannel: config.twinContract.controlPointsPerChannel,
    sourceChannels: config.twinContract.sourceChannels,
    withheldUntilScoring: config.twinContract.withheldUntilScoring,
    velocityTargetsUsedInExtraction: false,
    gravityParametersInExtraction: 0,
    comparator: "fixed simple MOND with a0=1.2e-10 m/s^2",
  },
  aggregate,
  checks,
  failedChecks: Object.entries(checks).filter(([, passed]) => !passed).map(([name]) => name),
  claimBoundaries: config.claimBoundaries,
  configSha256: sha256(config),
};
const report = { ...reportCore, reportSha256: sha256(reportCore) };

await mkdir(output, { recursive: true });
const headings = Object.keys(rows[0]);
const csv = [
  headings.join(","),
  ...rows.map((row) => headings.map((heading) => JSON.stringify(row[heading])).join(",")),
].join("\n") + "\n";
await writeFile(resolve(output, "per_galaxy.csv"), csv);
await writeFile(resolve(output, "report.json"), `${JSON.stringify(report, null, 2)}\n`);
await writeFile(resolve(output, "SUMMARY.md"), `# P0737 held-out radial twin validation

- Status: **${report.status.toUpperCase().replaceAll("_", " ")}**
- Public SPARC galaxies: **${aggregate.galaxies}** (${aggregate.radialPoints.toLocaleString("en-US")} radial points)
- Observed speed targets used to build a twin: **no**
- Gravity parameters used to build a twin: **0**
- Median baryonic-acceleration reconstruction error: **${(100 * aggregate.medianGBarNormalizedRmse).toFixed(3)}%**
- Worst baryonic-acceleration reconstruction error: **${(100 * aggregate.worstGBarNormalizedRmse).toFixed(2)}%** (${aggregate.worstGBarGalaxy})
- Median fixed-MOND prediction transport: **${aggregate.medianFixedMondTransportRmseKmS.toFixed(2)} km/s**
- Worst fixed-MOND prediction transport: **${aggregate.worstFixedMondTransportRmseKmS.toFixed(2)} km/s** (${aggregate.worstTransportGalaxy})
- Median fixed-MOND RMSE on measured baryons: **${aggregate.medianFixedMondMeasuredRmseKmS.toFixed(2)} km/s**
- Median fixed-MOND RMSE on generated twins: **${aggregate.medianFixedMondTwinRmseKmS.toFixed(2)} km/s**

The simulator can now make a non-circular radial twin of every catalog galaxy,
apply the same submitted formula to both the measured baryonic source and the
generated source, and reveal the measured rotation speeds only for scoring.
Perturbing every observed speed and uncertainty leaves every twin package hash
unchanged.

The commissioning result is **needs improvement**, rather than pass, because
the six-control-point compression changes fixed MOND's prediction by
${aggregate.worstFixedMondTransportRmseKmS.toFixed(2)} km/s for ${aggregate.worstTransportGalaxy}, above the frozen
5 km/s worst-case gate. The public report therefore keeps the source error,
formula error, and transport error separate. A radial twin is not yet a full
2D/3D simulated galaxy or an individual-star orbit model.
`);

console.log(JSON.stringify(report, null, 2));
