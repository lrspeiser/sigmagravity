import { sha256 } from "./canonical.mjs";
import { compileFormula, FIXED_MOND_FORMULA } from "./formula.mjs";

const KPC_M = 3.085677581491367e19;
const G_SI = 6.67430e-11;
const MSUN_KG = 1.98847e30;

function baryonicAcceleration(point) {
  const signedGas = point.vGasKmS * Math.abs(point.vGasKmS);
  const velocitySquared = Math.max(
    signedGas + 0.5 * point.vDiskKmS ** 2 + 0.7 * point.vBulgeKmS ** 2,
    0,
  );
  return (velocitySquared * 1e6) / (point.radiusKpc * KPC_M);
}

function surfaceDensity(point) {
  const mSunPerPc2 = Math.max(0, point.sbDisk * 0.5 + point.sbBulge * 0.7);
  const pcM = KPC_M / 1000;
  return (mSunPerPc2 * MSUN_KG) / (pcM ** 2);
}

function predictPoint(point, evaluate) {
  const radiusM = point.radiusKpc * KPC_M;
  const gBar = baryonicAcceleration(point);
  const gPred = evaluate({
    g_bar: gBar,
    radius: radiusM,
    surface_density: surfaceDensity(point),
  });
  if (!Number.isFinite(gPred) || gPred < 0) throw new Error("formula produced a non-finite or negative acceleration");
  return {
    radiusKpc: point.radiusKpc,
    observedKmS: point.vObsKmS,
    uncertaintyKmS: point.eVObsKmS,
    baryonicKmS: Math.sqrt(Math.max(gBar * radiusM, 0)) / 1000,
    predictedKmS: Math.sqrt(gPred * radiusM) / 1000,
    gBarMS2: gBar,
    gPredMS2: gPred,
  };
}

function score(points) {
  const residuals = points.map((point) => point.predictedKmS - point.observedKmS);
  const rmseKmS = Math.sqrt(residuals.reduce((sum, value) => sum + value ** 2, 0) / residuals.length);
  const maeKmS = residuals.reduce((sum, value) => sum + Math.abs(value), 0) / residuals.length;
  const chi2 = points.reduce(
    (sum, point) => sum + ((point.predictedKmS - point.observedKmS) / Math.max(point.uncertaintyKmS, 0.1)) ** 2,
    0,
  );
  return { pointCount: points.length, rmseKmS, maeKmS, chi2, reducedChi2: chi2 / Math.max(points.length - 1, 1) };
}

function evaluateSystem(system, compiled) {
  const predictions = system.points.map((point) => predictPoint(point, compiled.evaluate));
  return { systemId: system.id, metrics: score(predictions), predictions };
}

export function runRotationCurveBenchmark({ systems, formula }) {
  if (!Array.isArray(systems) || systems.length === 0) throw new Error("at least one system is required");
  if (systems.length > 25) throw new Error("interactive runs are limited to 25 systems; use the future batch worker for larger jobs");
  const compiled = compileFormula(formula);
  const mond = compileFormula(FIXED_MOND_FORMULA);
  const newtonian = { evaluate: ({ g_bar: value }) => value };
  const userResults = systems.map((system) => evaluateSystem(system, compiled));
  const mondResults = systems.map((system) => evaluateSystem(system, mond));
  const newtonianResults = systems.map((system) => evaluateSystem(system, newtonian));
  const aggregate = (items) => ({
    systemCount: items.length,
    pointCount: items.reduce((sum, item) => sum + item.metrics.pointCount, 0),
    meanSystemRmseKmS: items.reduce((sum, item) => sum + item.metrics.rmseKmS, 0) / items.length,
    medianSystemRmseKmS: [...items].map((item) => item.metrics.rmseKmS).sort((a, b) => a - b)[Math.floor(items.length / 2)],
  });
  const manifest = {
    serviceVersion: "0.1.0",
    formulaSha256: compiled.formulaSha256,
    formula: compiled.canonicalManifest,
    systemIds: systems.map((system) => system.id),
    datasetRelease: "sparc-2016-v1",
    test: "rotation_curve",
    parameterAccounting: compiled.parameterAccounting,
    assumptions: { diskMassToLight: 0.5, bulgeMassToLight: 0.7, fittedParameters: 0 },
  };
  return {
    id: `run_${sha256(manifest).slice(0, 24)}`,
    state: "succeeded",
    validationStatus: "exploratory",
    manifestSha256: sha256(manifest),
    manifest,
    scores: {
      submitted: aggregate(userResults),
      fixedMond: aggregate(mondResults),
      newtonian: aggregate(newtonianResults),
    },
    results: userResults,
    comparators: { fixedMond: mondResults, newtonian: newtonianResults },
    caveats: [
      "This fast path scores published radial mass-model components; it is not the repository's 2D/3D field solver.",
      "The SPARC sample is public and repeatedly examined, so this run is exploratory rather than blind validation.",
      "No nuisance parameters are fitted by the service.",
    ],
  };
}

function seededRandom(seed) {
  let state = (Number(seed) >>> 0) || 1;
  return () => {
    state = (1664525 * state + 1013904223) >>> 0;
    return state / 4294967296;
  };
}

export function createSyntheticGalaxy(request) {
  const physical = request?.physical ?? {};
  const observation = request?.observation ?? {};
  const seed = Number(request?.seed ?? 1);
  const mass = Number(physical.baryonicMassMsolar);
  const gasFraction = Number(physical.gasFraction);
  const bulgeFraction = Number(physical.bulgeFraction);
  const diskScaleKpc = Number(physical.diskScaleKpc);
  const pointCount = Math.min(96, Math.max(12, Number(observation.pointCount ?? 32)));
  if (![mass, gasFraction, bulgeFraction, diskScaleKpc].every(Number.isFinite)) throw new Error("synthetic physical parameters must be finite");
  if (mass <= 0 || diskScaleKpc <= 0) throw new Error("mass and disk scale must be positive");
  if (gasFraction < 0 || gasFraction > 1 || bulgeFraction < 0 || bulgeFraction > 1 || gasFraction + bulgeFraction > 1) {
    throw new Error("gas and bulge fractions must be between 0 and 1 and sum to at most 1");
  }
  const random = seededRandom(seed);
  const noiseKmS = Math.max(0, Number(observation.noiseKmS ?? 2));
  const points = [];
  for (let index = 0; index < pointCount; index += 1) {
    const radiusKpc = diskScaleKpc * (0.2 + (6.8 * index) / (pointCount - 1));
    const diskFraction = 1 - Math.exp(-radiusKpc / diskScaleKpc) * (1 + radiusKpc / diskScaleKpc);
    const gasScale = diskScaleKpc * 1.7;
    const gasEnclosed = 1 - Math.exp(-radiusKpc / gasScale) * (1 + radiusKpc / gasScale);
    const bulgeScale = diskScaleKpc * 0.18;
    const bulgeEnclosed = radiusKpc ** 2 / (radiusKpc + bulgeScale) ** 2;
    const enclosed = mass * ((1 - gasFraction - bulgeFraction) * diskFraction + gasFraction * gasEnclosed + bulgeFraction * bulgeEnclosed);
    const radiusM = radiusKpc * KPC_M;
    const vBar = Math.sqrt((G_SI * enclosed * MSUN_KG) / radiusM) / 1000;
    const gBar = (vBar * 1000) ** 2 / radiusM;
    const a0 = 1.2e-10;
    const gMond = 0.5 * (gBar + Math.sqrt(gBar ** 2 + 4 * gBar * a0));
    const vIdeal = Math.sqrt(gMond * radiusM) / 1000;
    const noise = noiseKmS * ((random() + random() + random() + random()) - 2);
    points.push({
      radiusKpc,
      vObsKmS: Math.max(0, vIdeal + noise),
      eVObsKmS: Math.max(noiseKmS, 0.5),
      vGasKmS: vBar * Math.sqrt(gasFraction * gasEnclosed / Math.max(enclosed / mass, 1e-12)),
      vDiskKmS: (vBar / Math.sqrt(0.5)) * Math.sqrt((1 - gasFraction - bulgeFraction) * diskFraction / Math.max(enclosed / mass, 1e-12)),
      vBulgeKmS: (vBar / Math.sqrt(0.7)) * Math.sqrt(bulgeFraction * bulgeEnclosed / Math.max(enclosed / mass, 1e-12)),
      sbDisk: 0,
      sbBulge: 0,
    });
  }
  const definition = { seed, physical: { baryonicMassMsolar: mass, gasFraction, bulgeFraction, diskScaleKpc }, observation: { pointCount, noiseKmS } };
  return {
    id: `synthetic_${sha256(definition).slice(0, 16)}`,
    type: "galaxy",
    sampleState: "synthetic",
    supportedTests: ["rotation_curve"],
    generator: "radial-preview-v1",
    definition,
    points,
    caveats: ["This is a deterministic radial preview, not an observation-matched 2D image or velocity-field replica."],
  };
}
