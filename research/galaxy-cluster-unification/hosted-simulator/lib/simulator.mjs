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
  const withinOneSigmaFraction = points.filter(
    (point) => Math.abs(point.predictedKmS - point.observedKmS) <= Math.max(point.uncertaintyKmS, 0.1),
  ).length / points.length;
  return {
    pointCount: points.length,
    rmseKmS,
    maeKmS,
    chi2,
    reducedChi2: chi2 / Math.max(points.length - 1, 1),
    withinOneSigmaFraction,
  };
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
    serviceVersion: "0.24.0",
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

const TWIN_SOURCE_CHANNELS = ["vGasKmS", "vDiskKmS", "vBulgeKmS", "sbDisk", "sbBulge"];
const TWIN_WITHHELD_CHANNELS = ["vObsKmS", "eVObsKmS", "vFlatKmS", "vFlatErrorKmS"];

function radialCoordinate(radiusKpc, minimumRadiusKpc) {
  return Math.log1p(radiusKpc / minimumRadiusKpc);
}

function encodedChannelValue(channel, value) {
  if (channel.startsWith("v")) return Math.sign(value) * value ** 2;
  return Math.log1p(Math.max(0, value));
}

function decodedChannelValue(channel, value) {
  if (channel.startsWith("v")) return Math.sign(value) * Math.sqrt(Math.abs(value));
  return Math.max(0, Math.expm1(value));
}

function interpolate(knots, coordinate) {
  if (coordinate <= knots[0].coordinate) return knots[0].value;
  if (coordinate >= knots.at(-1).coordinate) return knots.at(-1).value;
  let upper = 1;
  while (knots[upper].coordinate < coordinate) upper += 1;
  const lower = upper - 1;
  const span = knots[upper].coordinate - knots[lower].coordinate;
  const fraction = span === 0 ? 0 : (coordinate - knots[lower].coordinate) / span;
  return knots[lower].value + fraction * (knots[upper].value - knots[lower].value);
}

function solveDense(matrix, vector) {
  const size = vector.length;
  const augmented = matrix.map((row, index) => [...row, vector[index]]);
  for (let column = 0; column < size; column += 1) {
    let pivot = column;
    for (let row = column + 1; row < size; row += 1) {
      if (Math.abs(augmented[row][column]) > Math.abs(augmented[pivot][column])) pivot = row;
    }
    [augmented[column], augmented[pivot]] = [augmented[pivot], augmented[column]];
    const divisor = augmented[column][column];
    if (Math.abs(divisor) <= Number.EPSILON) throw new Error("radial twin basis is singular");
    for (let item = column; item <= size; item += 1) augmented[column][item] /= divisor;
    for (let row = 0; row < size; row += 1) {
      if (row === column) continue;
      const factor = augmented[row][column];
      for (let item = column; item <= size; item += 1) augmented[row][item] -= factor * augmented[column][item];
    }
  }
  return augmented.map((row) => row[size]);
}

function piecewiseLinearWeights(knotCoordinates, coordinate) {
  const weights = Array(knotCoordinates.length).fill(0);
  if (coordinate <= knotCoordinates[0]) { weights[0] = 1; return weights; }
  if (coordinate >= knotCoordinates.at(-1)) { weights[weights.length - 1] = 1; return weights; }
  let upper = 1;
  while (knotCoordinates[upper] < coordinate) upper += 1;
  const lower = upper - 1;
  const fraction = (coordinate - knotCoordinates[lower]) / (knotCoordinates[upper] - knotCoordinates[lower]);
  weights[lower] = 1 - fraction;
  weights[upper] = fraction;
  return weights;
}

function extractChannelKnots(points, channel, coordinates, controlPointCount) {
  const samples = points.map((point, index) => ({
    coordinate: coordinates[index],
    value: encodedChannelValue(channel, point[channel]),
  }));
  if (samples.length <= controlPointCount) return samples;
  const fitValues = (knotCoordinates) => {
    const size = knotCoordinates.length;
    const normal = Array.from({ length: size }, () => Array(size).fill(0));
    const target = Array(size).fill(0);
    for (const sample of samples) {
      const weights = piecewiseLinearWeights(knotCoordinates, sample.coordinate);
      for (let row = 0; row < size; row += 1) {
        target[row] += weights[row] * sample.value;
        for (let column = 0; column < size; column += 1) {
          normal[row][column] += weights[row] * weights[column];
        }
      }
    }
    const trace = normal.reduce((sum, row, index) => sum + row[index], 0);
    const ridge = Math.max(trace * 1e-12 / size, Number.EPSILON);
    for (let index = 0; index < size; index += 1) normal[index][index] += ridge;
    return solveDense(normal, target);
  };
  const knotCoordinates = Array.from(
    { length: controlPointCount },
    (_, index) => coordinates[0] + ((coordinates.at(-1) - coordinates[0]) * index) / (controlPointCount - 1),
  );
  const values = fitValues(knotCoordinates);
  return knotCoordinates.map((coordinate, index) => ({ coordinate, value: values[index] }));
}

function normalizedRmse(reference, generated) {
  const numerator = reference.reduce((sum, value, index) => sum + (generated[index] - value) ** 2, 0);
  const denominator = reference.reduce((sum, value) => sum + value ** 2, 0);
  return Math.sqrt(numerator / Math.max(denominator, Number.EPSILON));
}

function median(values) {
  const sorted = [...values].sort((a, b) => a - b);
  const middle = Math.floor(sorted.length / 2);
  return sorted.length % 2 ? sorted[middle] : 0.5 * (sorted[middle - 1] + sorted[middle]);
}

function sourceReconstructionMetrics(referencePoints, twinPoints) {
  const referenceGBar = referencePoints.map(baryonicAcceleration);
  const twinGBar = twinPoints.map(baryonicAcceleration);
  const scale = Math.max(...referenceGBar.map(Math.abs), Number.EPSILON);
  const meaningful = referenceGBar
    .map((value, index) => ({ value, generated: twinGBar[index] }))
    .filter(({ value }) => Math.abs(value) >= scale * 1e-5);
  const componentNormalizedRmse = Object.fromEntries(TWIN_SOURCE_CHANNELS.map((channel) => [
    channel,
    normalizedRmse(
      referencePoints.map((point) => encodedChannelValue(channel, point[channel])),
      twinPoints.map((point) => encodedChannelValue(channel, point[channel])),
    ),
  ]));
  return {
    radialPointCount: referencePoints.length,
    gBarNormalizedRmse: normalizedRmse(referenceGBar, twinGBar),
    gBarMedianAbsolutePercentError: 100 * median(meaningful.map(
      ({ value, generated }) => Math.abs(generated - value) / Math.abs(value),
    )),
    componentNormalizedRmse,
  };
}

export function createObservedGalaxyRadialTwin(system, options = {}) {
  if (!system?.id || !Array.isArray(system.points) || system.points.length < 2) {
    throw new Error("an observed galaxy with at least two radial points is required");
  }
  const controlPointCount = Math.min(12, Math.max(3, Number(options.controlPointCount ?? 6)));
  const minimumRadiusKpc = Math.min(...system.points.map((point) => point.radiusKpc).filter((value) => value > 0));
  if (!Number.isFinite(minimumRadiusKpc)) throw new Error("twin source radii must contain a positive value");
  const coordinates = system.points.map((point) => radialCoordinate(point.radiusKpc, minimumRadiusKpc));
  const channels = Object.fromEntries(TWIN_SOURCE_CHANNELS.map((channel) => [channel, {
    encoding: channel.startsWith("v") ? "sign(v)*v^2" : "log1p(value)",
    knots: extractChannelKnots(system.points, channel, coordinates, controlPointCount),
  }]));
  const packageCore = {
    schemaVersion: "sigma-radial-twin/1",
    sourceSystemId: system.id,
    generator: "log-radius-piecewise-linear-v1",
    controlPointCount,
    radiusKpc: system.points.map((point) => point.radiusKpc),
    radialCoordinate: { kind: "log1p_radius_over_minimum", minimumRadiusKpc },
    metadata: {
      morphology: system.morphology,
      distanceMpc: system.distanceMpc,
      inclinationDeg: system.inclinationDeg,
      quality: system.quality,
    },
    channels,
    withheldUntilScoring: TWIN_WITHHELD_CHANNELS,
    velocityTargetsUsed: false,
    gravityParameters: [],
  };
  const parameterPackage = { ...packageCore, contentSha256: sha256(packageCore) };
  const sourcePoints = system.points.map((point, index) => Object.fromEntries([
    ["radiusKpc", point.radiusKpc],
    ...TWIN_SOURCE_CHANNELS.map((channel) => [
      channel,
      decodedChannelValue(channel, interpolate(channels[channel].knots, coordinates[index])),
    ]),
  ]));
  const reconstruction = sourceReconstructionMetrics(system.points, sourcePoints);
  return {
    id: `twin_${sha256(parameterPackage).slice(0, 20)}`,
    type: "compressed_radial_baryonic_surrogate",
    sampleState: "generated_from_public_baryonic_profile",
    parameterPackage,
    sourcePoints,
    reconstruction,
    caveats: [
      "This is a compressed one-dimensional baryonic surrogate, not a photorealistic or unique 3D reconstruction.",
      "Observed speeds and their uncertainties were excluded from twin extraction and are used only during final scoring.",
    ],
  };
}

function transportMetrics(measuredPredictions, twinPredictions) {
  const differences = measuredPredictions.map(
    (point, index) => twinPredictions[index].predictedKmS - point.predictedKmS,
  );
  return {
    predictionRmseKmS: Math.sqrt(differences.reduce((sum, value) => sum + value ** 2, 0) / differences.length),
    predictionMaximumAbsoluteDifferenceKmS: Math.max(...differences.map(Math.abs)),
  };
}

export function runHeldoutTwinBenchmark({ system, formula, twinOptions = {} }) {
  const twin = createObservedGalaxyRadialTwin(system, twinOptions);
  const scoringTwin = {
    id: `${system.id}__generated_twin`,
    points: twin.sourcePoints.map((point, index) => ({
      ...point,
      vObsKmS: system.points[index].vObsKmS,
      eVObsKmS: system.points[index].eVObsKmS,
    })),
  };
  const submitted = compileFormula(formula);
  const mond = compileFormula(FIXED_MOND_FORMULA);
  const newtonian = { evaluate: ({ g_bar: value }) => value };
  const submittedMeasured = evaluateSystem(system, submitted);
  const submittedTwin = evaluateSystem(scoringTwin, submitted);
  const fixedMondTwin = evaluateSystem(scoringTwin, mond);
  const newtonianTwin = evaluateSystem(scoringTwin, newtonian);
  const transport = transportMetrics(submittedMeasured.predictions, submittedTwin.predictions);
  const predictions = submittedTwin.predictions.map((point, index) => ({
    radiusKpc: point.radiusKpc,
    observedKmS: point.observedKmS,
    uncertaintyKmS: point.uncertaintyKmS,
    submittedMeasuredBaryonsKmS: submittedMeasured.predictions[index].predictedKmS,
    submittedTwinKmS: point.predictedKmS,
    fixedMondTwinKmS: fixedMondTwin.predictions[index].predictedKmS,
    newtonianTwinKmS: newtonianTwin.predictions[index].predictedKmS,
    submittedTwinResidualKmS: point.predictedKmS - point.observedKmS,
  }));
  const manifest = {
    serviceVersion: "0.24.0",
    test: "heldout_radial_twin_rotation_curve",
    datasetRelease: "sparc-2016-v1",
    systemId: system.id,
    formulaSha256: submitted.formulaSha256,
    formula: submitted.canonicalManifest,
    parameterAccounting: submitted.parameterAccounting,
    twinParameterPackageSha256: twin.parameterPackage.contentSha256,
    twinProtocol: {
      velocityTargetsUsedInExtraction: false,
      withheldUntilScoring: TWIN_WITHHELD_CHANNELS,
      gravityParametersInExtraction: 0,
      controlPointCount: twin.parameterPackage.controlPointCount,
    },
  };
  return {
    id: `twinrun_${sha256(manifest).slice(0, 24)}`,
    type: "heldout_radial_twin_validation",
    state: "succeeded",
    validationStatus: "exploratory_public_spent_validation",
    manifestSha256: sha256(manifest),
    manifest,
    system: {
      id: system.id,
      morphology: system.morphology,
      distanceMpc: system.distanceMpc,
      inclinationDeg: system.inclinationDeg,
      quality: system.quality,
    },
    twin: {
      id: twin.id,
      type: twin.type,
      sampleState: twin.sampleState,
      parameterPackage: twin.parameterPackage,
      reconstruction: twin.reconstruction,
    },
    metrics: {
      sourceReconstruction: twin.reconstruction,
      formulaOnMeasuredBaryons: submittedMeasured.metrics,
      formulaOnGeneratedTwin: submittedTwin.metrics,
      fixedMondOnGeneratedTwin: fixedMondTwin.metrics,
      newtonianOnGeneratedTwin: newtonianTwin.metrics,
      transport,
    },
    predictions,
    caveats: [
      ...twin.caveats,
      "A speed match is joint evidence about baryonic reconstruction and the submitted gravity formula; the report keeps those errors separate.",
      "SPARC is public and repeatedly examined, so this is not a blind holdout.",
      "This radial test does not establish resolved 2D/3D velocity-field, lensing, Solar-System, relativistic, or cosmological validity.",
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
