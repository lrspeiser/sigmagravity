import { canonicalize, sha256 } from "./canonical.mjs";
import { validateArrayBundle } from "./field-job-preflight.mjs";

const COMPONENTS = new Set(["gas", "stars"]);
const GENERATION_KEYS = new Map([
  ["massScale", "mass_scale"],
  ["radialScale", "radial_scale"],
  ["fourierScale", "fourier_scale"],
  ["residualScale", "residual_scale"],
  ["rotationDeg", "rotation_deg"],
  ["centerOffsetKpc", "center_offset_kpc"],
  ["axisRatioScale", "axis_ratio_scale"],
]);

const UNCERTAINTY_PRIORS = new Map([
  ["gasMassLnSigma", ["gas_mass_ln_sigma", 0, 1]],
  ["stellarMassLnSigma", ["stellar_mass_ln_sigma", 0, 1]],
  ["gasRadialScaleLnSigma", ["gas_radial_scale_ln_sigma", 0, 0.5]],
  ["stellarRadialScaleLnSigma", ["stellar_radial_scale_ln_sigma", 0, 0.5]],
  ["angularStructureLnSigma", ["angular_structure_ln_sigma", 0, 1]],
  ["localStructureLnSigma", ["local_structure_ln_sigma", 0, 1]],
  ["centerSigmaKpc", ["center_sigma_kpc", 0, 10]],
  ["rotationSigmaDeg", ["rotation_sigma_deg", 0, 180]],
  ["distanceScaleLnSigma", ["distance_scale_ln_sigma", 0, 0.5]],
  ["inclinationSigmaDeg", ["inclination_sigma_deg", 0, 20]],
  ["warpSigmaDeg", ["warp_sigma_deg", 0, 20]],
  ["coSpatialUnseenBaryonFractionMax", ["co_spatial_unseen_baryon_fraction_max", 0, 0.5]],
]);

function integer(value, fallback, minimum, maximum, label) {
  const result = value === undefined ? fallback : Number(value);
  if (!Number.isInteger(result) || result < minimum || result > maximum) {
    throw new Error(`${label} must be an integer between ${minimum} and ${maximum}`);
  }
  return result;
}

function extractionControls(value = {}) {
  if (!value || typeof value !== "object" || Array.isArray(value)) throw new Error("extractionControls must be an object");
  const allowed = new Set(["radialBins", "maximumFourierMode", "residualFeatureCountPerComponent"]);
  const unknown = Object.keys(value).filter((key) => !allowed.has(key));
  if (unknown.length) throw new Error(`unknown extraction controls: ${unknown.join(", ")}`);
  return {
    radialBins: integer(value.radialBins, 24, 6, 64, "radialBins"),
    maximumFourierMode: integer(value.maximumFourierMode, 4, 0, 8, "maximumFourierMode"),
    residualFeatureCountPerComponent: integer(value.residualFeatureCountPerComponent, 64, 0, 256, "residualFeatureCountPerComponent"),
  };
}

function verticalControls(value = {}) {
  if (!value || typeof value !== "object" || Array.isArray(value)) throw new Error("vertical must be an object");
  const allowed = new Set(["enabled", "realizations", "zCells", "seed"]);
  const unknown = Object.keys(value).filter((key) => !allowed.has(key));
  if (unknown.length) throw new Error(`unknown vertical controls: ${unknown.join(", ")}`);
  const zCells = integer(value.zCells, 33, 9, 129, "zCells");
  if (zCells % 2 === 0) throw new Error("zCells must be odd");
  return {
    enabled: value.enabled === undefined ? true : Boolean(value.enabled),
    realizations: integer(value.realizations, 3, 1, 8, "realizations"),
    zCells,
    seed: integer(value.seed, 0, 0, 2 ** 31 - 1, "seed"),
  };
}

function uncertaintyControls(value = {}, sourceObservables = {}) {
  if (!value || typeof value !== "object" || Array.isArray(value)) throw new Error("uncertaintyEnsemble must be an object");
  const allowed = new Set(["enabled", "realizations", "seed", "priors", "conditioning"]);
  const unknown = Object.keys(value).filter((key) => !allowed.has(key));
  if (unknown.length) throw new Error(`unknown uncertainty ensemble controls: ${unknown.join(", ")}`);
  const priorsInput = value.priors ?? {};
  if (!priorsInput || typeof priorsInput !== "object" || Array.isArray(priorsInput)) throw new Error("uncertaintyEnsemble.priors must be an object");
  const priorAllowed = new Set([...UNCERTAINTY_PRIORS.keys(), "referenceInclinationDeg"]);
  const unknownPriors = Object.keys(priorsInput).filter((key) => !priorAllowed.has(key));
  if (unknownPriors.length) throw new Error(`unknown baryonic uncertainty priors: ${unknownPriors.join(", ")}`);
  const priors = {};
  for (const [inputKey, [workerKey, minimum, maximum]] of UNCERTAINTY_PRIORS) {
    const number = priorsInput[inputKey] === undefined ? 0 : Number(priorsInput[inputKey]);
    if (!Number.isFinite(number) || number < minimum || number > maximum) {
      throw new Error(`${inputKey} must be finite and between ${minimum} and ${maximum}`);
    }
    priors[workerKey] = number;
  }
  const rawReference = priorsInput.referenceInclinationDeg ?? sourceObservables.inclinationDeg;
  if (rawReference === undefined || rawReference === null) {
    priors.reference_inclination_deg = null;
  } else {
    const reference = Number(rawReference);
    if (!Number.isFinite(reference) || reference < 0 || reference > 85) throw new Error("referenceInclinationDeg must be finite and between 0 and 85");
    priors.reference_inclination_deg = reference;
  }
  if (priors.inclination_sigma_deg > 0 && priors.reference_inclination_deg === null) {
    throw new Error("inclinationSigmaDeg requires referenceInclinationDeg or sourceObservables.inclinationDeg");
  }
  const conditioningInput = value.conditioning ?? {};
  if (!conditioningInput || typeof conditioningInput !== "object" || Array.isArray(conditioningInput)) {
    throw new Error("uncertaintyEnsemble.conditioning must be an object");
  }
  const conditioningAllowed = new Set([
    "enabled",
    "likelihood",
    "useMask",
    "minimumValidPixelsPerComponent",
    "correlationAreaPixels",
  ]);
  const unknownConditioning = Object.keys(conditioningInput).filter((key) => !conditioningAllowed.has(key));
  if (unknownConditioning.length) {
    throw new Error(`unknown baryonic conditioning controls: ${unknownConditioning.join(", ")}`);
  }
  const likelihood = conditioningInput.likelihood ?? "diagonal_gaussian_surface_density";
  if (likelihood !== "diagonal_gaussian_surface_density") {
    throw new Error("conditioning.likelihood must be diagonal_gaussian_surface_density");
  }
  const correlationAreaPixels = conditioningInput.correlationAreaPixels === undefined
    ? 1
    : Number(conditioningInput.correlationAreaPixels);
  if (!Number.isFinite(correlationAreaPixels) || correlationAreaPixels < 1 || correlationAreaPixels > 4096) {
    throw new Error("conditioning.correlationAreaPixels must be finite and between 1 and 4096");
  }
  const conditioning = {
    enabled: conditioningInput.enabled === undefined ? false : Boolean(conditioningInput.enabled),
    likelihood,
    use_mask: conditioningInput.useMask === undefined ? false : Boolean(conditioningInput.useMask),
    minimum_valid_pixels_per_component: integer(
      conditioningInput.minimumValidPixelsPerComponent,
      25,
      5,
      1000000,
      "conditioning.minimumValidPixelsPerComponent",
    ),
    correlation_area_pixels: correlationAreaPixels,
  };
  const enabled = value.enabled === undefined ? false : Boolean(value.enabled);
  if (conditioning.enabled && !enabled) {
    throw new Error("baryonic conditioning requires uncertaintyEnsemble.enabled=true");
  }
  return {
    enabled,
    realizations: enabled ? integer(value.realizations, 5, 1, 16, "uncertaintyEnsemble.realizations") : 1,
    seed: integer(value.seed, 0, 0, 2 ** 31 - 1, "uncertaintyEnsemble.seed"),
    priors,
    conditioning,
  };
}

function generationControls(value = {}) {
  if (!value || typeof value !== "object" || Array.isArray(value)) throw new Error("generationControls must be an object");
  const result = {};
  for (const [component, controls] of Object.entries(value)) {
    if (!COMPONENTS.has(component) || !controls || typeof controls !== "object" || Array.isArray(controls)) {
      throw new Error("generationControls may contain only gas and stars objects");
    }
    result[component] = {};
    for (const [key, raw] of Object.entries(controls)) {
      const workerKey = GENERATION_KEYS.get(key);
      if (!workerKey) throw new Error(`unknown ${component} generation control: ${key}`);
      if (key === "centerOffsetKpc") {
        if (!Array.isArray(raw) || raw.length !== 2 || !raw.every(Number.isFinite)) throw new Error(`${component}.centerOffsetKpc requires two finite numbers`);
        result[component][workerKey] = raw.map(Number);
      } else {
        const number = Number(raw);
        if (!Number.isFinite(number)) throw new Error(`${component}.${key} must be finite`);
        if (["massScale", "radialScale", "axisRatioScale"].includes(key) && number <= 0) throw new Error(`${component}.${key} must be positive`);
        if (["fourierScale", "residualScale"].includes(key) && number < 0) throw new Error(`${component}.${key} must be non-negative`);
        result[component][workerKey] = number;
      }
    }
  }
  return result;
}

function outputLicense(value) {
  if (!value || typeof value !== "object" || typeof value.id !== "string" || !value.id || typeof value.redistributionAllowed !== "boolean") {
    throw new Error("outputLicense requires id and redistributionAllowed");
  }
  return { id: value.id, redistributionAllowed: value.redistributionAllowed };
}

function outputGrid(operation, value, defaultCells) {
  if (value === undefined) return null;
  if (operation !== "generate") throw new Error("outputGrid is available only for generate jobs");
  const allowed = new Set(["cellsPerAxis", "extentScale"]);
  if (!value || typeof value !== "object" || Array.isArray(value) || Object.keys(value).some((key) => !allowed.has(key))) {
    throw new Error("outputGrid supports cellsPerAxis and extentScale");
  }
  const cellsPerAxis = integer(value.cellsPerAxis, defaultCells, 9, 513, "outputGrid.cellsPerAxis");
  if (cellsPerAxis % 2 === 0) throw new Error("outputGrid.cellsPerAxis must be odd");
  const extentScale = value.extentScale === undefined ? 1 : Number(value.extentScale);
  if (!Number.isFinite(extentScale) || extentScale < 1 || extentScale > 4) {
    throw new Error("outputGrid.extentScale must be finite and between 1 and 4");
  }
  return { cellsPerAxis, extentScale };
}

function verifyPackage(value) {
  if (!value || typeof value !== "object" || Array.isArray(value)) throw new Error("generate requires parameterPackage");
  if (value.schemaVersion !== "1.0.0" || value.generator !== "radial-fourier-sparse-residual") throw new Error("unsupported parameterPackage schema or generator");
  const core = Object.fromEntries(Object.entries(value).filter(([key]) => key !== "contentSha256"));
  if (value.contentSha256 !== sha256(core)) throw new Error("parameterPackage content hash mismatch");
  if (Object.keys(value.gravityParameters ?? {}).length !== 0 || value.velocityTargetsUsed !== false) throw new Error("parameterPackage must remain gravity-independent");
  const cells = Number(value.grid?.cellsPerAxis);
  if (!Number.isInteger(cells) || cells < 9 || cells > 513) throw new Error("parameterPackage cellsPerAxis is outside 9..513");
  for (const component of COMPONENTS) {
    if (!value.components?.[component]) throw new Error(`parameterPackage is missing ${component}`);
  }
  return value;
}

function extractionInput(bundle, conditioning) {
  const records = validateArrayBundle(bundle);
  if (bundle.geometry?.coordinateSystem !== "cartesian_2d" || bundle.geometry?.dimensions !== 2 || bundle.geometry?.lengthUnit !== "kpc") {
    throw new Error("extract_roundtrip requires cartesian_2d geometry in kpc");
  }
  const gas = records.get("gas_surface_density");
  const stars = records.get("stellar_surface_density");
  if (!gas || !stars) throw new Error("extract_roundtrip requires gas_surface_density and stellar_surface_density");
  for (const [key, record] of [["gas_surface_density", gas], ["stellar_surface_density", stars]]) {
    if (record.rank !== "scalar" || record.unit !== "M_sun/kpc^2") throw new Error(`${key} must be scalar M_sun/kpc^2`);
  }
  if (JSON.stringify(gas.shape) !== JSON.stringify(stars.shape) || gas.shape.length !== 2 || gas.shape[0] !== gas.shape[1]) {
    throw new Error("gas and stellar surface maps must share one square grid");
  }
  if (gas.shape[0] > 513) throw new Error("local galaxy jobs are limited to 513 cells per axis");
  if (conditioning.enabled) {
    for (const [key, label] of [
      ["gas_surface_density_uncertainty", "gas"],
      ["stellar_surface_density_uncertainty", "stellar"],
    ]) {
      const record = records.get(key);
      if (!record) throw new Error(`baryonic conditioning requires ${key}`);
      if (
        record.rank !== "scalar"
        || record.unit !== "M_sun/kpc^2"
        || record.role !== "uncertainty"
        || JSON.stringify(record.shape) !== JSON.stringify(gas.shape)
      ) {
        throw new Error(`${label} conditioning uncertainty must be a scalar M_sun/kpc^2 uncertainty map on the source grid`);
      }
    }
    if (conditioning.use_mask) {
      const mask = records.get("baryonic_conditioning_mask");
      if (
        !mask
        || mask.rank !== "scalar"
        || mask.unit !== "1"
        || mask.role !== "mask"
        || JSON.stringify(mask.shape) !== JSON.stringify(gas.shape)
      ) {
        throw new Error("conditioning.useMask requires baryonic_conditioning_mask as a scalar dimensionless mask on the source grid");
      }
    }
  }
  const rawSpacing = bundle.geometry.spacing;
  const spacing = Array.isArray(rawSpacing) ? rawSpacing.map(Number) : [Number(rawSpacing), Number(rawSpacing)];
  if (spacing.length !== 2 || !spacing.every((item) => Number.isFinite(item) && item > 0) || spacing[0] !== spacing[1]) {
    throw new Error("extract_roundtrip requires equal positive x/y spacing");
  }
  return { shape: gas.shape, spacing };
}

export function prepareGalaxyJob({ submission, inputBundle = null }) {
  if (!submission || typeof submission !== "object" || Array.isArray(submission)) throw new Error("galaxy job submission must be an object");
  if (submission.schemaVersion !== "sigma-galaxy-job-submit/1") throw new Error("galaxy job submission must use sigma-galaxy-job-submit/1");
  if (!["extract_roundtrip", "generate"].includes(submission.operation)) throw new Error("operation must be extract_roundtrip or generate");
  const extraction = extractionControls(submission.extractionControls);
  const generation = generationControls(submission.generationControls);
  const vertical = verticalControls(submission.vertical);
  const uncertainty = uncertaintyControls(
    submission.uncertaintyEnsemble,
    submission.sourceObservables ?? submission.parameterPackage?.sourceObservables ?? {},
  );
  const license = outputLicense(submission.outputLicense);
  let shape;
  let bundleSha256 = null;
  let parameterPackage = null;
  let gridControls = null;
  if (submission.operation === "extract_roundtrip") {
    if (!inputBundle) throw new Error("extract_roundtrip requires a ready data upload");
    ({ shape } = extractionInput(inputBundle, uncertainty.conditioning));
    bundleSha256 = inputBundle.bundleSha256;
  } else {
    if (uncertainty.conditioning.enabled) {
      throw new Error("baryonic conditioning is available only for extract_roundtrip jobs with observed surface-density maps");
    }
    parameterPackage = verifyPackage(submission.parameterPackage);
    gridControls = outputGrid(
      submission.operation,
      submission.outputGrid,
      parameterPackage.grid.cellsPerAxis,
    );
    const outputCells = gridControls?.cellsPerAxis ?? parameterPackage.grid.cellsPerAxis;
    shape = [outputCells, outputCells];
  }
  if (submission.operation === "extract_roundtrip") {
    gridControls = outputGrid(submission.operation, submission.outputGrid, shape[0]);
  }
  const zCells = vertical.enabled ? vertical.zCells : 1;
  const surfaceRealizations = uncertainty.realizations;
  const verticalRealizations = vertical.enabled ? vertical.realizations : 0;
  const ensembleRawArrayBytes = shape[0] * shape[1] * 3 * 8 * (
    surfaceRealizations + surfaceRealizations * verticalRealizations * zCells
  );
  if (ensembleRawArrayBytes > 256 * 1024 ** 2) {
    throw new Error("requested 2D/3D uncertainty ensemble exceeds the 256 MiB raw-array limit");
  }
  const estimatedMemoryBytes = shape[0] * shape[1] * (18 * 8 + zCells * 4 * 8) + 2 * ensembleRawArrayBytes;
  const workerRequest = canonicalize({
    operation: submission.operation,
    galaxy: String(submission.galaxy ?? parameterPackage?.galaxy ?? "uploaded-galaxy"),
    sourceObservables: submission.sourceObservables ?? {},
    extractionControls: extraction,
    generationControls: generation,
    vertical,
    uncertaintyEnsemble: uncertainty,
    outputLicense: license,
    outputGrid: gridControls,
    parameterPackage,
  });
  const core = canonicalize({
    schemaVersion: "sigma-galaxy-job-preflight/1",
    inputBundleSha256: bundleSha256,
    parameterPackageSha256: parameterPackage?.contentSha256 ?? null,
    workerRequest,
  });
  const preflightSha256 = sha256(core);
  return {
    valid: true,
    id: `galaxypreflight_${preflightSha256.slice(0, 24)}`,
    preflightSha256,
    inputBundleSha256: bundleSha256,
    parameterPackageSha256: parameterPackage?.contentSha256 ?? null,
    operation: submission.operation,
    gridShape: shape,
    resourceEstimate: {
      estimatedMemoryBytes,
      ensembleRawArrayBytes,
      surfaceRealizations,
      verticalRealizationsPerSurface: verticalRealizations,
      resourceClass: estimatedMemoryBytes <= 512 * 1024 ** 2 ? "cpu_small" : "cpu_medium",
      estimateOnly: true,
    },
    parameterAccounting: {
      gravityUniversal: 0,
      gravityPerObject: 0,
      baryonicRepresentationValuesAreNotGravityParameters: true,
    },
    workerRequest,
    warnings: [
      "Vertical structure is an assumed prior ensemble, not a unique 3D recovery.",
      uncertainty.conditioning.enabled
        ? "Surface weights use only declared baryonic density maps, uncertainty maps, and an optional mask; vertical structure remains an unconditioned prior."
        : "Baryonic uncertainty draws are declared priors, not a likelihood-derived posterior.",
      "Velocity, lensing, and gravity-field targets are forbidden from baryonic conditioning.",
      "extract_roundtrip scores representation fidelity, not a law of gravity.",
    ],
  };
}
