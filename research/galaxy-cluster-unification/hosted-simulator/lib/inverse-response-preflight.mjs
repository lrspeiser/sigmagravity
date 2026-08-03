import { canonicalize, sha256 } from "./canonical.mjs";
import { validateArrayBundle } from "./field-job-preflight.mjs";

const SYSTEM_ID = /^[A-Za-z0-9_.-]{1,64}$/;

function integer(value, fallback, minimum, maximum, label) {
  const result = value === undefined ? fallback : value;
  if (!Number.isInteger(result) || result < minimum || result > maximum) {
    throw new Error(`${label} must be an integer from ${minimum} to ${maximum}`);
  }
  return result;
}

function finiteNonnegative(value, fallback, label) {
  const result = value === undefined ? fallback : Number(value);
  if (!Number.isFinite(result) || result < 0) {
    throw new Error(`${label} must be finite and non-negative`);
  }
  return result;
}

function outputLicense(value) {
  if (!value || typeof value !== "object" || Array.isArray(value)
      || typeof value.id !== "string" || !value.id
      || typeof value.redistributionAllowed !== "boolean") {
    throw new Error("outputLicense requires id and redistributionAllowed");
  }
  return { id: value.id, redistributionAllowed: value.redistributionAllowed };
}

function requireRecord(records, key, scientificRole, operationalRole) {
  const record = records.get(key);
  if (!record) throw new Error(`inputBundle is missing array ${key}`);
  if (record.rank !== "scalar") throw new Error(`array ${key} must declare rank=scalar`);
  if (record.scientificRole !== scientificRole) {
    throw new Error(`array ${key} must declare scientificRole=${scientificRole}`);
  }
  if (record.role !== operationalRole) {
    throw new Error(`array ${key} must declare role=${operationalRole}`);
  }
  return record;
}

export function prepareInverseResponseJob({ submission, inputBundle }) {
  if (!submission || typeof submission !== "object" || Array.isArray(submission)) {
    throw new Error("inverse response submission must be an object");
  }
  if (submission.schemaVersion !== "sigma-inverse-response-job-submit/1") {
    throw new Error("inverse response submission must use sigma-inverse-response-job-submit/1");
  }
  const records = validateArrayBundle(inputBundle);
  const dimensions = inputBundle.geometry?.dimensions;
  const coordinateSystem = inputBundle.geometry?.coordinateSystem;
  const expectedCoordinate = dimensions === 2 ? "cartesian_2d" : dimensions === 3 ? "cartesian_3d" : null;
  if (!expectedCoordinate || coordinateSystem !== expectedCoordinate) {
    throw new Error("inverse response requires Cartesian 2D or 3D bundle geometry");
  }
  const rawSpacing = inputBundle.geometry?.spacing;
  const spacing = Array.isArray(rawSpacing) ? rawSpacing.map(Number) : Array(dimensions).fill(Number(rawSpacing));
  if (spacing.length !== dimensions || !spacing.every((value) => Number.isFinite(value) && value > 0)) {
    throw new Error("bundle spacing requires one positive value per dimension");
  }

  if (!Array.isArray(submission.systems) || submission.systems.length === 0) {
    throw new Error("systems must be a non-empty array");
  }
  const ids = new Set();
  let referenceShape = null;
  const systems = submission.systems.map((raw) => {
    if (!raw || typeof raw !== "object" || Array.isArray(raw) || !SYSTEM_ID.test(raw.id ?? "") || ids.has(raw.id)) {
      throw new Error("system ids must be unique and use 1-64 letters, numbers, dots, underscores, or hyphens");
    }
    ids.add(raw.id);
    const source = requireRecord(records, raw.sourceKey, "baryonic_input", "source");
    const target = requireRecord(records, raw.targetKey, "model_derived_discovery_target", "auxiliary");
    const uncertainty = requireRecord(records, raw.uncertaintyKey, "nuisance_or_calibration", "uncertainty");
    if (source.unit !== target.unit || target.unit !== uncertainty.unit) {
      throw new Error(`system ${raw.id} source, target, and uncertainty units must match`);
    }
    if (source.shape.length !== dimensions
        || JSON.stringify(source.shape) !== JSON.stringify(target.shape)
        || JSON.stringify(source.shape) !== JSON.stringify(uncertainty.shape)) {
      throw new Error(`system ${raw.id} arrays must share the declared spatial grid`);
    }
    if (referenceShape === null) referenceShape = source.shape;
    if (JSON.stringify(referenceShape) !== JSON.stringify(source.shape)) {
      throw new Error("all systems must share one grid shape in inverse-response v1");
    }
    let maskKey;
    if (raw.maskKey !== undefined) {
      const mask = requireRecord(records, raw.maskKey, "nuisance_or_calibration", "mask");
      if (!["1", "dimensionless"].includes(mask.unit)
          || JSON.stringify(mask.shape) !== JSON.stringify(source.shape)) {
        throw new Error(`system ${raw.id} mask must be dimensionless on the source grid`);
      }
      maskKey = raw.maskKey;
    }
    return canonicalize({
      id: raw.id,
      sourceKey: raw.sourceKey,
      targetKey: raw.targetKey,
      uncertaintyKey: raw.uncertaintyKey,
      ...(maskKey ? { maskKey } : {}),
    });
  });

  const kernel = submission.kernel ?? {};
  if (!Array.isArray(kernel.shape) || kernel.shape.length !== dimensions) {
    throw new Error("kernel.shape requires one odd integer per map dimension");
  }
  const kernelShape = kernel.shape.map((value) => {
    if (!Number.isInteger(value) || value < 3 || value % 2 === 0) {
      throw new Error("kernel.shape values must be odd integers of at least three");
    }
    return value;
  });
  if (kernelShape.some((value, index) => value > referenceShape[index])) {
    throw new Error("kernel.shape cannot exceed the submitted map grid");
  }
  const multipliers = kernel.regularizationMultipliers ?? [0.1, 1, 10];
  if (!Array.isArray(multipliers) || multipliers.length === 0
      || !multipliers.every((value) => Number.isFinite(value) && value > 0)) {
    throw new Error("kernel.regularizationMultipliers must contain positive finite numbers");
  }
  if (kernel.nonnegative !== undefined && typeof kernel.nonnegative !== "boolean") {
    throw new Error("kernel.nonnegative must be a boolean");
  }
  const normalizedKernel = {
    shape: kernelShape,
    ridge: finiteNonnegative(kernel.ridge, 1e-8, "kernel.ridge"),
    smoothness: finiteNonnegative(kernel.smoothness, 1e-4, "kernel.smoothness"),
    nonnegative: kernel.nonnegative ?? true,
    regularizationMultipliers: multipliers.map(Number),
  };
  const uncertainty = {
    ensembleSize: integer(submission.uncertainty?.ensembleSize, 32, 20, 512, "uncertainty.ensembleSize"),
    seed: integer(submission.uncertainty?.seed, 0, 0, 2 ** 31 - 1, "uncertainty.seed"),
  };
  const nullControls = {
    kind: submission.nullControls?.kind ?? "source_radial_angle_shuffle",
    count: integer(submission.nullControls?.count, 19, 19, 999, "nullControls.count"),
    seed: integer(submission.nullControls?.seed, 1, 0, 2 ** 31 - 1, "nullControls.seed"),
  };
  if (nullControls.kind !== "source_radial_angle_shuffle") {
    throw new Error("nullControls.kind must be source_radial_angle_shuffle in v1");
  }
  const license = outputLicense(submission.outputLicense);
  const kernelCells = kernelShape.reduce((productValue, value) => productValue * value, 1);
  const mapCells = referenceShape.reduce((productValue, value) => productValue * value, 1);
  const fits = 1 + uncertainty.ensembleSize + nullControls.count + multipliers.length;
  const designBytes = submission.systems.length * mapCells * kernelCells * 8;
  const estimatedMemoryBytes = designBytes + fits * kernelCells * 8 + submission.systems.length * mapCells * 8 * 8;
  const workerRequest = canonicalize({
    systems,
    kernel: normalizedKernel,
    uncertainty,
    nullControls,
    outputLicense: license,
  });
  const core = canonicalize({
    schemaVersion: "sigma-inverse-response-job-preflight/1",
    inputBundleSha256: inputBundle.bundleSha256,
    geometry: { coordinateSystem, dimensions, spacing, shape: referenceShape },
    workerRequest,
  });
  const preflightSha256 = sha256(core);
  return {
    valid: true,
    id: `inversepreflight_${preflightSha256.slice(0, 24)}`,
    preflightSha256,
    inputBundleSha256: inputBundle.bundleSha256,
    geometry: core.geometry,
    systemCount: systems.length,
    resourceEstimate: {
      kernelCells,
      mapCellsPerSystem: mapCells,
      estimatedFits: fits,
      estimatedMemoryBytes,
      resourceClass: estimatedMemoryBytes <= 512 * 1024 ** 2 ? "cpu_small" : estimatedMemoryBytes <= 2 * 1024 ** 3 ? "cpu_medium" : "cpu_large",
      estimateOnly: true,
    },
    parameterAccounting: {
      fittedDiscoveryKernelCells: kernelCells,
      fittedUniversalResponseAmplitudes: 1,
      fittedPerSystemGravityParameters: 0,
      classification: "hypothesis_generator_not_forward_theory_fit",
    },
    dataRoleAudit: systems.map((system) => ({
      systemId: system.id,
      sourceRole: "baryonic_input",
      targetRole: "model_derived_discovery_target",
      uncertaintyRole: "nuisance_or_calibration",
      heldOutRawObservationsUsed: false,
    })),
    workerRequest,
    warnings: [
      "The target is model-derived and may generate hypotheses; it cannot validate the recovered kernel.",
      "The kernel must be frozen before predicting withheld raw observations.",
      "Rank, regularization sensitivity, uncertainty intervals, and radial-angle nulls remain part of the result.",
    ],
  };
}
