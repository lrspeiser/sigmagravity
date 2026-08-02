import { canonicalize, sha256 } from "./canonical.mjs";
import { validateFieldModel } from "./field-model.mjs";
import { validateObservationTargets } from "./observation-target.mjs";

function finitePositive(value, label) {
  const number = Number(value);
  if (!Number.isFinite(number) || number <= 0) throw new Error(`${label} must be finite and positive`);
  return number;
}

function bundleCore(bundle) {
  return Object.fromEntries(Object.entries(bundle).filter(([key]) => key !== "bundleSha256"));
}

export function validateArrayBundle(bundle) {
  if (!bundle || bundle.schemaVersion !== "sigma-array-bundle/1") throw new Error("inputBundle must use sigma-array-bundle/1");
  if (bundle.bundleSha256 !== sha256(bundleCore(bundle))) throw new Error("inputBundle manifest hash mismatch");
  if (!Array.isArray(bundle.arrays) || bundle.arrays.length === 0) throw new Error("inputBundle requires array records");
  const records = new Map();
  for (const record of bundle.arrays) {
    if (typeof record.key !== "string" || records.has(record.key)) throw new Error(`invalid or duplicate input array key: ${record.key}`);
    if (!Array.isArray(record.shape) || record.shape.length < 2 || !record.shape.every((value) => Number.isInteger(value) && value >= 5)) throw new Error(`input array ${record.key} has invalid shape`);
    if (!/^[0-9a-f]{64}$/.test(record.contentSha256 ?? "")) throw new Error(`input array ${record.key} requires a content SHA-256`);
    records.set(record.key, record);
  }
  return records;
}

function resourceEstimate(shape, solvedFields, observableCount) {
  const cellCount = shape.reduce((product, value) => product * value, 1);
  const workingArrays = 12 + 2 * solvedFields + observableCount;
  const estimatedMemoryBytes = cellCount * workingArrays * 8;
  const resourceClass = cellCount <= 65 ** 3 ? "cpu_small" : cellCount <= 129 ** 3 ? "cpu_medium" : "cpu_large";
  return { cellCount, workingArrays, estimatedMemoryBytes, resourceClass, estimateOnly: true };
}

export function prepareFieldJob(payload) {
  if (!payload || typeof payload !== "object" || Array.isArray(payload)) throw new Error("field job payload must be an object");
  const validation = validateFieldModel(payload.model);
  if (!validation.valid) {
    return { valid: false, state: "invalid_model", errors: validation.errors, warnings: validation.warnings };
  }
  const bundle = payload.inputBundle;
  const records = validateArrayBundle(bundle);
  const modelGeometry = payload.model.geometry;
  if (bundle.geometry?.coordinateSystem !== modelGeometry.coordinateSystem || bundle.geometry?.dimensions !== modelGeometry.dimensions) {
    throw new Error("model and inputBundle geometry disagree");
  }
  const requirements = new Map(payload.model.dataRequirements.map((item) => [item.key, item]));
  for (const [key, requirement] of requirements) {
    const record = records.get(key);
    if (!record) throw new Error(`inputBundle is missing required array ${key}`);
    if (record.rank !== requirement.rank || record.unit !== requirement.unit) throw new Error(`input array ${key} rank or unit does not match the model`);
  }
  const sourceShapes = [...requirements.keys()].map((key) => records.get(key).shape);
  const referenceShape = sourceShapes[0];
  if (referenceShape.length !== modelGeometry.dimensions || sourceShapes.some((shape) => JSON.stringify(shape) !== JSON.stringify(referenceShape))) {
    throw new Error("model source arrays must share one grid shape matching geometry dimensions");
  }
  const request = payload.request ?? {};
  if (request.schemaVersion !== "sigma-field-job-request/1") throw new Error("request must use sigma-field-job-request/1");
  const rawSpacing = request.spacing ?? bundle.geometry?.spacing;
  const spacing = Array.isArray(rawSpacing)
    ? rawSpacing.map((value, index) => finitePositive(value, `spacing[${index}]`))
    : Array(modelGeometry.dimensions).fill(finitePositive(rawSpacing, "spacing"));
  if (spacing.length !== modelGeometry.dimensions) throw new Error("spacing requires one value per dimension");
  const observableIds = new Set(payload.model.observables.map((item) => item.id));
  const requestedObservables = [...new Set(request.requestedObservables ?? [...observableIds])].sort();
  const unknown = requestedObservables.filter((id) => !observableIds.has(id));
  if (unknown.length) throw new Error(`unknown requested observables: ${unknown.join(", ")}`);
  const observationTargets = validateObservationTargets({
    targets: request.observationTargets ?? [],
    model: payload.model,
    inputBundle: bundle,
    requestedObservables,
    fieldShape: referenceShape,
  });
  const boundaries = request.boundaryFields ?? {};
  for (const [fieldName, specification] of Object.entries(boundaries)) {
    if (specification && typeof specification === "object" && specification.arrayKey && !records.has(specification.arrayKey)) {
      throw new Error(`boundary ${fieldName} references missing array ${specification.arrayKey}`);
    }
  }
  const solvedFields = Object.values(payload.model.fields).filter((field) => field.role === "solved").length;
  const estimate = resourceEstimate(referenceShape, solvedFields, requestedObservables.length);
  const preflightCore = canonicalize({
    schemaVersion: "sigma-field-job-preflight/1",
    modelSha256: validation.modelSha256,
    inputBundleSha256: bundle.bundleSha256,
    geometry: { ...modelGeometry, spacing, shape: referenceShape },
    boundaryFields: boundaries,
    requestedObservables,
    observationTargets,
    solver: payload.model.solver,
    parameterPolicy: payload.model.parameterPolicy,
    seed: Number.isInteger(request.seed) ? request.seed : 0,
  });
  const preflightSha256 = sha256(preflightCore);
  return {
    valid: true,
    id: `preflight_${preflightSha256.slice(0, 24)}`,
    state: "worker_not_connected",
    preflightSha256,
    modelSha256: validation.modelSha256,
    inputBundleSha256: bundle.bundleSha256,
    geometry: preflightCore.geometry,
    parameterAccounting: validation.parameterAccounting,
    observationTargets,
    requiredCapabilities: validation.requiredCapabilities,
    resourceEstimate: estimate,
    executionReadiness: {
      state: "worker_not_connected",
      blockers: ["array_bytes_not_uploaded", "generic_scientific_worker_not_connected"],
    },
    warnings: validation.warnings,
  };
}
