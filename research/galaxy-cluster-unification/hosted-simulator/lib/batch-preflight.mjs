import { canonicalize, sha256 } from "./canonical.mjs";
import { prepareFieldJob, validateArrayBundle } from "./field-job-preflight.mjs";
import { validateObservationTargets } from "./observation-target.mjs";

const POLICY_MODES = new Set([
  "published_fixed",
  "universal_fixed",
  "universal_fit",
  "train_validation_holdout",
  "hierarchical",
  "per_object",
]);
const EXECUTABLE_MODES = new Set(["published_fixed", "universal_fixed"]);

function explicitPolicy(value, model, systemIds) {
  if (!value || typeof value !== "object" || Array.isArray(value)) throw new Error("parameterPolicy must be an object");
  if (!POLICY_MODES.has(value.mode)) throw new Error(`unknown parameterPolicy mode: ${value.mode}`);
  const modelMode = model.parameterPolicy?.mode;
  if (modelMode !== value.mode) throw new Error(`batch parameterPolicy ${value.mode} does not match model policy ${modelMode}`);
  const perObjectParameters = [...new Set(value.perObjectParameters ?? [])].sort();
  if (!perObjectParameters.every((item) => typeof item === "string" && item)) throw new Error("perObjectParameters must contain names");
  const modelPerObject = [...new Set(model.parameterPolicy?.perObjectParameters ?? [])].sort();
  if (JSON.stringify(perObjectParameters) !== JSON.stringify(modelPerObject)) throw new Error("batch and model per-object parameter disclosures differ");
  if (["published_fixed", "universal_fixed", "universal_fit", "train_validation_holdout"].includes(value.mode) && perObjectParameters.length) {
    throw new Error(`${value.mode} cannot declare per-object parameters`);
  }
  const result = { mode: value.mode, perObjectParameters };
  if (value.mode === "train_validation_holdout") {
    const splits = value.splits;
    if (!splits || typeof splits !== "object") throw new Error("train_validation_holdout requires splits");
    const normalized = {};
    const seen = new Set();
    for (const name of ["train", "validation", "holdout"]) {
      if (!Array.isArray(splits[name]) || splits[name].length === 0) throw new Error(`${name} split must be non-empty`);
      normalized[name] = [...splits[name]];
      for (const id of normalized[name]) {
        if (!systemIds.includes(id) || seen.has(id)) throw new Error(`invalid or repeated split system: ${id}`);
        seen.add(id);
      }
    }
    if (seen.size !== systemIds.length) throw new Error("train/validation/holdout splits must cover every system exactly once");
    result.splits = normalized;
  }
  if (value.mode === "hierarchical") {
    if (!Array.isArray(value.populationParameters) || value.populationParameters.length === 0) throw new Error("hierarchical requires populationParameters");
    result.populationParameters = [...new Set(value.populationParameters)].sort();
  }
  return result;
}

export function prepareBatch({ submission, resolvedSystems }) {
  if (!submission || typeof submission !== "object" || Array.isArray(submission)) throw new Error("batch submission must be an object");
  if (submission.schemaVersion !== "sigma-batch-submit/1") throw new Error("batch submission must use sigma-batch-submit/1");
  if (!Array.isArray(resolvedSystems) || resolvedSystems.length === 0) throw new Error("batch requires resolved systems");
  if (resolvedSystems.length > 1000) throw new Error("batch contract supports at most 1000 systems per submission");
  const ids = resolvedSystems.map((item) => item.id);
  if (ids.some((id) => typeof id !== "string" || !id) || new Set(ids).size !== ids.length) throw new Error("batch system IDs must be non-empty and unique");
  const policy = explicitPolicy(submission.parameterPolicy, submission.model, ids);
  const fieldRequest = submission.fieldRequest ?? {};
  const childPreflights = resolvedSystems.map((system) => {
    const fieldPreflight = prepareFieldJob({
      model: submission.model,
      inputBundle: system.inputBundle,
      request: fieldRequest,
    });
    if (!fieldPreflight.valid) throw new Error(`system ${system.id} failed field-only validation: ${fieldPreflight.errors.join("; ")}`);
    const observationBundle = system.observationBundle ?? system.inputBundle;
    validateArrayBundle(observationBundle);
    const requestedObservables = [...new Set(
      fieldRequest.requestedObservables ?? submission.model.observables.map((item) => item.id),
    )].sort();
    const observationTargetSummary = validateObservationTargets({
      targets: system.observationTargets ?? [],
      model: submission.model,
      inputBundle: observationBundle,
      requestedObservables,
      fieldShape: fieldPreflight.geometry.shape,
    });
    return {
      systemId: system.id,
      parentSystemId: system.parentSystemId ?? system.id,
      realization: system.realization ?? null,
      source: system.source,
      inputBundleSha256: system.inputBundle.bundleSha256,
      observationBundleSha256: observationBundle.bundleSha256,
      fieldPreflightSha256: fieldPreflight.preflightSha256,
      observationBindingSha256: sha256({
        observationBundleSha256: observationBundle.bundleSha256,
        observationTargets: canonicalize(system.observationTargets ?? []),
      }),
      resourceEstimate: fieldPreflight.resourceEstimate,
      observationTargets: canonicalize(system.observationTargets ?? []),
      observationTargetSummary,
    };
  });
  const totalEstimatedMemoryBytes = childPreflights.reduce(
    (sum, item) => sum + item.resourceEstimate.estimatedMemoryBytes,
    0,
  );
  const modelSha256 = prepareFieldJob({
    model: submission.model,
    inputBundle: resolvedSystems[0].inputBundle,
    request: fieldRequest,
  }).modelSha256;
  const core = canonicalize({
    schemaVersion: "sigma-batch-preflight/2",
    modelSha256,
    parameterPolicy: policy,
    fieldRequest,
    systems: childPreflights.map(({ systemId, parentSystemId, realization, source, inputBundleSha256, observationBundleSha256, fieldPreflightSha256, observationBindingSha256, observationTargets, observationTargetSummary }) => ({
      systemId,
      parentSystemId,
      realization,
      source,
      inputBundleSha256,
      observationBundleSha256,
      fieldPreflightSha256,
      observationBindingSha256,
      observationTargets,
      observationTargetSummary,
    })),
  });
  const preflightSha256 = sha256(core);
  const executable = EXECUTABLE_MODES.has(policy.mode);
  const scoredObservationTargets = childPreflights.reduce(
    (sum, item) => sum + item.observationTargetSummary.filter((target) => target.scored).length,
    0,
  );
  return {
    valid: true,
    id: `batchpreflight_${preflightSha256.slice(0, 24)}`,
    preflightSha256,
    modelSha256,
    parameterPolicy: policy,
    systemCount: resolvedSystems.length,
    systems: childPreflights,
    resourceEstimate: {
      totalEstimatedMemoryBytes,
      maximumChildMemoryBytes: Math.max(...childPreflights.map((item) => item.resourceEstimate.estimatedMemoryBytes)),
      estimateOnly: true,
    },
    executionReadiness: {
      executable,
      blockers: executable ? [] : [`parameter fitting for ${policy.mode} is not implemented`],
    },
    scoredObservationTargets,
    claimBoundary: scoredObservationTargets
      ? [
        "Massive-tracer and photon-lensing scores are produced by separately content-addressed observation jobs after each immutable field solve.",
        "Changing a measured array, mask, uncertainty, beam, or target declaration does not alter the field-child identity when the model and source field inputs are unchanged.",
        "Velocity, deflection, and reduced-shear residuals remain in separate named channels and units; unsubmitted observables are not evaluated.",
        "Ensemble summaries, when present, measure prediction spread under declared baryonic priors; they are not likelihood-derived posterior intervals.",
      ]
      : [
        "A converged batch proves execution of one frozen model, not agreement with observations.",
        "Rotation-curve and lensing scores require explicit theory-to-observable adapters and target uncertainties.",
      ],
  };
}
