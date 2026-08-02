import { canonicalize, sha256 } from "./canonical.mjs";
import { validateArrayBundle } from "./field-job-preflight.mjs";
import { validateFieldModel } from "./field-model.mjs";
import { validateObservationTargets } from "./observation-target.mjs";

function fieldObservableIds(scientificResult) {
  if (!Array.isArray(scientificResult?.observables) || scientificResult.observables.length === 0) {
    throw new Error("source field result has no observable records");
  }
  return [...new Set(scientificResult.observables.map((record) => String(record.key).split("__axis", 1)[0]))].sort();
}

function estimateResources(scientificResult, observationBundle) {
  const observableElements = scientificResult.observables.reduce(
    (sum, record) => sum + record.shape.reduce((product, value) => product * value, 1),
    0,
  );
  const observationElements = observationBundle.arrays.reduce(
    (sum, record) => sum + record.shape.reduce((product, value) => product * value, 1),
    0,
  );
  const estimatedMemoryBytes = 8 * (observableElements + 6 * observationElements);
  return {
    observableElements,
    observationElements,
    estimatedMemoryBytes,
    resourceClass: estimatedMemoryBytes <= 512 * 1024 ** 2 ? "cpu_small" : "cpu_medium",
    estimateOnly: true,
  };
}

export function prepareObservationEvaluationJob(payload) {
  if (!payload || typeof payload !== "object" || Array.isArray(payload)) {
    throw new Error("observation evaluation payload must be an object");
  }
  const {
    gatewayFieldJob,
    fieldManifest,
    fieldJob,
    model,
    scientificResult,
    observationBundle,
    observationTargets,
    fieldArtifactHashes,
  } = payload;
  if (gatewayFieldJob?.jobType !== "field" || gatewayFieldJob.state !== "succeeded") {
    throw new Error("source gateway field job must be succeeded");
  }
  if (fieldManifest?.schemaVersion !== "sigma-field-run-manifest/1" || fieldManifest.state !== "succeeded") {
    throw new Error("source field manifest must be succeeded");
  }
  const manifestCore = Object.fromEntries(
    Object.entries(fieldManifest).filter(([key]) => !["manifestSha256", "createdAt"].includes(key)),
  );
  if (sha256(manifestCore) !== fieldManifest.manifestSha256) {
    throw new Error("source field manifest hash mismatch");
  }
  if (scientificResult?.schemaVersion !== "sigma-field-result/1" || scientificResult.state !== "succeeded" || scientificResult.converged !== true) {
    throw new Error("source scientific field result must be converged and succeeded");
  }
  if (fieldJob?.schemaVersion !== "sigma-field-job/1") throw new Error("source field job artifact is invalid");
  if (scientificResult.jobId !== fieldJob.id || scientificResult.jobSha256 !== fieldJob.jobSha256) {
    throw new Error("source field job and scientific result disagree");
  }
  if (fieldManifest.jobId !== fieldJob.id || fieldManifest.scientificResultSha256 !== scientificResult.resultSha256) {
    throw new Error("source field manifest and scientific result disagree");
  }
  const validation = validateFieldModel(model);
  if (!validation.valid) throw new Error("source field model no longer validates");
  if (validation.modelSha256 !== fieldJob.modelSha256) throw new Error("source field model hash mismatch");
  validateArrayBundle(observationBundle);
  const observableIds = fieldObservableIds(scientificResult);
  const targets = validateObservationTargets({
    targets: observationTargets,
    model,
    inputBundle: observationBundle,
    requestedObservables: observableIds,
    fieldShape: scientificResult.observables[0].shape,
  });
  if (targets.length === 0) throw new Error("observationTargets must contain at least one target");
  const reference = {
    gatewayJobId: gatewayFieldJob.id,
    manifestSha256: fieldManifest.manifestSha256,
    modelArtifactSha256: fieldArtifactHashes.model,
    jobArtifactSha256: fieldArtifactHashes.job,
    scientificResultArtifactSha256: fieldArtifactHashes.scientificResult,
    observableArchiveSha256: fieldArtifactHashes.observables,
  };
  for (const [name, value] of Object.entries(reference)) {
    if (typeof value !== "string" || !value) throw new Error(`source field reference ${name} is missing`);
  }
  const preflightCore = canonicalize({
    schemaVersion: "sigma-observation-evaluation-job-preflight/1",
    field: {
      ...reference,
      fieldJobId: fieldJob.id,
      fieldJobSha256: fieldJob.jobSha256,
      fieldScientificResultSha256: scientificResult.resultSha256,
      modelSha256: validation.modelSha256,
      geometry: fieldJob.geometry,
      observableIds,
    },
    observationBundleSha256: observationBundle.bundleSha256,
    observationTargets: targets,
  });
  const preflightSha256 = sha256(preflightCore);
  return {
    valid: true,
    id: `observation_preflight_${preflightSha256.slice(0, 24)}`,
    state: "ready_for_local_worker",
    preflightSha256,
    ...preflightCore,
    parameterAccounting: validation.parameterAccounting,
    evaluationAddedGravityParameters: 0,
    resourceEstimate: estimateResources(scientificResult, observationBundle),
    workerRequest: {
      schemaVersion: "sigma-observation-evaluation-job-request/1",
      observationTargets: canonicalize(observationTargets),
    },
  };
}
