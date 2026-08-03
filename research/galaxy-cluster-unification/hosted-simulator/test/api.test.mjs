import test from "node:test";
import assert from "node:assert/strict";
import datasets from "../api/v1/datasets.mjs";
import health from "../api/v1/health.mjs";
import storageReadiness from "../api/v1/storage-readiness.mjs";
import queueCanary from "../api/v1/queue-canary.mjs";
import systems from "../api/v1/systems.mjs";
import system from "../api/v1/system.mjs";
import validate from "../api/v1/formulas/validate.mjs";
import validateModel from "../api/v1/models/validate.mjs";
import confirmModel from "../api/v1/models/confirm.mjs";
import prepareFieldJob from "../api/v1/field-jobs/prepare.mjs";
import hostedFieldJobs from "../api/v1/field-jobs.mjs";
import hostedGalaxyJobs from "../api/v1/galaxy-jobs.mjs";
import hostedInverseResponseJobs from "../api/v1/inverse-response-jobs.mjs";
import hostedObservationEvaluationJobs from "../api/v1/observation-evaluation-jobs.mjs";
import hostedBatches from "../api/v1/batches.mjs";
import hostedDataUploads from "../api/v1/data-uploads.mjs";
import hostedDataUpload from "../api/v1/data-upload.mjs";
import hostedFieldJob from "../api/v1/field-job.mjs";
import hostedGalaxyJob from "../api/v1/galaxy-job.mjs";
import runs from "../api/v1/runs.mjs";
import twinRuns from "../api/v1/twin-runs.mjs";
import resolvedTwinEvidence from "../api/v1/resolved-twin-evidence.mjs";
import clusterEvidence from "../api/v1/cluster-evidence.mjs";
import { FIXED_MOND_FORMULA } from "../lib/formula.mjs";
import { readFileSync } from "node:fs";
import { sha256 } from "../lib/canonical.mjs";

function response() {
  return {
    headers: {},
    statusCode: 200,
    body: null,
    setHeader(name, value) { this.headers[name] = value; },
    status(code) { this.statusCode = code; return this; },
    json(body) { this.body = body; return this; },
    end() { return this; },
  };
}

function call(handler, { method = "GET", query = {}, body = undefined } = {}) {
  const output = response();
  handler({ method, query, body }, output);
  return output;
}

async function asyncCall(handler, { method = "GET", query = {}, body = undefined } = {}) {
  const output = response();
  await handler({ method, query, body }, output);
  return output;
}

test("health API identifies the deployed contract version and local worker boundary", () => {
  const output = call(health);
  assert.equal(output.statusCode, 200);
  assert.equal(output.body.version, "0.33.0-preview");
  assert.equal(output.body.capabilities.durablePrivateObjectStorage, "not_configured");
  assert.equal(output.body.capabilities.durableQueue, "not_configured");
  assert.equal(output.body.capabilities.transactionalJobDatabase, "not_configured");
  assert.equal(output.body.capabilities.statelessScientificWorker, "not_configured");
  assert.equal(
    output.body.capabilities.authenticatedFieldWorkerConnector,
    "available_requires_external_worker_configuration",
  );
  assert.equal(
    output.body.capabilities.authenticatedGalaxyWorkerConnector,
    "available_requires_external_worker_configuration",
  );
  assert.equal(
    output.body.capabilities.resolvedGalaxyExtractionAndGeneration,
    "production_worker_not_connected",
  );
  assert.equal(output.body.capabilities.localBaryonicImageConditioning, "available_in_dev_server");
  assert.equal(output.body.capabilities.localBaryonicEnsemblePropagation, "available_in_dev_server");
  assert.equal(output.body.capabilities.localInverseResponseMultiNullSuite, "available_in_dev_server");
  assert.equal(output.body.capabilities.researcherGuide, "available");
  assert.equal(output.body.capabilities.exactModelHashConfirmation, "required_for_execution");
  assert.equal(output.body.capabilities.heldoutObservedGalaxyTwins, "available");
  assert.equal(output.body.capabilities.resolvedTwinDevelopmentEvidence, "available");
  assert.equal(output.body.capabilities.resolvedTwinFinalHoldoutEvidence, "available");
  assert.equal(output.body.capabilities.resolvedClusterEvidenceRegistry, "available");
  assert.equal(
    output.body.capabilities.localDecoupledObservationEvaluationJobs,
    "available_in_dev_server",
  );
  assert.equal(output.body.capabilities.fieldSolvers2d3d, "worker_not_connected");
  assert.equal(
    output.body.capabilities.localComposedFieldObservationBatches,
    "available_in_dev_server",
  );
  assert.equal(
    output.body.capabilities.localRawMultipleImageLensing,
    "available_in_dev_server",
  );
  assert.equal(
    output.body.capabilities.localNonlocalConvolution,
    "available_in_dev_server",
  );
  assert.equal(
    output.body.capabilities.localInverseHaloResponseDiscovery,
    "available_in_dev_server",
  );
  assert.equal(
    output.body.capabilities.localCoupledTwoPotentialPhotonMatter,
    "available_in_dev_server",
  );
  assert.equal(
    output.body.capabilities.localAxisymmetricCylindricalFields,
    "available_in_dev_server",
  );
  assert.equal(
    output.body.capabilities.localAxisymmetricGalaxyObservations,
    "available_in_dev_server",
  );
  assert.equal(
    output.body.capabilities.localAxisymmetricPhotonLensing,
    "available_in_dev_server",
  );
  assert.equal(
    output.body.capabilities.localAxisymmetricRawMultipleImageLensing,
    "available_in_dev_server",
  );
});

test("storage readiness separates durable objects from an incomplete job lifecycle", async () => {
  const output = await asyncCall(storageReadiness);
  assert.equal(output.statusCode, 200);
  assert.equal(output.body.schemaVersion, "sigma-production-storage-readiness/1");
  assert.equal(output.body.objectStorage.state, "not_configured");
  assert.equal(output.body.queue.state, "not_configured");
  assert.equal(output.body.jobMetadataDatabase.state, "not_configured");
  assert.equal(output.body.outboxScheduler.state, "not_configured");
  assert.equal(output.body.statelessScientificContainer.state, "not_configured");
  assert.equal(output.body.productionExecution, "not_ready");
});

test("queue canary is unavailable without a deployment identity", async () => {
  const output = await asyncCall(queueCanary, { method: "POST" });
  assert.equal(output.statusCode, 503);
  assert.equal(output.body.error, "production_queue_not_configured");
});

test("published batch schema exposes bounded galaxy ensemble fan-out", () => {
  const schema = JSON.parse(readFileSync(
    new URL("../schemas/batch-submit-v1.schema.json", import.meta.url),
    "utf8",
  ));
  const system = schema.properties.systems.items.properties;
  assert.ok(system.galaxyArtifact.enum.includes("surface_density_ensemble"));
  assert.ok(system.galaxyArtifact.enum.includes("volume_density_ensemble"));
  assert.equal(system.ensembleSelection.properties.maximumChildren.maximum, 128);
  assert.equal(
    system.ensembleSelection.properties.surfaceRealizations.oneOf[1].maxItems,
    16,
  );
  assert.equal(
    system.ensembleSelection.properties.verticalRealizations.oneOf[1].maxItems,
    8,
  );
});

test("published galaxy schema exposes gravity-independent image conditioning", () => {
  const schema = JSON.parse(readFileSync(
    new URL("../schemas/galaxy-job-submit-v1.schema.json", import.meta.url),
    "utf8",
  ));
  const conditioning = schema.properties.uncertaintyEnsemble.properties.conditioning;
  assert.equal(conditioning.additionalProperties, false);
  assert.equal(
    conditioning.properties.likelihood.const,
    "diagonal_gaussian_surface_density",
  );
  assert.equal(conditioning.properties.correlationAreaPixels.minimum, 1);
  assert.equal(conditioning.properties.correlationAreaPixels.maximum, 4096);
});

test("cluster evidence keeps baryons, inferred halo maps, and raw lensing distinct", () => {
  const all = call(clusterEvidence);
  assert.equal(all.statusCode, 200);
  assert.equal(all.body.schemaVersion, "sigma-resolved-cluster-evidence/1");
  assert.equal(all.body.systems.length, 4);
  assert.equal(all.body.sample.inverseDiscoverySystems, 4);
  assert.equal(all.body.sample.rawCatalogReadinessGateSystems, 2);
  assert.equal(all.body.sample.rawForwardScoreReadySystems, 0);
  assert.equal(all.body.sample.prospectiveHoldoutSystems, 0);
  const { registrySha256, ...registryCore } = all.body;
  assert.equal(sha256(registryCore), registrySha256);
  for (const system of all.body.systems) {
    assert.equal(system.sampleState, "spent_development");
    assert.equal(system.baryonicEvidence.scientificRole, "baryonic_input");
    assert.equal(
      system.modelDerivedLensingEvidence.scientificRole,
      "model_derived_discovery_target",
    );
    assert.equal(system.rawLensingEvidence.scientificRole, "raw_observation");
    assert.equal(system.rawLensingEvidence.scoreReadyNow, false);
  }
  const selected = call(clusterEvidence, { query: { system: "AS295" } });
  assert.equal(selected.statusCode, 200);
  assert.equal(selected.body.systems.length, 1);
  assert.equal(selected.body.systems[0].rawLensingEvidence.secureImages, 18);
  const missing = call(clusterEvidence, { query: { system: "NOT-A-CLUSTER" } });
  assert.equal(missing.statusCode, 404);
  assert.equal(missing.body.error, "unknown_system");
});

test("catalog API lists and retrieves systems", () => {
  const datasetResponse = call(datasets);
  assert.equal(datasetResponse.statusCode, 200);
  assert.equal(datasetResponse.body.items[0].systemCount, 175);

  const listResponse = call(systems, { query: { q: "DDO", limit: "4" } });
  assert.equal(listResponse.statusCode, 200);
  assert.equal(listResponse.body.items.length, 4);
  assert.ok(listResponse.body.page.total > 4);

  const detailResponse = call(system, { query: { id: "DDO154" } });
  assert.equal(detailResponse.statusCode, 200);
  assert.equal(detailResponse.body.points.length, 12);
});

test("twin API withholds velocity targets until scoring", () => {
  const output = call(twinRuns, {
    method: "POST",
    body: { systemId: "DDO154", formula: FIXED_MOND_FORMULA },
  });
  assert.equal(output.statusCode, 200);
  assert.equal(output.body.type, "heldout_radial_twin_validation");
  assert.equal(output.body.manifest.twinProtocol.velocityTargetsUsedInExtraction, false);
  assert.equal(output.body.twin.parameterPackage.gravityParameters.length, 0);
  assert.equal(output.body.predictions.length, 12);
});

test("resolved twin evidence keeps generator, transport, and observation scores separate", () => {
  const all = call(resolvedTwinEvidence);
  assert.equal(all.statusCode, 200);
  assert.equal(all.body.evidenceClass, "precomputed_development_validation_and_final_holdout_result");
  assert.equal(all.body.sample.scoredVelocityPixels, 146532);
  assert.equal(all.body.systems.length, 8);
  assert.equal(all.body.generator.velocityTargetsUsed, false);
  assert.equal(all.body.generator.gravityParameters, 0);
  assert.equal(all.body.executionBoundary.arbitraryHosted2dFormulaExecution, false);
  const { evidenceSha256, ...evidenceCore } = all.body;
  assert.equal(sha256(evidenceCore), evidenceSha256);
  const selected = call(resolvedTwinEvidence, { query: { galaxy: "NGC3198" } });
  assert.equal(selected.statusCode, 200);
  assert.equal(selected.body.systems.length, 1);
  const system = selected.body.systems[0];
  assert.ok(system.twinFidelity.totalMapPixelCorrelation > 0.98);
  assert.ok(system.models.fixed_simple_mond.sourceToTwinTransport.lineOfSightRmseKmS < 8);
  assert.equal(system.models.fixed_simple_mond.twinVersusObserved.classification, "consistent");
  const validation = call(resolvedTwinEvidence, { query: { galaxy: "NGC6946" } });
  assert.equal(validation.statusCode, 200);
  assert.equal(validation.body.systems[0].split, "validation");
  assert.ok(validation.body.systems[0].geometryDiagnostic.axisOffsetDeg > 50);
  assert.ok(validation.body.systems[0].geometryDiagnostic.models.fixed_simple_mond.sourceVersusObserved.rmseKmS < 30);
  const holdout = call(resolvedTwinEvidence, { query: { galaxy: "NGC7331" } });
  assert.equal(holdout.statusCode, 200);
  assert.equal(holdout.body.systems[0].split, "holdout");
  assert.equal(holdout.body.systems[0].scoreProtocol, "preregistered_kinematic_axis_final_holdout");
  assert.equal(holdout.body.systems[0].simulatorFidelityLimitKmS, 12);
  assert.equal(holdout.body.systems[0].models.fixed_simple_mond.sourceVersusObserved.classification, "consistent");
  assert.equal(holdout.body.finalHoldoutVerdicts[0].competitiveButIncomplete, true);
  const missing = call(resolvedTwinEvidence, { query: { galaxy: "NOT-A-GALAXY" } });
  assert.equal(missing.statusCode, 404);
  assert.equal(missing.body.error, "unknown_system");
});

test("formula API returns canonical safety audit", () => {
  const output = call(validate, { method: "POST", body: FIXED_MOND_FORMULA });
  assert.equal(output.statusCode, 200);
  assert.equal(output.body.valid, true);
  assert.equal(output.body.safetyAudit.arbitraryCodeExecuted, false);
  assert.equal("evaluate" in output.body, false);
});

test("field-model API accepts one contract for distinct 2D/3D theories", () => {
  for (const name of ["newtonian-poisson", "refracted-gravity", "nonlocal-response", "two-potential"]) {
    const model = JSON.parse(readFileSync(new URL(`../examples/models/${name}.json`, import.meta.url), "utf8"));
    const output = call(validateModel, { method: "POST", body: model });
    assert.equal(output.statusCode, 200);
    assert.equal(output.body.valid, true, output.body.errors.join("; "));
    assert.equal(output.body.executionReadiness.state, "worker_not_connected");
  }
});

test("field-model confirmation is an explicit hash-bound handshake", () => {
  const draft = JSON.parse(readFileSync(new URL("../examples/models/newtonian-poisson.json", import.meta.url), "utf8"));
  draft.source.confirmedCanonical = false;
  delete draft.source.confirmedModelSha256;
  const validation = call(validateModel, { method: "POST", body: draft });
  assert.equal(validation.statusCode, 200);
  assert.equal(validation.body.valid, true);
  assert.equal(validation.body.confirmation.confirmed, false);
  assert.equal(validation.body.executionReadiness.state, "awaiting_researcher_confirmation");

  const wrong = call(confirmModel, {
    method: "POST",
    body: {
      schemaVersion: "sigma-model-confirmation-request/1",
      model: draft,
      expectedModelSha256: "0".repeat(64),
      acknowledgement: validation.body.confirmation.acknowledgement,
    },
  });
  assert.equal(wrong.statusCode, 409);
  assert.equal(wrong.body.error, "model_hash_changed");

  const confirmed = call(confirmModel, {
    method: "POST",
    body: {
      schemaVersion: "sigma-model-confirmation-request/1",
      model: draft,
      expectedModelSha256: validation.body.modelSha256,
      acknowledgement: validation.body.confirmation.acknowledgement,
    },
  });
  assert.equal(confirmed.statusCode, 200);
  assert.equal(confirmed.body.modelSha256, validation.body.modelSha256);
  assert.equal(
    confirmed.body.confirmedModel.source.confirmedModelSha256,
    validation.body.modelSha256,
  );
  const finalValidation = call(validateModel, { method: "POST", body: confirmed.body.confirmedModel });
  assert.equal(finalValidation.body.confirmation.confirmed, true);
  assert.equal(finalValidation.body.executionReadiness.state, "worker_not_connected");
});

test("field-job API preflights model and array metadata without claiming execution", () => {
  const model = JSON.parse(readFileSync(new URL("../examples/models/newtonian-poisson.json", import.meta.url), "utf8"));
  const core = {
    schemaVersion: "sigma-array-bundle/1",
    geometry: { coordinateSystem: "cartesian_3d", dimensions: 3, spacing: [1, 1, 1], lengthUnit: "m" },
    arrays: [{ key: "baryon_density", npzKey: "baryon_density", unit: "kg/m^3", rank: "scalar", role: "source", dtype: "<f8", shape: [17, 17, 17], elementCount: 4913, contentSha256: "2".repeat(64) }],
    provenance: { kind: "test" },
    license: { id: "CC0-1.0", redistributionAllowed: true },
  };
  const output = call(prepareFieldJob, {
    method: "POST",
    body: {
      model,
      inputBundle: { ...core, bundleSha256: sha256(core) },
      request: { schemaVersion: "sigma-field-job-request/1", requestedObservables: ["massive_tracer_acceleration"] },
    },
  });
  assert.equal(output.statusCode, 200);
  assert.equal(output.body.valid, true);
  assert.equal(output.body.state, "worker_not_connected");
});

test("run API produces a content-addressed comparator result", () => {
  const output = call(runs, {
    method: "POST",
    body: { systemIds: ["DDO154"], tests: ["rotation_curve"], formula: FIXED_MOND_FORMULA },
  });
  assert.equal(output.statusCode, 200);
  assert.match(output.body.id, /^run_[0-9a-f]{24}$/);
  assert.equal(output.body.state, "succeeded");
  assert.equal(output.body.manifest.parameterAccounting.perObject, 0);
});

test("heavy solver requests are never replaced with radial proxies", () => {
  const output = call(runs, {
    method: "POST",
    body: { systemIds: ["DDO154"], tests: ["raw_lensing_roots"], formula: FIXED_MOND_FORMULA },
  });
  assert.equal(output.statusCode, 503);
  assert.equal(output.body.error, "worker_not_connected");
});

test("production upload and queue endpoints disclose missing infrastructure", async () => {
  const upload = await asyncCall(hostedDataUploads, { method: "POST", body: {} });
  assert.equal(upload.statusCode, 503);
  assert.equal(upload.body.error, "production_control_plane_not_connected");
  const job = await asyncCall(hostedFieldJobs, { method: "POST", body: {} });
  assert.equal(job.statusCode, 503);
  assert.equal(job.body.error, "production_control_plane_not_connected");
  const uploadDetail = await asyncCall(hostedDataUpload, {
    query: { id: "upload_0123456789abcdef01234567" },
  });
  assert.equal(uploadDetail.statusCode, 503);
  assert.equal(uploadDetail.body.error, "production_control_plane_not_connected");
  const jobDetail = await asyncCall(hostedFieldJob, {
    query: { id: "job_0123456789abcdef01234567", resource: "events" },
  });
  assert.equal(jobDetail.statusCode, 503);
  assert.equal(jobDetail.body.error, "production_control_plane_not_connected");
  const unsafeArtifact = await asyncCall(hostedFieldJob, {
    query: { id: "job_0123456789abcdef01234567", resource: "artifact", name: "../secret" },
  });
  assert.equal(unsafeArtifact.statusCode, 404);
  const galaxyJob = await asyncCall(hostedGalaxyJobs, { method: "POST", body: {} });
  assert.equal(galaxyJob.statusCode, 503);
  assert.equal(galaxyJob.body.error, "production_control_plane_not_connected");
  const galaxyJobDetail = await asyncCall(hostedGalaxyJob, {
    query: { id: "job_0123456789abcdef01234567", resource: "artifacts" },
  });
  assert.equal(galaxyJobDetail.statusCode, 503);
  assert.equal(galaxyJobDetail.body.error, "production_control_plane_not_connected");
  const unsafeGalaxyArtifact = await asyncCall(hostedGalaxyJob, {
    query: { id: "job_0123456789abcdef01234567", resource: "artifact", name: "../secret" },
  });
  assert.equal(unsafeGalaxyArtifact.statusCode, 404);
  const observationJob = await asyncCall(hostedObservationEvaluationJobs, { method: "POST", body: {} });
  assert.equal(observationJob.statusCode, 503);
  assert.equal(observationJob.body.error, "production_control_plane_not_connected");
  const inverseJob = await asyncCall(hostedInverseResponseJobs, { method: "POST", body: {} });
  assert.equal(inverseJob.statusCode, 503);
  assert.equal(inverseJob.body.error, "production_control_plane_not_connected");
  const batch = await asyncCall(hostedBatches, { method: "POST", body: {} });
  assert.equal(batch.statusCode, 503);
  assert.equal(batch.body.error, "production_control_plane_not_connected");
});

test("Vercel rewrites expose authenticated field and galaxy lifecycles without a catch-all proxy", () => {
  const configuration = JSON.parse(readFileSync(new URL("../vercel.json", import.meta.url), "utf8"));
  const sources = new Set(configuration.rewrites.map((value) => value.source));
  for (const source of [
    "/api/v1/models/:sha256",
    "/api/v1/data-uploads/:id/content",
    "/api/v1/data-uploads/:id",
    "/api/v1/field-jobs/:id/events",
    "/api/v1/field-jobs/:id/artifacts/:name",
    "/api/v1/field-jobs/:id/cancel",
    "/api/v1/galaxy-jobs/:id/events",
    "/api/v1/galaxy-jobs/:id/artifacts/:name",
    "/api/v1/galaxy-jobs/:id/cancel",
    "/api/v1/jobs/:id/events",
    "/api/v1/jobs/:id/artifacts/:name",
    "/api/v1/jobs/:id/cancel",
  ]) {
    assert.equal(sources.has(source), true, `missing Vercel rewrite ${source}`);
  }
  assert.equal(configuration.rewrites.some((value) => value.source.includes("(.*)")), false);
  assert.equal(
    configuration.functions["api/v1/queue-canary-consumer.mjs"].experimentalTriggers[0].topic,
    "sigma-control-plane-canary-v1",
  );
  assert.equal(
    configuration.functions["api/v1/queue-job-consumer.mjs"].experimentalTriggers[0].topic,
    "sigma-control-plane-jobs-v1",
  );
});
