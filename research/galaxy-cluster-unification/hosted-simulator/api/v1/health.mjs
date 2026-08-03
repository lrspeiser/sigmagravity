import { options, requireMethod, send } from "../../lib/http.mjs";
import { privateBlobStorageState } from "../../lib/private-blob-store.mjs";
import { productionDatabaseState } from "../../lib/production-database.mjs";
import { productionQueueState } from "../../lib/production-queue.mjs";
import { statelessWorkerState } from "../../lib/stateless-worker-client.mjs";

export default function handler(request, response) {
  if (options(request, response) || !requireMethod(request, response, "GET")) return;
  send(response, 200, {
    status: "ok",
    service: "sigma-gravity-research-simulator",
    version: "0.33.0-preview",
    capabilities: {
      researcherGuide: "available",
      radialRotationCurves: "available",
      syntheticRadialGalaxies: "available",
      heldoutObservedGalaxyTwins: "available",
      resolvedTwinDevelopmentEvidence: "available",
      resolvedTwinValidationEvidence: "available",
      resolvedTwinFinalHoldoutEvidence: "available",
      resolvedClusterEvidenceRegistry: "available",
      typedFieldModelValidation: "available",
      exactModelHashConfirmation: "required_for_execution",
      fieldJobPreflight: "available",
      localAsyncFieldJobs: "available_in_dev_server",
      localResolvedGalaxyJobs: "available_in_dev_server",
      localBaryonicUncertaintyEnsembles: "available_in_dev_server",
      localBaryonicEnsemblePropagation: "available_in_dev_server",
      localBaryonicImageConditioning: "available_in_dev_server",
      localMultiSystemBatches: "available_in_dev_server",
      localCircularSpeedObservationAdapter: "available_in_dev_server",
      localDecoupledObservationEvaluationJobs: "available_in_dev_server",
      localComposedFieldObservationBatches: "available_in_dev_server",
      localTypedPhotonLensingMaps: "available_in_dev_server",
      localRawMultipleImageLensing: "available_in_dev_server",
      localNonlocalConvolution: "available_in_dev_server",
      localInverseHaloResponseDiscovery: "available_in_dev_server",
      localInverseResponseMultiNullSuite: "available_in_dev_server",
      localCoupledTwoPotentialPhotonMatter: "available_in_dev_server",
      localAxisymmetricCylindricalFields: "available_in_dev_server",
      localAxisymmetricGalaxyObservations: "available_in_dev_server",
      localAxisymmetricPhotonLensing: "available_in_dev_server",
      localAxisymmetricRawMultipleImageLensing: "available_in_dev_server",
      authenticatedFieldWorkerConnector: "available_requires_external_worker_configuration",
      authenticatedGalaxyWorkerConnector: "available_requires_external_worker_configuration",
      durablePrivateObjectStorage: {
        configured: "connected_private_content_addressed",
        misconfigured: "misconfigured",
        not_configured: "not_configured",
      }[privateBlobStorageState()],
      durableQueue: {
        configured: "configured_canary_required",
        misconfigured: "misconfigured",
        not_configured: "not_configured",
      }[productionQueueState()],
      transactionalJobDatabase: productionDatabaseState(),
      projectScopedResearchApi: productionDatabaseState() === "configured"
        ? "configured_migration_verification_required"
        : "not_configured",
      productionModelUploadJobRegistry: "available_requires_database_migration",
      productionProjectQuotasAndAudit: "available_requires_database_migration",
      statelessScientificWorker: statelessWorkerState(),
      resolvedGalaxyExtractionAndGeneration: "production_worker_not_connected",
      fieldSolvers2d3d: "worker_not_connected",
      rawMultipleImageLensing: "production_worker_not_connected",
    },
  });
}
