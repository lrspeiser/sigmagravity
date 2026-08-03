import { options, requireMethod, send } from "../../lib/http.mjs";

export default function handler(request, response) {
  if (options(request, response) || !requireMethod(request, response, "GET")) return;
  send(response, 200, {
    status: "ok",
    service: "sigma-gravity-research-simulator",
    version: "0.21.0-preview",
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
      fieldSolvers2d3d: "worker_not_connected",
      rawMultipleImageLensing: "production_worker_not_connected",
    },
  });
}
