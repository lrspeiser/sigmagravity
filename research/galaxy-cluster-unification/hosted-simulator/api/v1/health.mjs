import { options, requireMethod, send } from "../../lib/http.mjs";

export default function handler(request, response) {
  if (options(request, response) || !requireMethod(request, response, "GET")) return;
  send(response, 200, {
    status: "ok",
    service: "sigma-gravity-research-simulator",
    version: "0.8.0-preview",
    capabilities: {
      radialRotationCurves: "available",
      syntheticRadialGalaxies: "available",
      typedFieldModelValidation: "available",
      fieldJobPreflight: "available",
      localAsyncFieldJobs: "available_in_dev_server",
      localResolvedGalaxyJobs: "available_in_dev_server",
      localMultiSystemBatches: "available_in_dev_server",
      localCircularSpeedObservationAdapter: "available_in_dev_server",
      localDecoupledObservationEvaluationJobs: "available_in_dev_server",
      localComposedFieldObservationBatches: "available_in_dev_server",
      fieldSolvers2d3d: "worker_not_connected",
      rawClusterLensing: "worker_not_connected",
    },
  });
}
