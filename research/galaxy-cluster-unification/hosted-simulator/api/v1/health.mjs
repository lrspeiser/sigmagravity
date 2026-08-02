import { options, requireMethod, send } from "../../lib/http.mjs";

export default function handler(request, response) {
  if (options(request, response) || !requireMethod(request, response, "GET")) return;
  send(response, 200, {
    status: "ok",
    service: "sigma-gravity-research-simulator",
    version: "0.2.0-preview",
    capabilities: {
      radialRotationCurves: "available",
      syntheticRadialGalaxies: "available",
      typedFieldModelValidation: "available",
      fieldSolvers2d3d: "worker_not_connected",
      rawClusterLensing: "worker_not_connected",
    },
  });
}
