import { options } from "../../lib/http.mjs";
import { proxyWorkerRequest } from "../../lib/remote-worker-proxy.mjs";

export default async function handler(request, response) {
  if (options(request, response)) return;
  if (!["GET", "POST"].includes(request.method)) {
    response.setHeader("Allow", "OPTIONS, GET, POST");
    send(response, 405, { error: "method_not_allowed", allowed: ["GET", "POST"] });
    return;
  }
  response.setHeader("Cache-Control", "no-store");
  await proxyWorkerRequest({
    request,
    response,
    path: "api/v1/field-jobs",
    unavailable: {
      error: "production_worker_not_connected",
      message: "The authenticated field-worker connector exists, but no durable worker endpoint is configured for this deployment.",
      localReference: "npm run dev",
      requestSchema: "/schemas/field-job-submit-v1.schema.json",
    },
  });
}
