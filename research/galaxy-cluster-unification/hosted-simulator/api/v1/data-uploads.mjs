import { options } from "../../lib/http.mjs";
import { proxyWorkerRequest } from "../../lib/remote-worker-proxy.mjs";

export default async function handler(request, response) {
  if (options(request, response)) return;
  if (request.method !== "POST") {
    response.setHeader("Allow", "OPTIONS, POST");
    send(response, 405, { error: "method_not_allowed", allowed: ["POST"] });
    return;
  }
  response.setHeader("Cache-Control", "no-store");
  await proxyWorkerRequest({
    request,
    response,
    path: "api/v1/data-uploads",
    unavailable: {
      error: "production_storage_not_connected",
      message: "The authenticated worker connector exists, but durable storage and a worker endpoint are not configured for this deployment.",
      localReference: "npm run dev",
      requestSchema: "/schemas/data-upload-request-v1.schema.json",
    },
  });
}
