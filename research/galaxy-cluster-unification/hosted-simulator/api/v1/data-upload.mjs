import { options, send } from "../../lib/http.mjs";
import { proxyWorkerRequest } from "../../lib/remote-worker-proxy.mjs";

const IDENTIFIER = /^upload_[0-9a-f]{24}$/;

export default async function handler(request, response) {
  if (options(request, response)) return;
  const id = request.query?.id;
  const resource = request.query?.resource;
  if (typeof id !== "string" || !IDENTIFIER.test(id) || ![undefined, "content"].includes(resource)) {
    send(response, 404, { error: "not_found" });
    return;
  }
  const allowed = resource === "content" ? "PUT" : "GET";
  if (request.method !== allowed) {
    response.setHeader("Allow", `OPTIONS, ${allowed}`);
    send(response, 405, { error: "method_not_allowed", allowed: [allowed] });
    return;
  }
  await proxyWorkerRequest({
    request,
    response,
    path: `api/v1/data-uploads/${id}${resource ? "/content" : ""}`,
    unavailable: {
      error: "production_storage_not_connected",
      message: "The authenticated worker connector exists, but durable storage and a worker endpoint are not configured for this deployment.",
    },
  });
}
