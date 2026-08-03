import { options, send } from "../../lib/http.mjs";
import { proxyWorkerRequest } from "../../lib/remote-worker-proxy.mjs";

const IDENTIFIER = /^job_[0-9a-f]{24}$/;

function route(request) {
  const id = request.query?.id;
  const resource = request.query?.resource;
  const name = request.query?.name;
  if (typeof id !== "string" || !IDENTIFIER.test(id)) return null;
  if (resource === undefined && name === undefined) return { path: `api/v1/galaxy-jobs/${id}`, method: "GET" };
  if (resource === "events" && name === undefined) return { path: `api/v1/galaxy-jobs/${id}/events`, method: "GET" };
  if (resource === "artifacts" && name === undefined) return { path: `api/v1/galaxy-jobs/${id}/artifacts`, method: "GET" };
  if (resource === "cancel" && name === undefined) return { path: `api/v1/galaxy-jobs/${id}/cancel`, method: "POST" };
  if (
    resource === "artifact"
    && typeof name === "string"
    && name.length > 0
    && name.length <= 160
    && !name.includes("/")
    && !name.includes("\\")
    && name !== "."
    && name !== ".."
  ) {
    return { path: `api/v1/galaxy-jobs/${id}/artifacts/${encodeURIComponent(name)}`, method: "GET" };
  }
  return null;
}

export default async function handler(request, response) {
  if (options(request, response)) return;
  const target = route(request);
  if (!target) {
    send(response, 404, { error: "not_found" });
    return;
  }
  if (request.method !== target.method) {
    response.setHeader("Allow", `OPTIONS, ${target.method}`);
    send(response, 405, { error: "method_not_allowed", allowed: [target.method] });
    return;
  }
  await proxyWorkerRequest({
    request,
    response,
    path: target.path,
    unavailable: {
      error: "production_worker_not_connected",
      message: "The authenticated galaxy-worker connector exists, but no durable worker endpoint is configured for this deployment.",
    },
  });
}
