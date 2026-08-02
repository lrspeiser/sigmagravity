import { fail, send, setCors } from "./http.mjs";

function method(response, allowed) {
  response.setHeader("Allow", `OPTIONS, ${allowed.join(", ")}`);
  send(response, 405, { error: "method_not_allowed", allowed });
}

export function createLocalBatchRouter(service) {
  return async function route(request, response, url) {
    if (!url.pathname.startsWith("/api/v1/batches")) return false;
    response.setHeader("Cache-Control", "no-store");
    try {
      if (url.pathname === "/api/v1/batches") {
        if (request.method === "POST") send(response, 202, await service.createBatch(request.body));
        else if (request.method === "GET") send(response, 200, await service.listBatches());
        else if (request.method === "OPTIONS") { setCors(response); response.status(204).end(); }
        else method(response, ["GET", "POST"]);
        return true;
      }
      const match = url.pathname.match(/^\/api\/v1\/batches\/(batch_[0-9a-f]{24})(?:\/(events|artifacts|cancel)(?:\/(.+))?)?$/);
      if (!match) return false;
      const [, id, resource, artifactName] = match;
      if (!resource && request.method === "GET") send(response, 200, await service.getBatch(id));
      else if (resource === "events" && request.method === "GET") send(response, 200, await service.getEvents(id));
      else if (resource === "artifacts" && !artifactName && request.method === "GET") send(response, 200, await service.getArtifacts(id));
      else if (resource === "artifacts" && artifactName && request.method === "GET") {
        const artifact = await service.getArtifact(id, artifactName);
        setCors(response);
        response.setHeader("Content-Type", artifact.record.path.endsWith(".html") ? "text/html; charset=utf-8" : "application/octet-stream");
        response.setHeader("Content-Length", String(artifact.record.bytes));
        response.setHeader("X-Content-SHA256", artifact.record.sha256);
        response.status(200).end(artifact.content);
      } else if (resource === "cancel" && request.method === "POST") send(response, 200, await service.cancelBatch(id));
      else if (request.method === "OPTIONS") { setCors(response); response.status(204).end(); }
      else method(response, resource === "cancel" ? ["POST"] : ["GET"]);
      return true;
    } catch (error) {
      fail(response, error, error.statusCode ?? 500);
      return true;
    }
  };
}
