import { fail, send, setCors } from "./http.mjs";

function method(response, allowed) {
  response.setHeader("Allow", `OPTIONS, ${allowed.join(", ")}`);
  send(response, 405, { error: "method_not_allowed", allowed });
}

export function createLocalFieldJobRouter(service) {
  return async function route(request, response, url) {
    if (!url.pathname.startsWith("/api/v1/data-uploads") && !url.pathname.startsWith("/api/v1/field-jobs") && !url.pathname.startsWith("/api/v1/galaxy-jobs") && !url.pathname.startsWith("/api/v1/observation-evaluation-jobs")) return false;
    response.setHeader("Cache-Control", "no-store");
    try {
      if (url.pathname === "/api/v1/data-uploads") {
        if (request.method === "POST") {
          send(response, 201, await service.createUpload(request.body));
        } else if (request.method === "OPTIONS") {
          setCors(response); response.status(204).end();
        } else method(response, ["POST"]);
        return true;
      }
      const uploadMatch = url.pathname.match(/^\/api\/v1\/data-uploads\/(upload_[0-9a-f]{24})(?:\/(content))?$/);
      if (uploadMatch) {
        if (!uploadMatch[2] && request.method === "GET") send(response, 200, await service.getUpload(uploadMatch[1]));
        else if (uploadMatch[2] === "content" && request.method === "PUT") send(response, 200, await service.putUploadContent(uploadMatch[1], request.body));
        else if (request.method === "OPTIONS") { setCors(response); response.status(204).end(); }
        else method(response, uploadMatch[2] ? ["PUT"] : ["GET"]);
        return true;
      }
      if (url.pathname === "/api/v1/field-jobs") {
        if (request.method === "POST") send(response, 202, await service.createFieldJob(request.body));
        else if (request.method === "GET") send(response, 200, await service.listFieldJobs());
        else if (request.method === "OPTIONS") { setCors(response); response.status(204).end(); }
        else method(response, ["GET", "POST"]);
        return true;
      }
      if (url.pathname === "/api/v1/galaxy-jobs") {
        if (request.method === "POST") send(response, 202, await service.createGalaxyJob(request.body));
        else if (request.method === "GET") send(response, 200, await service.listGalaxyJobs());
        else if (request.method === "OPTIONS") { setCors(response); response.status(204).end(); }
        else method(response, ["GET", "POST"]);
        return true;
      }
      if (url.pathname === "/api/v1/observation-evaluation-jobs") {
        if (request.method === "POST") send(response, 202, await service.createObservationEvaluationJob(request.body));
        else if (request.method === "GET") send(response, 200, await service.listObservationEvaluationJobs());
        else if (request.method === "OPTIONS") { setCors(response); response.status(204).end(); }
        else method(response, ["GET", "POST"]);
        return true;
      }
      const observationJobMatch = url.pathname.match(/^\/api\/v1\/observation-evaluation-jobs\/(job_[0-9a-f]{24})(?:\/(events|artifacts|cancel)(?:\/(.+))?)?$/);
      if (observationJobMatch) {
        const [, id, resource, artifactName] = observationJobMatch;
        await service.getObservationEvaluationJob(id);
        if (!resource && request.method === "GET") send(response, 200, await service.getObservationEvaluationJob(id));
        else if (resource === "events" && request.method === "GET") send(response, 200, await service.getEvents(id));
        else if (resource === "artifacts" && !artifactName && request.method === "GET") send(response, 200, await service.getArtifacts(id));
        else if (resource === "artifacts" && artifactName && request.method === "GET") {
          const artifact = await service.getArtifact(id, artifactName);
          setCors(response);
          response.setHeader("Content-Type", "application/octet-stream");
          response.setHeader("Content-Length", String(artifact.record.bytes));
          response.setHeader("X-Content-SHA256", artifact.record.sha256);
          response.status(200).end(artifact.content);
        } else if (resource === "cancel" && request.method === "POST") send(response, 200, await service.cancelObservationEvaluationJob(id));
        else if (request.method === "OPTIONS") { setCors(response); response.status(204).end(); }
        else method(response, resource === "cancel" ? ["POST"] : ["GET"]);
        return true;
      }
      const galaxyJobMatch = url.pathname.match(/^\/api\/v1\/galaxy-jobs\/(job_[0-9a-f]{24})(?:\/(events|artifacts|cancel)(?:\/(.+))?)?$/);
      if (galaxyJobMatch) {
        const [, id, resource, artifactName] = galaxyJobMatch;
        await service.getGalaxyJob(id);
        if (!resource && request.method === "GET") send(response, 200, await service.getGalaxyJob(id));
        else if (resource === "events" && request.method === "GET") send(response, 200, await service.getEvents(id));
        else if (resource === "artifacts" && !artifactName && request.method === "GET") send(response, 200, await service.getArtifacts(id));
        else if (resource === "artifacts" && artifactName && request.method === "GET") {
          const artifact = await service.getArtifact(id, artifactName);
          setCors(response);
          response.setHeader("Content-Type", "application/octet-stream");
          response.setHeader("Content-Length", String(artifact.record.bytes));
          response.setHeader("X-Content-SHA256", artifact.record.sha256);
          response.status(200).end(artifact.content);
        } else if (resource === "cancel" && request.method === "POST") send(response, 200, await service.cancelGalaxyJob(id));
        else if (request.method === "OPTIONS") { setCors(response); response.status(204).end(); }
        else method(response, resource === "cancel" ? ["POST"] : ["GET"]);
        return true;
      }
      const jobMatch = url.pathname.match(/^\/api\/v1\/field-jobs\/(job_[0-9a-f]{24})(?:\/(events|artifacts|cancel)(?:\/(.+))?)?$/);
      if (!jobMatch) return false;
      const [, id, resource, artifactName] = jobMatch;
      await service.getFieldJob(id);
      if (!resource && request.method === "GET") send(response, 200, await service.getFieldJob(id));
      else if (resource === "events" && request.method === "GET") send(response, 200, await service.getEvents(id));
      else if (resource === "artifacts" && !artifactName && request.method === "GET") send(response, 200, await service.getArtifacts(id));
      else if (resource === "artifacts" && artifactName && request.method === "GET") {
        const artifact = await service.getArtifact(id, artifactName);
        setCors(response);
        response.setHeader("Content-Type", "application/octet-stream");
        response.setHeader("Content-Length", String(artifact.record.bytes));
        response.setHeader("X-Content-SHA256", artifact.record.sha256);
        response.status(200).end(artifact.content);
      } else if (resource === "cancel" && request.method === "POST") send(response, 200, await service.cancelFieldJob(id));
      else if (request.method === "OPTIONS") { setCors(response); response.status(204).end(); }
      else method(response, resource === "cancel" ? ["POST"] : ["GET"]);
      return true;
    } catch (error) {
      fail(response, error, error.statusCode ?? 500);
      return true;
    }
  };
}
