import { options, requireMethod, send } from "../../lib/http.mjs";

const specification = {
  openapi: "3.1.0",
  info: {
    title: "Sigma Gravity Research Simulator API",
    version: "0.4.0-preview",
    description: "Stateless radial tests plus a dimension-checked, formula-independent 2D/3D contract. The local development server includes an asynchronous reference queue; the public deployment still requires durable storage and isolated workers.",
  },
  paths: {
    "/api/v1/health": { get: { summary: "Service status" } },
    "/api/v1/datasets": { get: { summary: "List versioned datasets" } },
    "/api/v1/systems": { get: { summary: "List and filter real systems" } },
    "/api/v1/systems/{id}": { get: { summary: "Retrieve a galaxy and its radial measurements" } },
    "/api/v1/synthetic-galaxies": { post: { summary: "Create a deterministic synthetic radial galaxy" } },
    "/api/v1/formulas/validate": { post: { summary: "Validate and hash a safe formula AST" } },
    "/api/v1/models/validate": { post: { summary: "Validate and hash a scalar/vector/tensor 2D/3D field-model manifest" } },
    "/api/v1/field-jobs/prepare": { post: { summary: "Preflight a model, content-hashed array manifest, grid, boundary, and observable request" } },
    "/api/v1/data-uploads": { post: { summary: "Create an immutable NPZ array-upload ticket (local reference backend)" } },
    "/api/v1/data-uploads/{id}": { get: { summary: "Inspect an array upload" } },
    "/api/v1/data-uploads/{id}/content": { put: { summary: "Upload hash- and size-bound NPZ bytes" } },
    "/api/v1/field-jobs": {
      get: { summary: "List local field jobs" },
      post: { summary: "Queue a confirmed field manifest against a ready data upload" },
    },
    "/api/v1/field-jobs/{id}": { get: { summary: "Read field-job state" } },
    "/api/v1/field-jobs/{id}/events": { get: { summary: "Read ordered field-job lifecycle events" } },
    "/api/v1/field-jobs/{id}/artifacts": { get: { summary: "Read the verified artifact index and scientific manifest" } },
    "/api/v1/field-jobs/{id}/artifacts/{name}": { get: { summary: "Download one allow-listed, rehashed artifact" } },
    "/api/v1/field-jobs/{id}/cancel": { post: { summary: "Cancel a queued or running local field job" } },
    "/api/v1/galaxy-jobs": {
      get: { summary: "List local resolved-galaxy extraction/generation jobs" },
      post: { summary: "Queue formula-independent extraction, generation, or 2D/3D round-trip work" },
    },
    "/api/v1/galaxy-jobs/{id}": { get: { summary: "Read resolved-galaxy job state" } },
    "/api/v1/galaxy-jobs/{id}/events": { get: { summary: "Read resolved-galaxy lifecycle events" } },
    "/api/v1/galaxy-jobs/{id}/artifacts": { get: { summary: "Read verified 2D/3D density and parameter artifacts" } },
    "/api/v1/galaxy-jobs/{id}/artifacts/{name}": { get: { summary: "Download one allow-listed resolved-galaxy artifact" } },
    "/api/v1/galaxy-jobs/{id}/cancel": { post: { summary: "Cancel a queued or running resolved-galaxy job" } },
    "/api/v1/runs": { post: { summary: "Score a formula without fitting object-specific gravity parameters" } },
  },
};

export default function handler(request, response) {
  if (options(request, response) || !requireMethod(request, response, "GET")) return;
  send(response, 200, specification);
}
