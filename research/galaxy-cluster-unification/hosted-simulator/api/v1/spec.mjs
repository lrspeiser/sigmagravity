import { options, requireMethod, send } from "../../lib/http.mjs";

const specification = {
  openapi: "3.1.0",
  info: {
    title: "Sigma Gravity Research Simulator API",
    version: "0.2.0-preview",
    description: "Stateless radial tests plus a dimension-checked, formula-independent contract for future 2D/3D field workers. Heavy field and lensing execution is not yet connected.",
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
    "/api/v1/runs": { post: { summary: "Score a formula without fitting object-specific gravity parameters" } },
  },
};

export default function handler(request, response) {
  if (options(request, response) || !requireMethod(request, response, "GET")) return;
  send(response, 200, specification);
}
