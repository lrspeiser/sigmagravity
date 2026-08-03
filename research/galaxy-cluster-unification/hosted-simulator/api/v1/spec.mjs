import { options, requireMethod, send } from "../../lib/http.mjs";

const specification = {
  openapi: "3.1.0",
  info: {
    title: "Sigma Gravity Research Simulator API",
    version: "0.23.0-preview",
    description: "Stateless radial and held-out observed-galaxy twin tests, frozen resolved development, validation, and one-shot final-holdout evidence, a role-safe four-cluster evidence registry, plus a dimension-checked formula-independent 2D/3D contract. The local worker executes explicit nonlocal convolutions, genuinely coupled field equations, separately typed photon and matter observables, seeded observation-conditioned baryonic prior ensembles, and deterministic fan-out of one confirmed field model across selected ensemble realizations with per-parent prediction quantiles. It also provides an inverse baryon-to-response discovery workbench whose targets must be labeled model-derived. Ensemble ranges are prior-induced prediction spread, and inverse results are hypothesis generators rather than forward theory tests. Public heavy execution still requires durable storage and isolated workers.",
  },
  paths: {
    "/api/v1/health": { get: { summary: "Service status" } },
    "/api/v1/datasets": { get: { summary: "List versioned datasets" } },
    "/api/v1/systems": { get: { summary: "List and filter real systems" } },
    "/api/v1/systems/{id}": { get: { summary: "Retrieve a galaxy and its radial measurements" } },
    "/api/v1/synthetic-galaxies": { post: { summary: "Create a deterministic synthetic radial galaxy" } },
    "/api/v1/twin-runs": { post: { summary: "Regenerate a compressed baryonic twin without observed speeds, then score a formula against the held-out rotation curve" } },
    "/api/v1/resolved-twin-evidence": { get: { summary: "Retrieve frozen four-development, two-validation, and two one-shot final-holdout 2D results with separate twin-fidelity, formula-transport, observed-velocity, and geometry scores" } },
    "/api/v1/cluster-evidence": { get: { summary: "Retrieve the role-safe four-system RELICS registry of baryonic inputs, model-derived lensing discovery targets, raw observations, readiness, hashes, and blockers" } },
    "/api/v1/formulas/validate": { post: { summary: "Validate and hash a safe formula AST" } },
    "/api/v1/models/validate": { post: { summary: "Validate and hash a scalar/vector/tensor 2D/3D field-model manifest" } },
    "/api/v1/models/confirm": { post: { summary: "Bind explicit researcher acknowledgement to the exact validated computational model hash" } },
    "/api/v1/field-jobs/prepare": { post: { summary: "Preflight a model, content-hashed arrays, 2D/3D grid, boundary, and observation-target request" } },
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
    "/api/v1/observation-evaluation-jobs": {
      get: { summary: "List local jobs that score immutable solved fields without re-solving gravity" },
      post: { summary: "Queue observation projection and scoring against a completed field job" },
    },
    "/api/v1/observation-evaluation-jobs/{id}": { get: { summary: "Read observation-evaluation job state" } },
    "/api/v1/observation-evaluation-jobs/{id}/events": { get: { summary: "Read ordered observation-evaluation lifecycle events" } },
    "/api/v1/observation-evaluation-jobs/{id}/artifacts": { get: { summary: "Read verified scores, predictions, and source-field references" } },
    "/api/v1/observation-evaluation-jobs/{id}/artifacts/{name}": { get: { summary: "Download one allow-listed, rehashed observation artifact" } },
    "/api/v1/observation-evaluation-jobs/{id}/cancel": { post: { summary: "Cancel a queued or running observation-evaluation job" } },
    "/api/v1/galaxy-jobs": {
      get: { summary: "List local resolved-galaxy extraction/generation jobs" },
      post: { summary: "Queue formula-independent extraction, generation, 2D/3D prior ensembles, or round-trip work" },
    },
    "/api/v1/galaxy-jobs/{id}": { get: { summary: "Read resolved-galaxy job state" } },
    "/api/v1/galaxy-jobs/{id}/events": { get: { summary: "Read resolved-galaxy lifecycle events" } },
    "/api/v1/galaxy-jobs/{id}/artifacts": { get: { summary: "Read verified 2D/3D density and parameter artifacts" } },
    "/api/v1/galaxy-jobs/{id}/artifacts/{name}": { get: { summary: "Download one allow-listed resolved-galaxy artifact" } },
    "/api/v1/galaxy-jobs/{id}/cancel": { post: { summary: "Cancel a queued or running resolved-galaxy job" } },
    "/api/v1/inverse-response-jobs": {
      get: { summary: "List local inverse baryon-to-response discovery jobs" },
      post: { summary: "Infer candidate stationary kernels from baryonic inputs and explicitly model-derived discovery targets" },
    },
    "/api/v1/inverse-response-jobs/{id}": { get: { summary: "Read inverse-response job state" } },
    "/api/v1/inverse-response-jobs/{id}/events": { get: { summary: "Read inverse-response lifecycle events" } },
    "/api/v1/inverse-response-jobs/{id}/artifacts": { get: { summary: "Read kernels, uncertainty, nulls, predictions, and deterministic reports" } },
    "/api/v1/inverse-response-jobs/{id}/artifacts/{name}": { get: { summary: "Download one allow-listed inverse-response artifact" } },
    "/api/v1/inverse-response-jobs/{id}/cancel": { post: { summary: "Cancel an inverse-response discovery job" } },
    "/api/v1/batches": {
      get: { summary: "List local multi-system field batches" },
      post: { summary: "Compose reusable field and observation children across uploads, generated systems, or selected baryonic ensemble realizations" },
    },
    "/api/v1/batches/{id}": { get: { summary: "Read batch state and child progress" } },
    "/api/v1/batches/{id}/events": { get: { summary: "Read ordered batch lifecycle events" } },
    "/api/v1/batches/{id}/artifacts": { get: { summary: "Read deterministic batch report artifacts" } },
    "/api/v1/batches/{id}/artifacts/{name}": { get: { summary: "Download one allow-listed batch artifact" } },
    "/api/v1/batches/{id}/cancel": { post: { summary: "Cancel a batch and its nonterminal child jobs" } },
    "/api/v1/runs": { post: { summary: "Score a formula without fitting object-specific gravity parameters" } },
  },
};

export default function handler(request, response) {
  if (options(request, response) || !requireMethod(request, response, "GET")) return;
  send(response, 200, specification);
}
