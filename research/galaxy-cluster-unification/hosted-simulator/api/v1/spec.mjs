import { options, requireMethod, send } from "../../lib/http.mjs";

const specification = {
  openapi: "3.1.0",
  info: {
    title: "Sigma Gravity Research Simulator API",
    version: "0.35.0-preview",
    description: "Formula-independent radial benchmarks and confirmed 2D/3D research contracts. A signed advanced-code manifest and separate single-use sandbox contract are now published and accepted in real container CI; uploaded code never executes in Vercel or the trusted safe-language worker. The production API has project-scoped hashed bearer credentials, immutable confirmed-model and data registration, PostgreSQL quotas and audit events, idempotent jobs, a transactional outbox, and verified artifacts. Public heavy execution remains fail-closed until PostgreSQL, a scheduler, and the appropriate scientific worker are connected.",
  },
  components: {
    securitySchemes: {
      projectBearer: { type: "http", scheme: "bearer", bearerFormat: "sgp project credential" },
    },
  },
  paths: {
    "/api/v1/health": { get: { summary: "Service status" } },
    "/api/v1/storage-readiness": { get: { summary: "Durable scientific object-storage and production execution readiness" } },
    "/api/v1/queue-canary": {
      get: { summary: "Read the current deployment's private queue-delivery acknowledgement" },
      post: { summary: "Publish one deployment-scoped, seven-day-deduplicated queue canary" },
    },
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
    "/api/v1/plugins/preflight": { post: { summary: "Verify a signed advanced plug-in manifest without executing code or conferring publisher trust" } },
    "/api/v1/models": {
      get: { summary: "List project-scoped immutable confirmed models", security: [{ projectBearer: [] }] },
      post: { summary: "Persist an exact-hash confirmation receipt and canonical model", security: [{ projectBearer: [] }] },
    },
    "/api/v1/models/{sha256}": { get: { summary: "Read one project-scoped confirmed model record", security: [{ projectBearer: [] }] } },
    "/api/v1/field-jobs/prepare": { post: { summary: "Preflight a model, content-hashed arrays, 2D/3D grid, boundary, and observation-target request" } },
    "/api/v1/data-uploads": {
      get: { summary: "List project-scoped immutable array uploads", security: [{ projectBearer: [] }] },
      post: { summary: "Register an exact manifest, archive hash, byte count, roles, and license", security: [{ projectBearer: [] }] },
    },
    "/api/v1/data-uploads/{id}": { get: { summary: "Inspect an array upload" } },
    "/api/v1/data-uploads/{id}/content": { put: { summary: "Upload hash- and size-bound NPZ bytes" } },
    "/api/v1/field-jobs": {
      get: { summary: "List project-scoped durable jobs", security: [{ projectBearer: [] }] },
      post: { summary: "Queue a registered confirmed model against a ready immutable upload", security: [{ projectBearer: [] }] },
    },
    "/api/v1/field-jobs/{id}": { get: { summary: "Read field-job state" } },
    "/api/v1/field-jobs/{id}/events": { get: { summary: "Read ordered field-job lifecycle events" } },
    "/api/v1/field-jobs/{id}/artifacts": { get: { summary: "Read the verified artifact index and scientific manifest" } },
    "/api/v1/field-jobs/{id}/artifacts/{name}": { get: { summary: "Download one allow-listed, rehashed artifact" } },
    "/api/v1/field-jobs/{id}/cancel": { post: { summary: "Cancel a queued or running field job" } },
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
      get: { summary: "List resolved-galaxy jobs through the local service or configured authenticated worker" },
      post: { summary: "Queue gravity-independent extraction, controlled generation, 2D/3D ensembles, or round-trip work" },
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
    "/api/v1/jobs": {
      get: { summary: "List every durable job class in the authenticated project", security: [{ projectBearer: [] }] },
      post: { summary: "Submit a formula-neutral durable job with an Idempotency-Key", security: [{ projectBearer: [] }] },
    },
    "/api/v1/jobs/{id}": { get: { summary: "Read one project-scoped durable job", security: [{ projectBearer: [] }] } },
    "/api/v1/jobs/{id}/events": { get: { summary: "Read ordered durable lifecycle events", security: [{ projectBearer: [] }] } },
    "/api/v1/jobs/{id}/artifacts": { get: { summary: "List SHA-verified immutable artifacts", security: [{ projectBearer: [] }] } },
    "/api/v1/jobs/{id}/artifacts/{name}": { get: { summary: "Download and rehash one allow-listed private artifact", security: [{ projectBearer: [] }] } },
    "/api/v1/jobs/{id}/cancel": { post: { summary: "Request cancellation with terminal-state precedence", security: [{ projectBearer: [] }] } },
    "/api/v1/inference-jobs": { post: { summary: "Alias for a durable gravity-independent galaxy inference job", security: [{ projectBearer: [] }] } },
    "/api/v1/generation-jobs": { post: { summary: "Alias for a durable forward galaxy generation job", security: [{ projectBearer: [] }] } },
    "/api/v1/runs": { post: { summary: "Score a formula without fitting object-specific gravity parameters" } },
  },
};

export default function handler(request, response) {
  if (options(request, response) || !requireMethod(request, response, "GET")) return;
  send(response, 200, specification);
}
