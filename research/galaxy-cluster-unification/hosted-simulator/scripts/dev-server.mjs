import { createServer } from "node:http";
import { readFile } from "node:fs/promises";
import { resolve } from "node:path";
import datasets from "../api/v1/datasets.mjs";
import formulasValidate from "../api/v1/formulas/validate.mjs";
import fieldJobsPrepare from "../api/v1/field-jobs/prepare.mjs";
import health from "../api/v1/health.mjs";
import storageReadiness from "../api/v1/storage-readiness.mjs";
import queueCanary from "../api/v1/queue-canary.mjs";
import modelsValidate from "../api/v1/models/validate.mjs";
import modelsConfirm from "../api/v1/models/confirm.mjs";
import pluginsPreflight from "../api/v1/plugins/preflight.mjs";
import runs from "../api/v1/runs.mjs";
import specification from "../api/v1/spec.mjs";
import syntheticGalaxies from "../api/v1/synthetic-galaxies.mjs";
import twinRuns from "../api/v1/twin-runs.mjs";
import resolvedTwinEvidence from "../api/v1/resolved-twin-evidence.mjs";
import clusterEvidence from "../api/v1/cluster-evidence.mjs";
import system from "../api/v1/system.mjs";
import systems from "../api/v1/systems.mjs";
import { createLocalFieldJobRouter } from "../lib/local-field-job-http.mjs";
import { LocalFieldJobService } from "../lib/local-field-job-service.mjs";
import { createLocalBatchRouter } from "../lib/local-batch-http.mjs";
import { LocalBatchService } from "../lib/local-batch-service.mjs";

const root = resolve(import.meta.dirname, "..");
const port = Number(process.env.PORT ?? 4173);
const host = process.env.HOST ?? "127.0.0.1";
const projectRoot = resolve(root, "..");
const localService = new LocalFieldJobService({
  root: process.env.SIMULATOR_LOCAL_STORE ?? resolve(projectRoot, "tmp", "hosted-field-job-service"),
  projectRoot,
});
await localService.initialize();
const localFieldJobs = createLocalFieldJobRouter(localService);
const localBatchService = new LocalBatchService({
  root: process.env.SIMULATOR_LOCAL_STORE ?? resolve(projectRoot, "tmp", "hosted-field-job-service"),
  fieldService: localService,
});
await localBatchService.initialize();
const localBatches = createLocalBatchRouter(localBatchService);
const apiRoutes = new Map([
  ["/api/v1/health", health],
  ["/api/v1/storage-readiness", storageReadiness],
  ["/api/v1/queue-canary", queueCanary],
  ["/api/v1/datasets", datasets],
  ["/api/v1/systems", systems],
  ["/api/v1/formulas/validate", formulasValidate],
  ["/api/v1/field-jobs/prepare", fieldJobsPrepare],
  ["/api/v1/models/validate", modelsValidate],
  ["/api/v1/models/confirm", modelsConfirm],
  ["/api/v1/plugins/preflight", pluginsPreflight],
  ["/api/v1/synthetic-galaxies", syntheticGalaxies],
  ["/api/v1/twin-runs", twinRuns],
  ["/api/v1/resolved-twin-evidence", resolvedTwinEvidence],
  ["/api/v1/cluster-evidence", clusterEvidence],
  ["/api/v1/runs", runs],
  ["/api/v1/openapi.json", specification],
]);
const staticFiles = new Map([
  ["/", ["index.html", "text/html; charset=utf-8"]],
  ["/index.html", ["index.html", "text/html; charset=utf-8"]],
  ["/guide", ["guide.html", "text/html; charset=utf-8"]],
  ["/guide.html", ["guide.html", "text/html; charset=utf-8"]],
  ["/assets/app.js", ["assets/app.js", "text/javascript; charset=utf-8"]],
  ["/assets/style.css", ["assets/style.css", "text/css; charset=utf-8"]],
  ["/assets/resolved-twin-development-atlas.png", ["assets/resolved-twin-development-atlas.png", "image/png"]],
  ["/assets/resolved-twin-validation-atlas.png", ["assets/resolved-twin-validation-atlas.png", "image/png"]],
  ["/assets/resolved-twin-geometry-diagnostic-atlas.png", ["assets/resolved-twin-geometry-diagnostic-atlas.png", "image/png"]],
  ["/assets/resolved-twin-holdout-atlas.png", ["assets/resolved-twin-holdout-atlas.png", "image/png"]],
  ["/assets/resolved-twin-holdout-curves.png", ["assets/resolved-twin-holdout-curves.png", "image/png"]],
  ["/data/resolved-twin-development-v1.json", ["data/resolved-twin-development-v1.json", "application/json; charset=utf-8"]],
  ["/data/resolved-cluster-evidence-v1.json", ["data/resolved-cluster-evidence-v1.json", "application/json; charset=utf-8"]],
  ["/schemas/model-manifest-v1.schema.json", ["schemas/model-manifest-v1.schema.json", "application/schema+json; charset=utf-8"]],
  ["/schemas/model-confirmation-request-v1.schema.json", ["schemas/model-confirmation-request-v1.schema.json", "application/schema+json; charset=utf-8"]],
  ["/schemas/array-bundle-request-v1.schema.json", ["schemas/array-bundle-request-v1.schema.json", "application/schema+json; charset=utf-8"]],
  ["/schemas/array-bundle-v1.schema.json", ["schemas/array-bundle-v1.schema.json", "application/schema+json; charset=utf-8"]],
  ["/schemas/field-job-request-v1.schema.json", ["schemas/field-job-request-v1.schema.json", "application/schema+json; charset=utf-8"]],
  ["/schemas/observation-target-v1.schema.json", ["schemas/observation-target-v1.schema.json", "application/schema+json; charset=utf-8"]],
  ["/schemas/field-job-cli-v1.schema.json", ["schemas/field-job-cli-v1.schema.json", "application/schema+json; charset=utf-8"]],
  ["/schemas/data-upload-request-v1.schema.json", ["schemas/data-upload-request-v1.schema.json", "application/schema+json; charset=utf-8"]],
  ["/schemas/field-job-submit-v1.schema.json", ["schemas/field-job-submit-v1.schema.json", "application/schema+json; charset=utf-8"]],
  ["/schemas/observation-evaluation-job-submit-v1.schema.json", ["schemas/observation-evaluation-job-submit-v1.schema.json", "application/schema+json; charset=utf-8"]],
  ["/schemas/galaxy-job-submit-v1.schema.json", ["schemas/galaxy-job-submit-v1.schema.json", "application/schema+json; charset=utf-8"]],
  ["/schemas/inverse-response-job-submit-v1.schema.json", ["schemas/inverse-response-job-submit-v1.schema.json", "application/schema+json; charset=utf-8"]],
  ["/schemas/batch-submit-v1.schema.json", ["schemas/batch-submit-v1.schema.json", "application/schema+json; charset=utf-8"]],
  ["/schemas/production-model-registration-v1.schema.json", ["schemas/production-model-registration-v1.schema.json", "application/schema+json; charset=utf-8"]],
  ["/schemas/production-upload-registration-v1.schema.json", ["schemas/production-upload-registration-v1.schema.json", "application/schema+json; charset=utf-8"]],
  ["/schemas/production-job-submit-v1.schema.json", ["schemas/production-job-submit-v1.schema.json", "application/schema+json; charset=utf-8"]],
  ["/schemas/advanced-plugin-v1.schema.json", ["schemas/advanced-plugin-v1.schema.json", "application/schema+json; charset=utf-8"]],
  ["/examples/models/newtonian-poisson.json", ["examples/models/newtonian-poisson.json", "application/json; charset=utf-8"]],
  ["/examples/models/aqual.json", ["examples/models/aqual.json", "application/json; charset=utf-8"]],
  ["/examples/models/qumond.json", ["examples/models/qumond.json", "application/json; charset=utf-8"]],
  ["/examples/models/refracted-gravity.json", ["examples/models/refracted-gravity.json", "application/json; charset=utf-8"]],
  ["/examples/models/nonlocal-response.json", ["examples/models/nonlocal-response.json", "application/json; charset=utf-8"]],
  ["/examples/models/two-potential.json", ["examples/models/two-potential.json", "application/json; charset=utf-8"]],
]);

async function body(request, url) {
  const chunks = [];
  let bytes = 0;
  const binaryUpload = /^\/api\/v1\/data-uploads\/upload_[0-9a-f]{24}\/content$/.test(url.pathname);
  const limit = binaryUpload ? localService.maxUploadBytes : 1_000_000;
  for await (const chunk of request) {
    bytes += chunk.length;
    if (bytes > limit) throw new Error(`request body exceeds ${limit} byte local limit`);
    chunks.push(chunk);
  }
  if (!chunks.length) return undefined;
  const payload = Buffer.concat(chunks);
  return binaryUpload ? payload : JSON.parse(payload.toString("utf8"));
}

function adaptResponse(response) {
  response.status = (code) => { response.statusCode = code; return response; };
  response.json = (payload) => { response.end(JSON.stringify(payload)); return response; };
  return response;
}

const server = createServer(async (request, rawResponse) => {
  const response = adaptResponse(rawResponse);
  try {
    const url = new URL(request.url, `http://${request.headers.host ?? `${host}:${port}`}`);
    request.query = Object.fromEntries(url.searchParams.entries());
    request.body = await body(request, url);
    if (await localBatches(request, response, url)) return;
    if (await localFieldJobs(request, response, url)) return;
    const staticEntry = staticFiles.get(url.pathname);
    if (staticEntry) {
      const [path, contentType] = staticEntry;
      response.setHeader("Content-Type", contentType);
      response.setHeader("Cache-Control", "no-store");
      response.end(await readFile(resolve(root, path)));
      return;
    }
    if (url.pathname === "/favicon.ico") { response.statusCode = 204; response.end(); return; }

    let handler = apiRoutes.get(url.pathname);
    const detailMatch = url.pathname.match(/^\/api\/v1\/systems\/([^/]+)$/);
    if (detailMatch) handler = system;
    if (!handler) {
      response.statusCode = 404;
      response.setHeader("Content-Type", "application/json; charset=utf-8");
      response.end(JSON.stringify({ error: "not_found" }));
      return;
    }
    if (detailMatch) request.query.id = decodeURIComponent(detailMatch[1]);
    await handler(request, response);
  } catch (error) {
    if (!response.headersSent) response.setHeader("Content-Type", "application/json; charset=utf-8");
    response.statusCode = 400;
    response.end(JSON.stringify({ error: "bad_request", message: error.message }));
  }
});

server.listen(port, host, () => {
  console.log(`Sigma Gravity simulator listening at http://${host}:${port}`);
});

for (const signal of ["SIGINT", "SIGTERM"]) {
  process.on(signal, () => server.close(async () => {
    await localBatchService.close();
    await localService.close();
    process.exit(0);
  }));
}
