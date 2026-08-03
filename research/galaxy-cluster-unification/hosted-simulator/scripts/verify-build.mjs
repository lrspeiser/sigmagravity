import { access, readFile } from "node:fs/promises";
import { resolve } from "node:path";

const root = resolve(import.meta.dirname, "..");
for (const path of ["index.html", "assets/app.js", "assets/style.css", "assets/resolved-twin-development-atlas.png", "assets/resolved-twin-validation-atlas.png", "assets/resolved-twin-geometry-diagnostic-atlas.png", "data/resolved-twin-development-v1.json", "dist/index.html", "dist/assets/app.js", "dist/assets/style.css", "dist/assets/resolved-twin-development-atlas.png", "dist/assets/resolved-twin-validation-atlas.png", "dist/assets/resolved-twin-geometry-diagnostic-atlas.png", "dist/data/resolved-twin-development-v1.json", "dist/schemas/model-manifest-v1.schema.json", "dist/schemas/array-bundle-request-v1.schema.json", "dist/schemas/array-bundle-v1.schema.json", "dist/schemas/field-job-request-v1.schema.json", "dist/schemas/observation-target-v1.schema.json", "dist/schemas/field-job-cli-v1.schema.json", "dist/schemas/data-upload-request-v1.schema.json", "dist/schemas/field-job-submit-v1.schema.json", "dist/schemas/observation-evaluation-job-submit-v1.schema.json", "dist/schemas/galaxy-job-submit-v1.schema.json", "dist/schemas/batch-submit-v1.schema.json", "dist/examples/models/refracted-gravity.json", "dist/examples/observation-targets/photon-lensing-map.json", "data/sparc-v1.json", "api/v1/runs.mjs", "api/v1/twin-runs.mjs", "api/v1/resolved-twin-evidence.mjs", "api/v1/data-uploads.mjs", "api/v1/field-jobs.mjs", "api/v1/observation-evaluation-jobs.mjs", "api/v1/galaxy-jobs.mjs", "api/v1/batches.mjs", "lib/resolved-twin-evidence.mjs", "lib/local-field-job-service.mjs", "lib/observation-evaluation-preflight.mjs", "lib/galaxy-job-preflight.mjs", "lib/observation-target.mjs", "lib/batch-preflight.mjs", "lib/local-batch-service.mjs"]) {
  await access(resolve(root, path));
}
const catalog = JSON.parse(await readFile(resolve(root, "data", "sparc-v1.json"), "utf8"));
if (catalog.systems.length !== 175) throw new Error("catalog must contain all 175 SPARC galaxies");
const resolvedEvidence = JSON.parse(await readFile(resolve(root, "data", "resolved-twin-development-v1.json"), "utf8"));
if (resolvedEvidence.systems.length !== 6 || resolvedEvidence.sample.scoredVelocityPixels !== 107211) {
  throw new Error("resolved twin evidence must contain six systems and 107,211 scored pixels");
}
console.log(`verified static application, ${catalog.systems.length}-galaxy radial catalog, and ${resolvedEvidence.systems.length}-galaxy resolved evidence`);
