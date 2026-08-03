import { access, readFile } from "node:fs/promises";
import { resolve } from "node:path";

const root = resolve(import.meta.dirname, "..");
const requiredPaths = [
  "index.html",
  "guide.html",
  "assets/app.js",
  "assets/style.css",
  "assets/resolved-twin-development-atlas.png",
  "assets/resolved-twin-validation-atlas.png",
  "assets/resolved-twin-geometry-diagnostic-atlas.png",
  "assets/resolved-twin-holdout-atlas.png",
  "assets/resolved-twin-holdout-curves.png",
  "data/resolved-twin-development-v1.json",
  "data/sparc-v1.json",
  "dist/index.html",
  "dist/guide.html",
  "dist/assets/app.js",
  "dist/assets/style.css",
  "dist/assets/resolved-twin-development-atlas.png",
  "dist/assets/resolved-twin-validation-atlas.png",
  "dist/assets/resolved-twin-geometry-diagnostic-atlas.png",
  "dist/assets/resolved-twin-holdout-atlas.png",
  "dist/assets/resolved-twin-holdout-curves.png",
  "dist/data/resolved-twin-development-v1.json",
  "dist/schemas/model-manifest-v1.schema.json",
  "dist/schemas/model-confirmation-request-v1.schema.json",
  "dist/schemas/array-bundle-request-v1.schema.json",
  "dist/schemas/array-bundle-v1.schema.json",
  "dist/schemas/field-job-request-v1.schema.json",
  "dist/schemas/observation-target-v1.schema.json",
  "dist/schemas/field-job-cli-v1.schema.json",
  "dist/schemas/data-upload-request-v1.schema.json",
  "dist/schemas/field-job-submit-v1.schema.json",
  "dist/schemas/observation-evaluation-job-submit-v1.schema.json",
  "dist/schemas/galaxy-job-submit-v1.schema.json",
  "dist/schemas/inverse-response-job-submit-v1.schema.json",
  "dist/schemas/batch-submit-v1.schema.json",
  "dist/examples/models/refracted-gravity.json",
  "dist/examples/models/nonlocal-response.json",
  "dist/examples/observation-targets/photon-lensing-map.json",
  "api/v1/runs.mjs",
  "api/v1/twin-runs.mjs",
  "api/v1/resolved-twin-evidence.mjs",
  "api/v1/models/confirm.mjs",
  "api/v1/data-uploads.mjs",
  "api/v1/field-jobs.mjs",
  "api/v1/observation-evaluation-jobs.mjs",
  "api/v1/galaxy-jobs.mjs",
  "api/v1/inverse-response-jobs.mjs",
  "api/v1/batches.mjs",
  "lib/resolved-twin-evidence.mjs",
  "lib/local-field-job-service.mjs",
  "lib/observation-evaluation-preflight.mjs",
  "lib/galaxy-job-preflight.mjs",
  "lib/inverse-response-preflight.mjs",
  "lib/observation-target.mjs",
  "lib/batch-preflight.mjs",
  "lib/local-batch-service.mjs",
];

for (const path of requiredPaths) await access(resolve(root, path));

const guide = await readFile(resolve(root, "dist", "guide.html"), "utf8");
for (const phrase of [
  "What works where",
  "Inputs, outputs, and meaning",
  "What it cannot tell you yet",
  "Use halo maps only for discovery",
  "Nonlocal baryon-to-response convolution",
  "Inverse baryon-to-response discovery",
  "all_declared_families",
  "Why five controls",
  "A genuinely useful result is a prediction, not a reconstruction",
]) {
  if (!guide.includes(phrase)) throw new Error(`researcher guide is missing: ${phrase}`);
}

const catalog = JSON.parse(await readFile(resolve(root, "data", "sparc-v1.json"), "utf8"));
if (catalog.systems.length !== 175) throw new Error("catalog must contain all 175 SPARC galaxies");
const resolvedEvidence = JSON.parse(await readFile(resolve(root, "data", "resolved-twin-development-v1.json"), "utf8"));
if (resolvedEvidence.systems.length !== 8 || resolvedEvidence.sample.scoredVelocityPixels !== 146532) {
  throw new Error("resolved twin evidence must contain eight systems and 146,532 scored pixels");
}
const arrayBundleSchema = JSON.parse(await readFile(resolve(root, "dist", "schemas", "array-bundle-v1.schema.json"), "utf8"));
const scientificRoles = arrayBundleSchema.properties.arrays.items.properties.scientificRole.enum;
for (const role of ["baryonic_input", "model_derived_discovery_target", "raw_observation"]) {
  if (!scientificRoles.includes(role)) throw new Error(`array bundle schema is missing scientific role: ${role}`);
}
console.log(`verified static application, researcher guide, ${catalog.systems.length}-galaxy radial catalog, and ${resolvedEvidence.systems.length}-galaxy resolved evidence`);
