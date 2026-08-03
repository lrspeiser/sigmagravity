import { copyFile, mkdir } from "node:fs/promises";
import { resolve } from "node:path";

const root = resolve(import.meta.dirname, "..");
const dist = resolve(root, "dist");
await mkdir(resolve(dist, "assets"), { recursive: true });
await mkdir(resolve(dist, "data"), { recursive: true });
await mkdir(resolve(dist, "schemas"), { recursive: true });
await mkdir(resolve(dist, "examples", "models"), { recursive: true });
await mkdir(resolve(dist, "examples", "observation-targets"), { recursive: true });
for (const path of [
  "index.html",
  "assets/app.js",
  "assets/style.css",
  "assets/resolved-twin-development-atlas.png",
  "assets/resolved-twin-validation-atlas.png",
  "assets/resolved-twin-geometry-diagnostic-atlas.png",
  "assets/resolved-twin-holdout-atlas.png",
  "assets/resolved-twin-holdout-curves.png",
  "data/resolved-twin-development-v1.json",
  "schemas/model-manifest-v1.schema.json",
  "schemas/model-confirmation-request-v1.schema.json",
  "schemas/array-bundle-request-v1.schema.json",
  "schemas/field-job-request-v1.schema.json",
  "schemas/observation-target-v1.schema.json",
  "schemas/field-job-cli-v1.schema.json",
  "schemas/array-bundle-v1.schema.json",
  "schemas/data-upload-request-v1.schema.json",
  "schemas/field-job-submit-v1.schema.json",
  "schemas/observation-evaluation-job-submit-v1.schema.json",
  "schemas/galaxy-job-submit-v1.schema.json",
  "schemas/batch-submit-v1.schema.json",
  "examples/models/newtonian-poisson.json",
  "examples/models/aqual.json",
  "examples/models/qumond.json",
  "examples/models/refracted-gravity.json",
  "examples/models/two-potential.json",
  "examples/observation-targets/line-of-sight-velocity-field.json",
  "examples/observation-targets/photon-lensing-map.json",
  "examples/observation-targets/multiple-image-systems.json",
]) {
  await copyFile(resolve(root, path), resolve(dist, path));
}
console.log("built static workbench in dist");
