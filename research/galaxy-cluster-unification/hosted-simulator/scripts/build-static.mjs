import { copyFile, mkdir } from "node:fs/promises";
import { resolve } from "node:path";

const root = resolve(import.meta.dirname, "..");
const dist = resolve(root, "dist");
await mkdir(resolve(dist, "assets"), { recursive: true });
await mkdir(resolve(dist, "schemas"), { recursive: true });
await mkdir(resolve(dist, "examples", "models"), { recursive: true });
for (const path of [
  "index.html",
  "assets/app.js",
  "assets/style.css",
  "schemas/model-manifest-v1.schema.json",
  "schemas/array-bundle-request-v1.schema.json",
  "schemas/field-job-request-v1.schema.json",
  "schemas/field-job-cli-v1.schema.json",
  "schemas/array-bundle-v1.schema.json",
  "schemas/data-upload-request-v1.schema.json",
  "schemas/field-job-submit-v1.schema.json",
  "schemas/galaxy-job-submit-v1.schema.json",
  "examples/models/newtonian-poisson.json",
  "examples/models/aqual.json",
  "examples/models/qumond.json",
  "examples/models/refracted-gravity.json",
  "examples/models/two-potential.json",
]) {
  await copyFile(resolve(root, path), resolve(dist, path));
}
console.log("built static workbench in dist");
