import { access, readFile } from "node:fs/promises";
import { resolve } from "node:path";

const root = resolve(import.meta.dirname, "..");
for (const path of ["index.html", "assets/app.js", "assets/style.css", "dist/index.html", "dist/assets/app.js", "dist/assets/style.css", "dist/schemas/model-manifest-v1.schema.json", "dist/examples/models/refracted-gravity.json", "data/sparc-v1.json", "api/v1/runs.mjs"]) {
  await access(resolve(root, path));
}
const catalog = JSON.parse(await readFile(resolve(root, "data", "sparc-v1.json"), "utf8"));
if (catalog.systems.length !== 175) throw new Error("catalog must contain all 175 SPARC galaxies");
console.log(`verified static application and ${catalog.systems.length}-galaxy catalog`);
