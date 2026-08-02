import { copyFile, mkdir } from "node:fs/promises";
import { resolve } from "node:path";

const root = resolve(import.meta.dirname, "..");
const dist = resolve(root, "dist");
await mkdir(resolve(dist, "assets"), { recursive: true });
for (const path of ["index.html", "assets/app.js", "assets/style.css"]) {
  await copyFile(resolve(root, path), resolve(dist, path));
}
console.log("built static workbench in dist");
