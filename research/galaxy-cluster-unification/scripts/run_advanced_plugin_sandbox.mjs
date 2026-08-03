#!/usr/bin/env node
import { readFile, writeFile } from "node:fs/promises";
import path from "node:path";

import { canonicalJson } from "../hosted-simulator/lib/canonical.mjs";
import { runAdvancedPluginSandbox } from "../hosted-simulator/lib/advanced-plugin-launcher.mjs";

function argumentsFrom(argv) {
  const values = new Map();
  for (let index = 0; index < argv.length; index += 2) {
    const key = argv[index];
    const value = argv[index + 1];
    if (!key?.startsWith("--") || value === undefined) throw new Error(`invalid argument near ${key ?? "end of input"}`);
    values.set(key.slice(2), value);
  }
  return values;
}

const args = argumentsFrom(process.argv.slice(2));
for (const required of ["package", "data", "trust-store", "image", "output"]) {
  if (!args.has(required)) throw new Error(`--${required} is required`);
}
const trustStore = JSON.parse(await readFile(path.resolve(args.get("trust-store")), "utf8"));
const result = await runAdvancedPluginSandbox({
  packageDirectory: path.resolve(args.get("package")),
  dataDirectory: path.resolve(args.get("data")),
  trustStore,
  image: args.get("image"),
  dockerCommand: args.get("docker") ?? "docker",
  allowUnpinnedImage: args.get("allow-unpinned-image") === "true",
});
await writeFile(path.resolve(args.get("output")), `${canonicalJson(result)}\n`, { flag: "wx" });
process.stdout.write(`${canonicalJson({
  schemaVersion: "sigma-advanced-plugin-host-result/1",
  state: "succeeded",
  packageSha256: result.verification.packageSha256,
  inputSha256: result.execution.inputSha256,
  outputPath: path.resolve(args.get("output")),
})}\n`);
