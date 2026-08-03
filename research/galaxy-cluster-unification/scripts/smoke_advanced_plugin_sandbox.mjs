#!/usr/bin/env node
import assert from "node:assert/strict";
import { mkdtemp, readFile, writeFile } from "node:fs/promises";
import os from "node:os";
import path from "node:path";

import { AdvancedPluginError } from "../hosted-simulator/lib/advanced-plugin.mjs";
import { runAdvancedPluginSandbox } from "../hosted-simulator/lib/advanced-plugin-launcher.mjs";
import { compileFormula, FIXED_MOND_FORMULA } from "../hosted-simulator/lib/formula.mjs";
import { buildAdvancedPluginFixture } from "./build_advanced_plugin_fixture.mjs";

const image = process.env.SIGMA_PLUGIN_SANDBOX_IMAGE;
assert.ok(image, "SIGMA_PLUGIN_SANDBOX_IMAGE is required");
const root = await mkdtemp(path.join(os.tmpdir(), "sigma-plugin-smoke-"));
const fixture = await buildAdvancedPluginFixture({ destination: path.join(root, "valid") });

async function execute() {
  return runAdvancedPluginSandbox({
    packageDirectory: fixture.packageDirectory,
    dataDirectory: fixture.dataDirectory,
    trustStore: fixture.trustStore,
    image,
    allowUnpinnedImage: process.env.SIGMA_PLUGIN_ALLOW_UNPINNED_IMAGE === "true",
  });
}

const first = await execute();
const second = await execute();
const compiled = compileFormula(FIXED_MOND_FORMULA);
const expected = fixture.input.gBarMps2.map((gBar) => compiled.evaluate({ g_bar: gBar }));
const predicted = first.execution.output.result.accelerationMps2;
assert.equal(predicted.length, expected.length);
for (let index = 0; index < expected.length; index += 1) {
  assert.ok(Math.abs(predicted[index] - expected[index]) <= Math.max(1e-24, Math.abs(expected[index]) * 1e-12));
}
for (const run of [first, second]) {
  const observations = run.execution.output.sandboxObservations;
  assert.equal(observations.uid, 65532);
  assert.equal(observations.gid, 65532);
  assert.equal(observations.effectiveCapabilitiesHex, "0000000000000000");
  assert.equal(observations.noNewPrivileges, "1");
  assert.equal(observations.networkBlocked, true);
  assert.equal(observations.datasetWriteBlocked, true);
  assert.equal(observations.pluginWriteBlocked, true);
  assert.equal(observations.rootWriteBlocked, true);
  assert.equal(observations.dockerSocketAbsent, true);
  assert.deepEqual(observations.hostSecretEnvironmentNames, []);
  assert.equal(observations.sentinelExistedBeforeRun, false);
}

const pluginPath = path.join(fixture.packageDirectory, "plugin.py");
const original = await readFile(pluginPath);
await writeFile(pluginPath, Buffer.concat([original, Buffer.from("\n# tampered\n")]));
await assert.rejects(execute(), (error) => error instanceof AdvancedPluginError && error.code === "plugin_file_identity_mismatch");

process.stdout.write(`${JSON.stringify({
  schemaVersion: "sigma-advanced-plugin-container-acceptance/1",
  state: "pass",
  packageSha256: first.verification.packageSha256,
  safeFormulaSha256: compiled.formulaSha256,
  runs: 2,
  isolation: first.execution.output.sandboxObservations,
  tamperRejectedBeforeExecution: true,
})}\n`);
