#!/usr/bin/env node
import { createHash, generateKeyPairSync, sign } from "node:crypto";
import { mkdir, readFile, writeFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

import {
  ADVANCED_PLUGIN_ISOLATION,
  ADVANCED_PLUGIN_RUNTIME,
  advancedPluginManifestCore,
  advancedPluginPublisherKeyId,
  advancedPluginSigningBytes,
} from "../hosted-simulator/lib/advanced-plugin.mjs";
import { canonicalJson, sha256 } from "../hosted-simulator/lib/canonical.mjs";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const DEFAULT_SOURCE = path.resolve(HERE, "../plugin-sandbox/fixtures/fixed_mond_plugin.py");

function contentSha256(bytes) {
  return createHash("sha256").update(bytes).digest("hex");
}

export async function buildAdvancedPluginFixture({
  destination,
  sourceBytes = null,
  resources = {},
  request = null,
  name = "External fixed simple-MOND acceptance plug-in",
  version = "1.0.0",
} = {}) {
  if (!destination) throw new Error("destination is required");
  const root = path.resolve(destination);
  const packageDirectory = path.join(root, "package");
  const dataDirectory = path.join(root, "data");
  await mkdir(packageDirectory, { recursive: true });
  await mkdir(dataDirectory, { recursive: true });
  const pluginBytes = sourceBytes ?? await readFile(DEFAULT_SOURCE);
  await writeFile(path.join(packageDirectory, "plugin.py"), pluginBytes, { flag: "wx" });
  const { publicKey, privateKey } = generateKeyPairSync("ed25519");
  const publicKeyPem = publicKey.export({ type: "spki", format: "pem" });
  const keyId = advancedPluginPublisherKeyId(publicKeyPem);
  const unsigned = {
    schemaVersion: "sigma-advanced-plugin/1",
    name,
    version,
    description: "CI fixture proving an external signed implementation can reproduce the safe fixed-MOND comparator.",
    publisher: { algorithm: "ed25519", keyId },
    runtime: ADVANCED_PLUGIN_RUNTIME,
    isolation: ADVANCED_PLUGIN_ISOLATION,
    entrypoint: "plugin.py",
    files: [{ path: "plugin.py", bytes: pluginBytes.length, sha256: contentSha256(pluginBytes) }],
    resources: {
      wallTimeSeconds: 8,
      cpuSeconds: 8,
      cpuCores: 0.5,
      memoryMiB: 256,
      pids: 32,
      stdoutBytes: 256 * 1024,
      stderrBytes: 64 * 1024,
      temporaryBytes: 8 * 1024 * 1024,
      ...resources,
    },
    interface: {
      inputSchemaVersion: "sigma-advanced-plugin-input/1",
      outputSchemaVersion: "sigma-advanced-plugin-output/1",
    },
  };
  const core = advancedPluginManifestCore(unsigned);
  const packageSha256 = sha256(core);
  const manifestForSignature = { ...core, packageSha256 };
  const signatureValue = sign(null, advancedPluginSigningBytes(manifestForSignature), privateKey).toString("base64");
  const manifest = {
    ...manifestForSignature,
    signature: { algorithm: "ed25519", keyId, valueBase64: signatureValue },
  };
  const trustStore = {
    schemaVersion: "sigma-plugin-trust-store/1",
    publishers: [{ keyId, algorithm: "ed25519", status: "active", publicKeyPem }],
  };
  const input = request ?? {
    schemaVersion: "sigma-advanced-plugin-input/1",
    gBarMps2: [1e-13, 1e-12, 1e-11, 1e-10, 1e-9],
    a0Mps2: 1.2e-10,
  };
  const trustStorePath = path.join(root, "trust-store.json");
  await writeFile(path.join(packageDirectory, "plugin.json"), `${canonicalJson(manifest)}\n`, { flag: "wx" });
  await writeFile(path.join(dataDirectory, "request.json"), `${canonicalJson(input)}\n`, { flag: "wx" });
  await writeFile(trustStorePath, `${canonicalJson(trustStore)}\n`, { flag: "wx" });
  return { root, packageDirectory, dataDirectory, trustStorePath, trustStore, manifest, input };
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  const destinationIndex = process.argv.indexOf("--destination");
  if (destinationIndex < 0 || !process.argv[destinationIndex + 1]) throw new Error("--destination is required");
  const built = await buildAdvancedPluginFixture({ destination: process.argv[destinationIndex + 1] });
  process.stdout.write(`${canonicalJson({
    schemaVersion: "sigma-advanced-plugin-fixture/1",
    packageDirectory: built.packageDirectory,
    dataDirectory: built.dataDirectory,
    trustStorePath: built.trustStorePath,
    packageSha256: built.manifest.packageSha256,
  })}\n`);
}
