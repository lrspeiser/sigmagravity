import { createHash, createPublicKey, verify as verifySignature } from "node:crypto";
import { lstat, readFile, readdir, realpath } from "node:fs/promises";
import path from "node:path";

import { canonicalJson, canonicalize, sha256 } from "./canonical.mjs";

const HASH = /^[0-9a-f]{64}$/;
const KEY_ID = /^ed25519_[0-9a-f]{24}$/;
const VERSION = /^[0-9]+\.[0-9]+\.[0-9]+(?:-[A-Za-z0-9.-]+)?$/;
const SAFE_PATH = /^[A-Za-z0-9][A-Za-z0-9._/-]{0,159}$/;
const BASE64 = /^[A-Za-z0-9+/]+={0,2}$/;
const SIGNING_DOMAIN = "SIGMA_ADVANCED_PLUGIN_V1\n";

export const ADVANCED_PLUGIN_RUNTIME = Object.freeze({
  id: "sigma-python-plugin/1",
  pythonVersion: "3.13.7",
  packages: Object.freeze({ numpy: "2.2.6", scipy: "1.16.1" }),
});

export const ADVANCED_PLUGIN_POLICY = Object.freeze({
  maximumFiles: 64,
  maximumPackageBytes: 16 * 1024 * 1024,
  wallTimeSeconds: Object.freeze({ minimum: 1, maximum: 300 }),
  cpuSeconds: Object.freeze({ minimum: 1, maximum: 300 }),
  cpuCores: Object.freeze({ minimum: 0.25, maximum: 2 }),
  memoryMiB: Object.freeze({ minimum: 64, maximum: 2048 }),
  pids: Object.freeze({ minimum: 4, maximum: 128 }),
  stdoutBytes: Object.freeze({ minimum: 1024, maximum: 8 * 1024 * 1024 }),
  stderrBytes: Object.freeze({ minimum: 1024, maximum: 256 * 1024 }),
  temporaryBytes: Object.freeze({ minimum: 1024 * 1024, maximum: 64 * 1024 * 1024 }),
});

export const ADVANCED_PLUGIN_ISOLATION = Object.freeze({
  execution: "single_use_container",
  network: "none",
  rootFilesystem: "read_only",
  pluginMount: "read_only",
  datasetMount: "read_only",
  credentials: "none",
});

export class AdvancedPluginError extends Error {
  constructor(code, message, details = undefined) {
    super(message);
    this.name = "AdvancedPluginError";
    this.code = code;
    this.details = details;
  }
}

function fail(code, message, details) {
  throw new AdvancedPluginError(code, message, details);
}

function object(value, label) {
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    fail("invalid_plugin_manifest", `${label} must be an object`);
  }
  return value;
}

function integerWithin(value, range, label) {
  if (!Number.isSafeInteger(value) || value < range.minimum || value > range.maximum) {
    fail("invalid_plugin_resources", `${label} must be an integer from ${range.minimum} through ${range.maximum}`);
  }
  return value;
}

function numberWithin(value, range, label) {
  if (!Number.isFinite(value) || value < range.minimum || value > range.maximum) {
    fail("invalid_plugin_resources", `${label} must be from ${range.minimum} through ${range.maximum}`);
  }
  return value;
}

function safeRelativePath(value, label) {
  if (typeof value !== "string" || !SAFE_PATH.test(value) || value.includes("\\")) {
    fail("invalid_plugin_path", `${label} must be a portable relative path`);
  }
  const normalized = path.posix.normalize(value);
  if (normalized !== value || value.startsWith("/") || value.split("/").some((part) => part === "." || part === "..")) {
    fail("invalid_plugin_path", `${label} must not contain traversal or ambiguous path segments`);
  }
  return value;
}

function exactObject(value, expected, label) {
  object(value, label);
  if (canonicalJson(value) !== canonicalJson(expected)) {
    fail("unsupported_plugin_runtime", `${label} must exactly match the published sandbox contract`, { expected });
  }
  return canonicalize(value);
}

export function advancedPluginManifestCore(manifest) {
  object(manifest, "advanced plug-in manifest");
  if (manifest.schemaVersion !== "sigma-advanced-plugin/1") {
    fail("invalid_plugin_manifest", "schemaVersion must be sigma-advanced-plugin/1");
  }
  if (typeof manifest.name !== "string" || manifest.name.trim().length < 1 || manifest.name.trim().length > 120) {
    fail("invalid_plugin_manifest", "name must contain 1-120 characters");
  }
  if (typeof manifest.version !== "string" || !VERSION.test(manifest.version)) {
    fail("invalid_plugin_manifest", "version must be a semantic version");
  }
  if (typeof manifest.description !== "string" || manifest.description.length > 1000) {
    fail("invalid_plugin_manifest", "description must be a string of at most 1000 characters");
  }
  const publisher = object(manifest.publisher, "publisher");
  if (publisher.algorithm !== "ed25519" || typeof publisher.keyId !== "string" || !KEY_ID.test(publisher.keyId)) {
    fail("invalid_plugin_publisher", "publisher must name an Ed25519 key identifier");
  }
  const runtime = exactObject(manifest.runtime, ADVANCED_PLUGIN_RUNTIME, "runtime");
  const isolation = exactObject(manifest.isolation, ADVANCED_PLUGIN_ISOLATION, "isolation");
  const entrypoint = safeRelativePath(manifest.entrypoint, "entrypoint");
  if (!entrypoint.endsWith(".py")) fail("invalid_plugin_path", "entrypoint must be a Python file");
  if (!Array.isArray(manifest.files) || manifest.files.length < 1 || manifest.files.length > ADVANCED_PLUGIN_POLICY.maximumFiles) {
    fail("invalid_plugin_files", `files must contain 1-${ADVANCED_PLUGIN_POLICY.maximumFiles} records`);
  }
  const seen = new Set();
  let packageBytes = 0;
  const files = manifest.files.map((record, index) => {
    object(record, `files[${index}]`);
    const filePath = safeRelativePath(record.path, `files[${index}].path`);
    if (filePath === "plugin.json" || seen.has(filePath)) {
      fail("invalid_plugin_files", `${filePath} is reserved or duplicated`);
    }
    if (!filePath.endsWith(".py") && !filePath.endsWith(".json")) {
      fail("invalid_plugin_files", `${filePath} is not an allowed source-package type; binary inputs belong in the read-only dataset`);
    }
    seen.add(filePath);
    if (typeof record.sha256 !== "string" || !HASH.test(record.sha256)) {
      fail("invalid_plugin_files", `${filePath} must declare a lowercase SHA-256`);
    }
    if (!Number.isSafeInteger(record.bytes) || record.bytes < 0) {
      fail("invalid_plugin_files", `${filePath} must declare a nonnegative byte count`);
    }
    packageBytes += record.bytes;
    return { path: filePath, bytes: record.bytes, sha256: record.sha256 };
  }).sort((left, right) => left.path.localeCompare(right.path));
  if (!seen.has(entrypoint)) fail("invalid_plugin_files", "entrypoint must appear in files");
  if (packageBytes > ADVANCED_PLUGIN_POLICY.maximumPackageBytes) {
    fail("plugin_package_too_large", `declared files exceed ${ADVANCED_PLUGIN_POLICY.maximumPackageBytes} bytes`);
  }
  const resources = object(manifest.resources, "resources");
  const normalizedResources = {
    wallTimeSeconds: integerWithin(resources.wallTimeSeconds, ADVANCED_PLUGIN_POLICY.wallTimeSeconds, "wallTimeSeconds"),
    cpuSeconds: integerWithin(resources.cpuSeconds, ADVANCED_PLUGIN_POLICY.cpuSeconds, "cpuSeconds"),
    cpuCores: numberWithin(resources.cpuCores, ADVANCED_PLUGIN_POLICY.cpuCores, "cpuCores"),
    memoryMiB: integerWithin(resources.memoryMiB, ADVANCED_PLUGIN_POLICY.memoryMiB, "memoryMiB"),
    pids: integerWithin(resources.pids, ADVANCED_PLUGIN_POLICY.pids, "pids"),
    stdoutBytes: integerWithin(resources.stdoutBytes, ADVANCED_PLUGIN_POLICY.stdoutBytes, "stdoutBytes"),
    stderrBytes: integerWithin(resources.stderrBytes, ADVANCED_PLUGIN_POLICY.stderrBytes, "stderrBytes"),
    temporaryBytes: integerWithin(resources.temporaryBytes, ADVANCED_PLUGIN_POLICY.temporaryBytes, "temporaryBytes"),
  };
  const interfaceContract = object(manifest.interface, "interface");
  if (interfaceContract.inputSchemaVersion !== "sigma-advanced-plugin-input/1"
    || interfaceContract.outputSchemaVersion !== "sigma-advanced-plugin-output/1") {
    fail("invalid_plugin_interface", "plug-in interface schema versions are unsupported");
  }
  return canonicalize({
    schemaVersion: "sigma-advanced-plugin/1",
    name: manifest.name.trim(),
    version: manifest.version,
    description: manifest.description,
    publisher: { algorithm: "ed25519", keyId: publisher.keyId },
    runtime,
    isolation,
    entrypoint,
    files,
    resources: normalizedResources,
    interface: {
      inputSchemaVersion: interfaceContract.inputSchemaVersion,
      outputSchemaVersion: interfaceContract.outputSchemaVersion,
    },
  });
}

export function advancedPluginPackageSha256(manifest) {
  return sha256(advancedPluginManifestCore(manifest));
}

export function advancedPluginSigningBytes(manifest) {
  const core = advancedPluginManifestCore(manifest);
  const document = { ...core, packageSha256: sha256(core) };
  return Buffer.from(`${SIGNING_DOMAIN}${canonicalJson(document)}\n`, "utf8");
}

export function advancedPluginPublisherKeyId(publicKeyPem) {
  let key;
  try {
    key = createPublicKey(publicKeyPem);
  } catch {
    fail("invalid_plugin_public_key", "publisher public key is not valid PEM");
  }
  if (key.asymmetricKeyType !== "ed25519") {
    fail("invalid_plugin_public_key", "publisher public key must be Ed25519");
  }
  const der = key.export({ format: "der", type: "spki" });
  return `ed25519_${createHash("sha256").update(der).digest("hex").slice(0, 24)}`;
}

function decodeSignature(value) {
  if (typeof value !== "string" || value.length > 128 || !BASE64.test(value)) {
    fail("invalid_plugin_signature", "signature.valueBase64 must be canonical base64");
  }
  const bytes = Buffer.from(value, "base64");
  if (bytes.length !== 64 || bytes.toString("base64") !== value) {
    fail("invalid_plugin_signature", "Ed25519 signature must contain exactly 64 bytes");
  }
  return bytes;
}

function trustRecord(trustStore, keyId) {
  if (!trustStore) return null;
  if (trustStore.schemaVersion !== "sigma-plugin-trust-store/1" || !Array.isArray(trustStore.publishers)) {
    fail("invalid_plugin_trust_store", "trust store schema is invalid");
  }
  const matches = trustStore.publishers.filter((record) => record?.keyId === keyId && record?.status === "active");
  if (matches.length !== 1) fail("untrusted_plugin_publisher", "publisher key is not uniquely active in the operator trust store");
  if (matches[0].algorithm !== "ed25519") fail("invalid_plugin_trust_store", "active publisher record must declare Ed25519");
  return matches[0];
}

export function verifyAdvancedPluginManifest(manifest, {
  publicKeyPem = null,
  trustStore = null,
  requireTrustedPublisher = true,
} = {}) {
  const core = advancedPluginManifestCore(manifest);
  const packageSha256 = sha256(core);
  if (manifest.packageSha256 !== packageSha256) {
    fail("plugin_package_identity_mismatch", "packageSha256 does not match the canonical manifest core");
  }
  const signature = object(manifest.signature, "signature");
  if (signature.algorithm !== "ed25519" || signature.keyId !== core.publisher.keyId) {
    fail("invalid_plugin_signature", "signature algorithm and key identifier must match the publisher");
  }
  const trusted = trustRecord(trustStore, core.publisher.keyId);
  const selectedPem = trusted?.publicKeyPem ?? publicKeyPem;
  if (typeof selectedPem !== "string") {
    fail("plugin_public_key_required", "publisher public key is required for signature verification");
  }
  const computedKeyId = advancedPluginPublisherKeyId(selectedPem);
  if (computedKeyId !== core.publisher.keyId) {
    fail("plugin_publisher_identity_mismatch", "public key does not match the declared publisher key identifier");
  }
  let key;
  try {
    key = createPublicKey(selectedPem);
  } catch {
    fail("invalid_plugin_public_key", "publisher public key is not valid PEM");
  }
  const valid = verifySignature(null, advancedPluginSigningBytes(manifest), key, decodeSignature(signature.valueBase64));
  if (!valid) fail("invalid_plugin_signature", "Ed25519 signature does not verify");
  if (requireTrustedPublisher && !trusted) {
    fail("untrusted_plugin_publisher", "a valid self-signed package is not sufficient for execution");
  }
  return {
    schemaVersion: "sigma-advanced-plugin-verification/1",
    packageSha256,
    publisherKeyId: computedKeyId,
    signatureValid: true,
    publisherTrusted: Boolean(trusted),
    runtime: core.runtime,
    isolation: core.isolation,
    resources: core.resources,
    files: core.files,
    entrypoint: core.entrypoint,
  };
}

async function walkPackage(root, current = "") {
  const directory = path.join(root, ...current.split("/").filter(Boolean));
  const entries = await readdir(directory, { withFileTypes: true });
  const files = [];
  for (const entry of entries) {
    const relative = current ? `${current}/${entry.name}` : entry.name;
    const absolute = path.join(directory, entry.name);
    const stat = await lstat(absolute);
    if (stat.isSymbolicLink()) fail("plugin_package_symlink", `symbolic links are forbidden: ${relative}`);
    if (stat.isDirectory()) files.push(...await walkPackage(root, relative));
    else if (stat.isFile()) files.push(relative.replaceAll("\\", "/"));
    else fail("invalid_plugin_package_entry", `unsupported package entry: ${relative}`);
  }
  return files;
}

export async function verifyAdvancedPluginPackage({ packageDirectory, trustStore }) {
  const suppliedStat = await lstat(packageDirectory);
  if (suppliedStat.isSymbolicLink()) fail("plugin_package_symlink", "package root must not be a symbolic link");
  const packageRoot = await realpath(packageDirectory);
  const stat = await lstat(packageRoot);
  if (!stat.isDirectory() || stat.isSymbolicLink()) fail("invalid_plugin_package", "package path must be a real directory");
  let manifest;
  try {
    manifest = JSON.parse(await readFile(path.join(packageRoot, "plugin.json"), "utf8"));
  } catch {
    fail("invalid_plugin_package", "plugin.json is missing or invalid JSON");
  }
  const verification = verifyAdvancedPluginManifest(manifest, { trustStore, requireTrustedPublisher: true });
  const actualFiles = (await walkPackage(packageRoot)).sort();
  const expectedFiles = ["plugin.json", ...verification.files.map((record) => record.path)].sort();
  if (canonicalJson(actualFiles) !== canonicalJson(expectedFiles)) {
    fail("plugin_package_file_set_mismatch", "package contains missing or undeclared files", { expectedFiles, actualFiles });
  }
  for (const record of verification.files) {
    const absolute = path.join(packageRoot, ...record.path.split("/"));
    const fileStat = await lstat(absolute);
    if (!fileStat.isFile() || fileStat.isSymbolicLink() || fileStat.size !== record.bytes) {
      fail("plugin_file_identity_mismatch", `${record.path} byte identity does not match its declaration`);
    }
    const bytes = await readFile(absolute);
    const digest = createHash("sha256").update(bytes).digest("hex");
    if (digest !== record.sha256) fail("plugin_file_identity_mismatch", `${record.path} SHA-256 does not match its declaration`);
  }
  return { packageRoot, manifest, verification };
}

export function buildAdvancedPluginDockerArgs({
  image,
  containerName,
  packageRoot,
  dataRoot,
  verification,
  allowUnpinnedImage = false,
}) {
  if (typeof image !== "string" || (!allowUnpinnedImage && !/@sha256:[0-9a-f]{64}$/.test(image))) {
    fail("unpinned_plugin_image", "production plug-in execution requires an allow-listed image digest");
  }
  if (typeof containerName !== "string" || !/^sigma-plugin-[a-z0-9-]{16,80}$/.test(containerName)) {
    fail("invalid_plugin_container_name", "container name is invalid");
  }
  const resources = verification.resources;
  const memory = `${resources.memoryMiB}m`;
  return [
    "run", "--rm",
    "--name", containerName,
    "--pull", "never",
    "--network", "none",
    "--ipc", "none",
    "--read-only",
    "--cap-drop", "ALL",
    "--security-opt", "no-new-privileges=true",
    "--user", "65532:65532",
    "--hostname", "sigma-plugin",
    "--pids-limit", String(resources.pids),
    "--memory", memory,
    "--memory-swap", memory,
    "--cpus", String(resources.cpuCores),
    "--ulimit", `cpu=${resources.cpuSeconds}:${resources.cpuSeconds}`,
    "--ulimit", "nofile=64:64",
    "--ulimit", `fsize=${resources.temporaryBytes}:${resources.temporaryBytes}`,
    "--mount", `type=bind,src=${packageRoot},dst=/plugin,readonly`,
    "--mount", `type=bind,src=${dataRoot},dst=/data,readonly`,
    "--tmpfs", `/tmp:rw,noexec,nosuid,nodev,size=${resources.temporaryBytes},uid=65532,gid=65532,mode=700`,
    "--label", `org.sigmagravity.plugin.sha256=${verification.packageSha256}`,
    image,
  ];
}
