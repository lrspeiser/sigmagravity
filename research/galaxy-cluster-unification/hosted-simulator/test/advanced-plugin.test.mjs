import assert from "node:assert/strict";
import { mkdtemp, readFile, rm, symlink, writeFile } from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import test from "node:test";

import preflightHandler from "../api/v1/plugins/preflight.mjs";
import {
  AdvancedPluginError,
  buildAdvancedPluginDockerArgs,
  verifyAdvancedPluginManifest,
  verifyAdvancedPluginPackage,
} from "../lib/advanced-plugin.mjs";
import { buildAdvancedPluginFixture } from "../../scripts/build_advanced_plugin_fixture.mjs";

async function fixture(t) {
  const root = await mkdtemp(path.join(os.tmpdir(), "sigma-plugin-unit-"));
  t.after(() => rm(root, { recursive: true, force: true }));
  return buildAdvancedPluginFixture({ destination: root });
}

function call(handler, request) {
  return new Promise((resolve, reject) => {
    const response = {
      headers: {},
      setHeader(name, value) { this.headers[name] = value; },
      status(code) { this.statusCode = code; return this; },
      json(body) { this.body = body; resolve(this); return this; },
      end() { resolve(this); },
    };
    Promise.resolve(handler(request, response)).catch(reject);
  });
}

test("Ed25519 manifest identity is valid but operator trust remains a separate gate", async (t) => {
  const built = await fixture(t);
  const publicKeyPem = built.trustStore.publishers[0].publicKeyPem;
  const selfVerified = verifyAdvancedPluginManifest(built.manifest, {
    publicKeyPem,
    requireTrustedPublisher: false,
  });
  assert.equal(selfVerified.signatureValid, true);
  assert.equal(selfVerified.publisherTrusted, false);
  const trusted = verifyAdvancedPluginManifest(built.manifest, { trustStore: built.trustStore });
  assert.equal(trusted.publisherTrusted, true);
  assert.equal(trusted.packageSha256, built.manifest.packageSha256);
  assert.throws(
    () => verifyAdvancedPluginManifest(built.manifest, {
      trustStore: { schemaVersion: "sigma-plugin-trust-store/1", publishers: [] },
    }),
    (error) => error instanceof AdvancedPluginError && error.code === "untrusted_plugin_publisher",
  );
  const changedSignature = {
    ...built.manifest,
    signature: { ...built.manifest.signature, valueBase64: `${built.manifest.signature.valueBase64.slice(0, -2)}AA` },
  };
  assert.throws(
    () => verifyAdvancedPluginManifest(changedSignature, { trustStore: built.trustStore }),
    (error) => error instanceof AdvancedPluginError && error.code === "invalid_plugin_signature",
  );
});

test("package verification rehashes every byte and rejects extras and links", async (t) => {
  const built = await fixture(t);
  const verified = await verifyAdvancedPluginPackage({ packageDirectory: built.packageDirectory, trustStore: built.trustStore });
  assert.equal(verified.verification.packageSha256, built.manifest.packageSha256);
  const pluginPath = path.join(built.packageDirectory, "plugin.py");
  const original = await readFile(pluginPath);
  await writeFile(pluginPath, Buffer.concat([original, Buffer.from("\n# changed\n")]));
  await assert.rejects(
    verifyAdvancedPluginPackage({ packageDirectory: built.packageDirectory, trustStore: built.trustStore }),
    (error) => error instanceof AdvancedPluginError && error.code === "plugin_file_identity_mismatch",
  );
  await writeFile(pluginPath, original);
  const extra = path.join(built.packageDirectory, "undeclared.py");
  await writeFile(extra, "print('undeclared')\n");
  await assert.rejects(
    verifyAdvancedPluginPackage({ packageDirectory: built.packageDirectory, trustStore: built.trustStore }),
    (error) => error instanceof AdvancedPluginError && error.code === "plugin_package_file_set_mismatch",
  );
  await rm(extra);
  try {
    await symlink(pluginPath, path.join(built.packageDirectory, "linked.py"));
    await assert.rejects(
      verifyAdvancedPluginPackage({ packageDirectory: built.packageDirectory, trustStore: built.trustStore }),
      (error) => error instanceof AdvancedPluginError && error.code === "plugin_package_symlink",
    );
  } catch (error) {
    if (error?.code !== "EPERM") throw error;
  }
});

test("container arguments enforce the isolated one-use policy and a pinned production image", async (t) => {
  const built = await fixture(t);
  const verified = await verifyAdvancedPluginPackage({ packageDirectory: built.packageDirectory, trustStore: built.trustStore });
  const image = `registry.example/sigma-plugin@sha256:${"a".repeat(64)}`;
  const args = buildAdvancedPluginDockerArgs({
    image,
    containerName: "sigma-plugin-0123456789abcdef-0123456789",
    packageRoot: verified.packageRoot,
    dataRoot: built.dataDirectory,
    verification: verified.verification,
  });
  const joined = args.join(" ");
  for (const required of [
    "--rm", "--network none", "--ipc none", "--read-only", "--cap-drop ALL",
    "--security-opt no-new-privileges=true", "--user 65532:65532", "--pids-limit",
    "--memory", "--memory-swap", "--cpus", "--ulimit cpu=", "--pull never",
    "dst=/plugin,readonly", "dst=/data,readonly", "--tmpfs /tmp:rw,noexec,nosuid,nodev",
  ]) assert.ok(joined.includes(required), required);
  assert.equal(joined.includes("docker.sock"), false);
  assert.equal(args.includes("--env"), false);
  assert.equal(args.at(-1), image);
  assert.throws(
    () => buildAdvancedPluginDockerArgs({
      image: "sigma-plugin-sandbox:latest",
      containerName: "sigma-plugin-0123456789abcdef-0123456789",
      packageRoot: verified.packageRoot,
      dataRoot: built.dataDirectory,
      verification: verified.verification,
    }),
    (error) => error instanceof AdvancedPluginError && error.code === "unpinned_plugin_image",
  );
});

test("public preflight verifies authorship without executing code or granting trust", async (t) => {
  const built = await fixture(t);
  const response = await call(preflightHandler, {
    method: "POST",
    body: {
      schemaVersion: "sigma-advanced-plugin-preflight-request/1",
      manifest: built.manifest,
      publicKeyPem: built.trustStore.publishers[0].publicKeyPem,
    },
  });
  assert.equal(response.statusCode, 200);
  assert.equal(response.body.signatureValid, true);
  assert.equal(response.body.publisherTrusted, false);
  assert.equal(response.body.packageBytesVerified, false);
  assert.equal(response.body.executableInVercel, false);
});
