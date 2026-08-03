import { spawn } from "node:child_process";
import { createHash, randomBytes } from "node:crypto";
import { lstat, readFile, readdir, realpath } from "node:fs/promises";
import path from "node:path";

import {
  AdvancedPluginError,
  buildAdvancedPluginDockerArgs,
  verifyAdvancedPluginPackage,
} from "./advanced-plugin.mjs";

function digest(bytes) {
  return createHash("sha256").update(bytes).digest("hex");
}

async function requireRegularDataTree(root, current = "") {
  const directory = path.join(root, current);
  for (const entry of await readdir(directory, { withFileTypes: true })) {
    const relative = path.join(current, entry.name);
    const stat = await lstat(path.join(root, relative));
    if (stat.isSymbolicLink()) throw new AdvancedPluginError("plugin_data_symlink", `dataset links are forbidden: ${relative}`);
    if (stat.isDirectory()) await requireRegularDataTree(root, relative);
    else if (!stat.isFile()) throw new AdvancedPluginError("invalid_plugin_data", `unsupported dataset entry: ${relative}`);
  }
}

function boundedProcess(command, args, {
  timeoutMs,
  stdoutLimit,
  stderrLimit,
  spawnImplementation = spawn,
} = {}) {
  return new Promise((resolve, reject) => {
    const child = spawnImplementation(command, args, {
      stdio: ["ignore", "pipe", "pipe"],
      windowsHide: true,
      shell: false,
    });
    const stdout = [];
    const stderr = [];
    let stdoutBytes = 0;
    let stderrBytes = 0;
    let limitError = null;
    let timedOut = false;
    const stop = () => {
      try { child.kill("SIGKILL"); } catch { /* best effort; named-container cleanup follows */ }
    };
    const timer = setTimeout(() => {
      timedOut = true;
      stop();
    }, timeoutMs);
    child.stdout?.on("data", (chunk) => {
      stdoutBytes += chunk.length;
      if (stdoutBytes > stdoutLimit && !limitError) {
        limitError = new AdvancedPluginError("plugin_output_limit_exceeded", "plug-in stdout exceeded its declared hard limit");
        stop();
      } else if (!limitError) stdout.push(chunk);
    });
    child.stderr?.on("data", (chunk) => {
      stderrBytes += chunk.length;
      if (stderrBytes > stderrLimit && !limitError) {
        limitError = new AdvancedPluginError("plugin_log_limit_exceeded", "plug-in stderr exceeded its declared hard limit");
        stop();
      } else if (!limitError) stderr.push(chunk);
    });
    child.once("error", (error) => {
      clearTimeout(timer);
      reject(new AdvancedPluginError("plugin_container_start_failed", "sandbox container could not start", { cause: error.message }));
    });
    child.once("close", (code, signal) => {
      clearTimeout(timer);
      if (timedOut) {
        reject(new AdvancedPluginError("plugin_wall_time_exceeded", "plug-in container exceeded its wall-time limit"));
        return;
      }
      if (limitError) {
        reject(limitError);
        return;
      }
      resolve({
        code,
        signal,
        stdout: Buffer.concat(stdout),
        stderr: Buffer.concat(stderr),
      });
    });
  });
}

async function removeContainer(dockerCommand, containerName, spawnImplementation) {
  try {
    await boundedProcess(dockerCommand, ["rm", "--force", containerName], {
      timeoutMs: 10_000,
      stdoutLimit: 64 * 1024,
      stderrLimit: 64 * 1024,
      spawnImplementation,
    });
  } catch {
    // A successful --rm run has already removed the container. Cleanup is best effort.
  }
}

export async function runAdvancedPluginSandbox({
  packageDirectory,
  dataDirectory,
  trustStore,
  image,
  dockerCommand = "docker",
  allowUnpinnedImage = false,
  spawnImplementation = spawn,
} = {}) {
  const verifiedPackage = await verifyAdvancedPluginPackage({ packageDirectory, trustStore });
  const suppliedDataStat = await lstat(dataDirectory);
  if (suppliedDataStat.isSymbolicLink()) {
    throw new AdvancedPluginError("invalid_plugin_data", "data root must not be a symbolic link");
  }
  const dataRoot = await realpath(dataDirectory);
  const dataStat = await lstat(dataRoot);
  if (!dataStat.isDirectory() || dataStat.isSymbolicLink()) {
    throw new AdvancedPluginError("invalid_plugin_data", "data path must be a real directory");
  }
  await requireRegularDataTree(dataRoot);
  const requestPath = path.join(dataRoot, "request.json");
  const requestStat = await lstat(requestPath);
  if (!requestStat.isFile() || requestStat.isSymbolicLink()) {
    throw new AdvancedPluginError("invalid_plugin_data", "data directory must contain a regular request.json");
  }
  const requestBytes = await readFile(requestPath);
  let request;
  try {
    request = JSON.parse(requestBytes.toString("utf8"));
  } catch {
    throw new AdvancedPluginError("invalid_plugin_data", "request.json must contain valid JSON");
  }
  if (request?.schemaVersion !== "sigma-advanced-plugin-input/1") {
    throw new AdvancedPluginError("invalid_plugin_data", "request.json schemaVersion is unsupported");
  }
  const containerName = `sigma-plugin-${verifiedPackage.verification.packageSha256.slice(0, 16)}-${randomBytes(5).toString("hex")}`;
  const args = buildAdvancedPluginDockerArgs({
    image,
    containerName,
    packageRoot: verifiedPackage.packageRoot,
    dataRoot,
    verification: verifiedPackage.verification,
    allowUnpinnedImage,
  });
  let completed;
  try {
    completed = await boundedProcess(dockerCommand, args, {
      timeoutMs: (verifiedPackage.verification.resources.wallTimeSeconds * 1000) + 2_000,
      stdoutLimit: verifiedPackage.verification.resources.stdoutBytes + 256 * 1024,
      stderrLimit: verifiedPackage.verification.resources.stderrBytes + 64 * 1024,
      spawnImplementation,
    });
  } finally {
    await removeContainer(dockerCommand, containerName, spawnImplementation);
  }
  if (completed.code !== 0) {
    throw new AdvancedPluginError("plugin_container_failed", "sandbox container rejected or failed the plug-in", {
      exitCode: completed.code,
      signal: completed.signal,
      stderrSha256: digest(completed.stderr),
      stderrBytes: completed.stderr.length,
    });
  }
  let execution;
  try {
    execution = JSON.parse(completed.stdout.toString("utf8"));
  } catch {
    throw new AdvancedPluginError("invalid_plugin_execution_result", "sandbox returned invalid JSON");
  }
  const requestSha256 = digest(requestBytes);
  if (execution?.schemaVersion !== "sigma-advanced-plugin-execution/1"
    || execution.pluginPackageSha256 !== verifiedPackage.verification.packageSha256
    || execution.inputSha256 !== requestSha256
    || execution.output?.schemaVersion !== "sigma-advanced-plugin-output/1") {
    throw new AdvancedPluginError("invalid_plugin_execution_result", "sandbox result identity does not match the verified package and input");
  }
  return {
    execution,
    verification: verifiedPackage.verification,
    containerPolicy: {
      singleUse: true,
      dockerCommand,
      argumentSha256: digest(Buffer.from(JSON.stringify(args), "utf8")),
      hostCredentialsForwarded: false,
      outputTransport: "bounded_stdout_json",
    },
  };
}
