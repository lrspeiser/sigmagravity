#!/usr/bin/env python3
"""Trusted ABI wrapper for one untrusted, already signature-verified plug-in."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import selectors
import signal
import subprocess
import sys
import time
from typing import Any


PLUGIN_ROOT = Path("/plugin")
DATA_ROOT = Path("/data")
REQUEST_PATH = DATA_ROOT / "request.json"
RUNTIME = {
    "id": "sigma-python-plugin/1",
    "pythonVersion": "3.13.7",
    "packages": {"numpy": "2.2.6", "scipy": "1.16.1"},
}


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def fail(code: str, message: str) -> None:
    sys.stderr.write(json.dumps({"error": code, "message": message}, separators=(",", ":")) + "\n")
    raise SystemExit(70)


def manifest_core(manifest: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in manifest.items() if key not in {"packageSha256", "signature"}}


def safe_relative(value: str) -> Path:
    if not isinstance(value, str) or not value or "\\" in value or value.startswith("/"):
        fail("invalid_plugin_path", "manifest contains an invalid package path")
    parts = value.split("/")
    if any(part in {"", ".", ".."} for part in parts):
        fail("invalid_plugin_path", "manifest contains traversal or ambiguous path segments")
    return Path(*parts)


def verify_package() -> tuple[dict[str, Any], str]:
    try:
        manifest = json.loads((PLUGIN_ROOT / "plugin.json").read_text(encoding="utf-8"))
    except Exception:
        fail("invalid_plugin_package", "plugin.json is missing or invalid")
    core = manifest_core(manifest)
    package_sha256 = sha256_bytes(canonical_bytes(core))
    if manifest.get("packageSha256") != package_sha256:
        fail("plugin_package_identity_mismatch", "manifest identity changed after host verification")
    if core.get("runtime") != RUNTIME:
        fail("unsupported_plugin_runtime", "manifest runtime does not match the container runtime")
    declared: set[str] = set()
    for record in core.get("files", []):
        relative = safe_relative(record.get("path"))
        declared.add(relative.as_posix())
        target = PLUGIN_ROOT / relative
        try:
            stat = target.lstat()
            content = target.read_bytes()
        except Exception:
            fail("plugin_file_identity_mismatch", f"declared file is missing: {relative.as_posix()}")
        if target.is_symlink() or not target.is_file() or stat.st_size != record.get("bytes"):
            fail("plugin_file_identity_mismatch", f"declared file metadata changed: {relative.as_posix()}")
        if sha256_bytes(content) != record.get("sha256"):
            fail("plugin_file_identity_mismatch", f"declared file hash changed: {relative.as_posix()}")
    actual: set[str] = set()
    for target in PLUGIN_ROOT.rglob("*"):
        if target.is_symlink():
            fail("plugin_package_symlink", "symbolic links are forbidden")
        if target.is_file() and target.name != "plugin.json":
            actual.add(target.relative_to(PLUGIN_ROOT).as_posix())
    if actual != declared:
        fail("plugin_package_file_set_mismatch", "package contains missing or undeclared files")
    return manifest, package_sha256


def kill_group(process: subprocess.Popen[bytes]) -> None:
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def execute_plugin(manifest: dict[str, Any]) -> tuple[dict[str, Any], bytes, int]:
    resources = manifest["resources"]
    entrypoint = PLUGIN_ROOT / safe_relative(manifest["entrypoint"])
    environment = {
        "PATH": "/usr/local/bin:/usr/bin:/bin",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONHASHSEED": "0",
        "PYTHONNOUSERSITE": "1",
        "PYTHONUNBUFFERED": "1",
    }
    launcher = (
        "import runpy,sys;"
        "sys.path.insert(0,'/plugin');"
        f"sys.argv=[{str(entrypoint)!r},{str(REQUEST_PATH)!r}];"
        f"runpy.run_path({str(entrypoint)!r},run_name='__main__')"
    )
    process = subprocess.Popen(
        [sys.executable, "-I", "-B", "-c", launcher],
        cwd="/tmp",
        env=environment,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        close_fds=True,
        start_new_session=True,
    )
    assert process.stdout is not None and process.stderr is not None
    selector = selectors.DefaultSelector()
    selector.register(process.stdout, selectors.EVENT_READ, "stdout")
    selector.register(process.stderr, selectors.EVENT_READ, "stderr")
    buffers: dict[str, bytearray] = {"stdout": bytearray(), "stderr": bytearray()}
    limits = {"stdout": resources["stdoutBytes"], "stderr": resources["stderrBytes"]}
    deadline = time.monotonic() + resources["wallTimeSeconds"]
    while selector.get_map():
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            kill_group(process)
            process.wait()
            fail("plugin_wall_time_exceeded", "plug-in exceeded its declared wall-time limit")
        for key, _ in selector.select(timeout=min(remaining, 0.1)):
            chunk = os.read(key.fileobj.fileno(), 65536)
            if not chunk:
                selector.unregister(key.fileobj)
                continue
            buffer = buffers[key.data]
            buffer.extend(chunk)
            if len(buffer) > limits[key.data]:
                kill_group(process)
                process.wait()
                fail(f"plugin_{key.data}_limit_exceeded", f"plug-in {key.data} exceeded its hard limit")
    exit_code = process.wait()
    if exit_code != 0:
        fail("plugin_process_failed", f"plug-in exited with status {exit_code}")
    try:
        output = json.loads(bytes(buffers["stdout"]).decode("utf-8"))
    except Exception:
        fail("invalid_plugin_output", "plug-in stdout must contain one JSON document")
    if not isinstance(output, dict) or output.get("schemaVersion") != "sigma-advanced-plugin-output/1":
        fail("invalid_plugin_output", "plug-in output schemaVersion is unsupported")
    return output, bytes(buffers["stderr"]), exit_code


def main() -> None:
    if os.getuid() != 65532 or os.getgid() != 65532:
        fail("sandbox_identity_mismatch", "sandbox must execute as the fixed non-root identity")
    manifest, package_sha256 = verify_package()
    try:
        request_bytes = REQUEST_PATH.read_bytes()
        request = json.loads(request_bytes)
    except Exception:
        fail("invalid_plugin_input", "request.json is missing or invalid")
    if request.get("schemaVersion") != "sigma-advanced-plugin-input/1":
        fail("invalid_plugin_input", "request schemaVersion is unsupported")
    output, stderr, exit_code = execute_plugin(manifest)
    envelope = {
        "schemaVersion": "sigma-advanced-plugin-execution/1",
        "pluginPackageSha256": package_sha256,
        "inputSha256": sha256_bytes(request_bytes),
        "runtime": RUNTIME,
        "output": output,
        "diagnostics": {
            "pluginExitCode": exit_code,
            "pluginStderrBytes": len(stderr),
            "pluginStderrSha256": sha256_bytes(stderr),
        },
    }
    sys.stdout.buffer.write(canonical_bytes(envelope) + b"\n")


if __name__ == "__main__":
    main()
