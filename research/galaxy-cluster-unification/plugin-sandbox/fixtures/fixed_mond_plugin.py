#!/usr/bin/env python3
"""External fixture: fixed simple-MOND plus explicit sandbox observations."""

from __future__ import annotations

import json
import os
from pathlib import Path
import socket
import sys

import numpy as np


def write_is_blocked(target: Path) -> bool:
    try:
        target.write_text("mutation", encoding="utf-8")
    except OSError:
        return True
    return False


def network_is_blocked() -> bool:
    try:
        with socket.create_connection(("1.1.1.1", 53), timeout=0.25):
            return False
    except OSError:
        return True


def status_fields() -> dict[str, str]:
    fields: dict[str, str] = {}
    for line in Path("/proc/self/status").read_text(encoding="utf-8").splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            fields[key] = value.strip()
    return fields


request = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
g_bar = np.asarray(request["gBarMps2"], dtype=float)
a0 = float(request["a0Mps2"])
prediction = 0.5 * (g_bar + np.sqrt((g_bar * g_bar) + (4.0 * g_bar * a0)))

sentinel = Path("/tmp/sigma-plugin-cross-run-sentinel")
sentinel_existed_before_run = sentinel.exists()
sentinel.write_text("single use", encoding="utf-8")
status = status_fields()
secret_names = sorted(
    name for name in os.environ
    if any(marker in name.upper() for marker in ("TOKEN", "SECRET", "PASSWORD", "CREDENTIAL"))
)

result = {
    "schemaVersion": "sigma-advanced-plugin-output/1",
    "result": {
        "accelerationMps2": prediction.tolist(),
        "universalGravityParameters": 1,
        "perObjectGravityParameters": 0,
    },
    "sandboxObservations": {
        "uid": os.getuid(),
        "gid": os.getgid(),
        "effectiveCapabilitiesHex": status.get("CapEff"),
        "noNewPrivileges": status.get("NoNewPrivs"),
        "networkBlocked": network_is_blocked(),
        "datasetWriteBlocked": write_is_blocked(Path("/data/forbidden-mutation")),
        "pluginWriteBlocked": write_is_blocked(Path(__file__).with_name("forbidden-mutation")),
        "rootWriteBlocked": write_is_blocked(Path("/etc/forbidden-mutation")),
        "dockerSocketAbsent": not Path("/var/run/docker.sock").exists(),
        "hostSecretEnvironmentNames": secret_names,
        "sentinelExistedBeforeRun": sentinel_existed_before_run,
    },
}
sys.stdout.write(json.dumps(result, sort_keys=True, separators=(",", ":")))
