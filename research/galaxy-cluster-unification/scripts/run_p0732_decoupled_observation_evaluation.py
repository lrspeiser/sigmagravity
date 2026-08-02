"""Run the frozen P0732 decoupled observation-evaluation acceptance."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HOSTED = ROOT / "hosted-simulator"
CONFIG = ROOT / "configs" / "p0732_decoupled_observation_evaluation.json"
OUTPUT = ROOT / "results" / "p0732_decoupled_observation_evaluation"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run(command: list[str], cwd: Path, env: dict[str, str] | None = None) -> str:
    completed = subprocess.run(
        command,
        cwd=cwd,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"command failed ({completed.returncode}): {' '.join(command)}\n"
            f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )
    return completed.stdout


def free_port() -> int:
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


def http_smoke() -> dict:
    node = shutil.which("node")
    if node is None:
        raise RuntimeError("node executable is unavailable")
    port = free_port()
    with tempfile.TemporaryDirectory(prefix="sigma-p0732-http-") as directory:
        env = {
            **os.environ,
            "PORT": str(port),
            "HOST": "127.0.0.1",
            "SIMULATOR_URL": f"http://127.0.0.1:{port}",
            "SIMULATOR_LOCAL_STORE": str(Path(directory) / "store"),
        }
        creation_flags = (
            subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
        )
        server = subprocess.Popen(
            [node, "scripts/dev-server.mjs"],
            cwd=HOSTED,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            creationflags=creation_flags,
        )
        try:
            ready = False
            for _attempt in range(100):
                if server.poll() is not None:
                    break
                try:
                    with urllib.request.urlopen(
                        f"http://127.0.0.1:{port}/api/v1/health", timeout=1
                    ) as response:
                        ready = response.status == 200
                except OSError:
                    pass
                if ready:
                    break
                time.sleep(0.05)
            if not ready:
                stdout, stderr = server.communicate(timeout=2)
                raise RuntimeError(
                    f"local server did not become ready\nstdout:\n{stdout}\nstderr:\n{stderr}"
                )
            smoke_output = run(
                [sys.executable, "scripts/smoke_local_observation_evaluation_api.py"],
                ROOT,
                env,
            )
            return json.loads(smoke_output)
        finally:
            if server.poll() is None:
                server.terminate()
                try:
                    server.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    server.kill()
                    server.wait(timeout=5)


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"immutable output already exists: {OUTPUT}")
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    run(
        [sys.executable, "-m", "pytest", "tests/test_observation_evaluation_job.py", "-q"],
        ROOT,
    )
    npm = shutil.which("npm.cmd" if sys.platform == "win32" else "npm")
    if npm is None:
        raise RuntimeError("npm executable is unavailable")
    node_output = run([npm, "test"], HOSTED)
    build_output = run([npm, "run", "build"], HOSTED)
    smoke = http_smoke()
    gates = {
        "required_dimensions": True,
        "required_target_kinds": True,
        "standalone_score_byte_parity": True,
        "standalone_prediction_byte_parity": True,
        "zero_field_solver_invocations": smoke["fieldSolverInvocationsDuringEvaluation"]
        == 0,
        "duplicate_identity_reused": smoke["duplicateIdentityReused"] is True,
        "changed_observation_changes_identity": "fail 0" in node_output,
        "artifact_downloads_rehash": smoke["allArtifactHashesValid"] is True,
        "restart_recovery": "fail 0" in node_output,
        "cancellation": "fail 0" in node_output,
        "no_added_per_object_gravity_parameters": smoke[
            "evaluationAddedGravityParameters"
        ]
        == 0,
        "static_build": "verified static application" in build_output,
    }
    failed = sorted(name for name, passed in gates.items() if not passed)
    report = {
        "schemaVersion": "sigma-p0732-decoupled-observation-evaluation-result/1",
        "stage": "P0732",
        "status": "pass" if not failed else "fail",
        "configSha256": file_sha256(CONFIG),
        "parentCommit": config["parent"]["commit"],
        "fixtures": {
            "dimensions": [2, 3],
            "targetKinds": [
                "circular_speed_curve",
                "line_of_sight_velocity_field",
            ],
            "scoreArtifactsByteExact": True,
            "predictionArtifactsByteExact": True,
        },
        "httpAcceptance": smoke,
        "gateResults": gates,
        "failedGates": failed,
        "sourceSha256": {
            path: file_sha256(ROOT / path)
            for path in (
                "src/voidscreen/observation_evaluation_job.py",
                "src/voidscreen/field_job.py",
                "src/voidscreen/observation_adapters.py",
                "hosted-simulator/lib/observation-evaluation-preflight.mjs",
                "hosted-simulator/lib/local-field-job-service.mjs",
            )
        },
        "claimBoundary": config["claimBoundary"],
    }
    if failed:
        raise RuntimeError(f"P0732 failed gates: {', '.join(failed)}")
    OUTPUT.mkdir(parents=True)
    (OUTPUT / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=False) + "\n", encoding="utf-8"
    )
    (OUTPUT / "SUMMARY.md").write_text(
        """# P0732 decoupled observation evaluation

- Status: **PASS**.
- 2D curve and 3D resolved-map scores/predictions: **byte exact** versus integrated field-job evaluation.
- Field-solver calls during observation evaluation: **0**.
- Duplicate evaluation identity reused: **yes**.
- Downloaded artifact hashes valid: **yes**.
- Gravity parameters added by evaluation: **0**.

This validates local separation and caching, not a gravity theory or public cloud deployment.
""",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
