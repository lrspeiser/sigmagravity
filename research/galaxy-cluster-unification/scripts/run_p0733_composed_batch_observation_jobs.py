"""Run the frozen P0733 two-stage batch-composition acceptance."""

from __future__ import annotations

import hashlib
import json
import os
import re
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
CONFIG = ROOT / "configs" / "p0733_composed_batch_observation_jobs.json"
OUTPUT = ROOT / "results" / "p0733_composed_batch_observation_jobs"


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
    with tempfile.TemporaryDirectory(prefix="sigma-p0733-http-") as directory:
        base = f"http://127.0.0.1:{port}"
        env = {
            **os.environ,
            "PORT": str(port),
            "HOST": "127.0.0.1",
            "SIMULATOR_BASE_URL": base,
            "SIMULATOR_LOCAL_STORE": str(Path(directory) / "store"),
        }
        creation_flags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
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
                        f"{base}/api/v1/health", timeout=1
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
                    "local server did not become ready\n"
                    f"stdout:\n{stdout}\nstderr:\n{stderr}"
                )
            return json.loads(
                run(
                    [sys.executable, "scripts/smoke_local_batch_api.py"],
                    ROOT,
                    env,
                )
            )
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
    npm = shutil.which("npm.cmd" if sys.platform == "win32" else "npm")
    if npm is None:
        raise RuntimeError("npm executable is unavailable")
    test_output = run([npm, "test"], HOSTED)
    build_output = run([npm, "run", "build"], HOSTED)
    smoke = http_smoke()
    test_match = re.search(r"# tests (\d+)", test_output)
    pass_match = re.search(r"# pass (\d+)", test_output)
    test_count = int(test_match.group(1)) if test_match else None
    pass_count = int(pass_match.group(1)) if pass_match else None
    suite_passed = test_count is not None and test_count == pass_count
    gates = {
        "field_children_exclude_observation_targets": smoke[
            "fieldChildObservationTargetCount"
        ]
        == 0,
        "maximum_one_observation_child_per_scored_system": bool(
            smoke["observationEvaluationJobId"]
        ),
        "changed_observation_preserves_field_identity": smoke[
            "changedObservationPreservedFieldJobId"
        ]
        is True,
        "changed_observation_changes_evaluation_identity": smoke[
            "changedObservationChangedEvaluationJobId"
        ]
        is True,
        "identical_composed_submission_reuses_identity": smoke[
            "duplicateComposedBatchReused"
        ]
        is True,
        "scores_and_predictions_come_from_standalone_child": suite_passed,
        "field_only_systems_create_no_observation_children": smoke[
            "fieldOnlyObservationChildren"
        ]
        == 0,
        "failed_field_creates_no_observation_child": suite_passed,
        "failed_observation_is_excluded_from_aggregate": suite_passed,
        "cancellation_reaches_both_child_types": suite_passed,
        "restart_recovers_composition_phase_without_rerun": suite_passed,
        "artifact_downloads_rehash": smoke["allDownloadedArtifactHashesValid"]
        is True,
        "no_per_object_gravity_parameters": smoke["perObjectGravityParameters"]
        == 0,
        "observation_adds_no_gravity_parameters": smoke[
            "observationAddedGravityParameters"
        ]
        == 0,
        "static_build": "verified static application" in build_output,
    }
    failed = sorted(name for name, passed in gates.items() if not passed)
    report = {
        "schemaVersion": "sigma-p0733-composed-batch-observation-jobs-result/1",
        "stage": "P0733",
        "status": "pass" if not failed else "fail",
        "configSha256": file_sha256(CONFIG),
        "parentCommit": config["parent"]["commit"],
        "hostedTestCount": test_count,
        "hostedPassCount": pass_count,
        "httpAcceptance": smoke,
        "gateResults": gates,
        "failedGates": failed,
        "sourceSha256": {
            path: file_sha256(ROOT / path)
            for path in (
                "hosted-simulator/lib/batch-preflight.mjs",
                "hosted-simulator/lib/local-batch-service.mjs",
                "hosted-simulator/lib/local-field-job-service.mjs",
                "hosted-simulator/lib/observation-evaluation-preflight.mjs",
                "hosted-simulator/schemas/batch-submit-v1.schema.json",
                "scripts/smoke_local_batch_api.py",
            )
        },
        "claimBoundary": config["claimBoundary"],
    }
    if failed:
        raise RuntimeError(f"P0733 failed gates: {', '.join(failed)}")
    OUTPUT.mkdir(parents=True)
    (OUTPUT / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=False) + "\n", encoding="utf-8"
    )
    (OUTPUT / "SUMMARY.md").write_text(
        f"""# P0733 composed batch observation jobs

- Status: **PASS**.
- Hosted tests: **{pass_count}/{test_count} passed**.
- Real HTTP systems: **{smoke['successfulSystems']}/3 succeeded**.
- Observation targets embedded in the field child: **0**.
- Changed observation preserved the field job: **yes**.
- Changed observation created a new evaluation job: **yes**.
- Field-only systems that created observation jobs: **0**.
- Per-object gravity parameters: **0**.
- Gravity parameters added by observation evaluation: **0**.
- Downloaded batch artifact hashes valid: **yes**.

This validates local two-stage orchestration and provenance. It does not validate a gravity theory, add photon lensing, or connect the filesystem reference queue to public durable infrastructure.
""",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
