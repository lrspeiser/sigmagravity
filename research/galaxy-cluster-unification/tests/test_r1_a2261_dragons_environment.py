from __future__ import annotations

import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WSL_PYTHON = "/home/henry/.local/share/sigmagravity-dragons/miniforge3/envs/dragons-4.2.2/bin/python"


def test_a2261_p1_recognizes_all_raw_inputs_without_edits() -> None:
    wsl_root = "/mnt/c/Users/henry/Documents/Codex/2026-07-18/sigmagravity-frontiers-main/research/galaxy-cluster-unification"
    subprocess.run(
        ["wsl.exe", "bash", "-lc", f"{WSL_PYTHON} {wsl_root}/scripts/audit_r1_a2261_dragons_environment.py"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True
    )
    report = json.loads((ROOT / "results/r1_a2261_dragons_environment/report.json").read_text())
    assert report["bpm_checksum_passed"] is True
    assert all(report["recognition_gates"].values())
    assert report["gates"]["exact_software_versions_passed"] is True
    assert report["gates"]["P1_environment_and_bpm_gate_passed"] is True
    assert report["authorization"]["execute_frozen_P2_calibration_reduction"] is True
    assert report["authorization"]["fit_stellar_kinematics"] is False
    assert report["gates"]["gravity_response_fit_authorized"] is False
