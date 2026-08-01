#!/usr/bin/env python3
"""Audit the frozen RX J2129 SAS/HEASoft/CCF environment without reading XMM arrays."""

from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/r1_rxj2129_xmm_environment_protocol.json"
ACQUISITION = ROOT / "results/r1_rxj2129_xmm_acquisition/report.json"
REPORT = ROOT / "results/r1_rxj2129_xmm_environment/report.json"


def wsl(command: str) -> dict:
    result = subprocess.run(
        ["wsl.exe", "-d", "Ubuntu-24.04", "--", "bash", "-lc", command],
        capture_output=True,
        text=True,
        check=False,
    )
    return {
        "exit_code": result.returncode,
        "stdout": result.stdout.strip(),
        "stderr": result.stderr.strip(),
    }


def build_report() -> dict:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    acquisition = json.loads(ACQUISITION.read_text(encoding="utf-8"))
    paths = config["paths"]
    manifest_path = ROOT / paths["repo_environment_manifest"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else None

    protocol_gate = bool(
        config["status"]
        == "corrected_and_refrozen_after_SAS_command_interface_audit_and_before_successful_environment_verification_or_XMM_task_execution"
        and config["selection_blind"]
        and config["obsid"] == "0093030201"
        and config["sas"]["version"] == "22.1.0"
        and config["sas"]["archive_bytes"] == 1534501526
        and config["sas"]["archive_sha256"]
        == "f8085d71b0231b9b21fc7a4545746c948f4bc5920d7fa7cce953b2b560bed88c"
        and config["sas"]["bundled_python_requirement_mapping"]["pyds9"].endswith(
            "b4f198f5d29b749f721c491f8384f6293e43ec417bd0492be36bffb5c3904b2a"
        )
        and config["heasoft"]["version"] == "6.36"
        and "xspec-data" in config["heasoft"]["packages"]
        and config["ccf"]["analysis_date"] == config["ccf"]["snapshot_date_utc"]
        and acquisition["gates"]["R1B3_XMM_acquisition_gate_passed"]
        and not any(
            config["authorization"][key]
            for key in (
                "run_cifbuild_or_odfingest",
                "inspect_XMM_event_arrays",
                "run_EPIC_reduction",
                "infer_gas_profile",
                "measure_HST_arc_pixels",
                "infer_dynamical_or_Weyl_response",
                "fit_new_force_or_action",
            )
        )
    )

    runtime = {
        "host": wsl("cat /etc/os-release && uname -m"),
        "sasversion": wsl(
            f"export CONDA_PREFIX={paths['heasoft_prefix']} && "
            f"source {paths['heasoft_prefix']}/bin/heainit.sh >/dev/null 2>&1 && "
            f"test -f {paths['sas_prefix']}/.configuration_complete && "
            f"source {paths['sas_prefix']}/setsas.sh >/dev/null 2>&1 && sasversion"
        ),
        "ftversion": wsl(
            f"export CONDA_PREFIX={paths['heasoft_prefix']} && "
            f"source {paths['heasoft_prefix']}/bin/heainit.sh >/dev/null 2>&1 && ftversion"
        ),
        "xspec_data": wsl(
            f"test -d {paths['heasoft_prefix']}/heasoft/spectral/modelData && "
            f"find {paths['heasoft_prefix']}/heasoft/spectral/modelData -type f -print -quit | grep -q ."
        ),
        "ccf": wsl(
            f"test -f {paths['ccf_snapshot']}/CCF_MANIFEST.sha256 && "
            f"find {paths['ccf_snapshot']} -type f ! -name CCF_MANIFEST.sha256 | wc -l && "
            f"du -sb {paths['ccf_snapshot']} | cut -f1"
        ),
    }

    manifest_gate = bool(
        manifest
        and manifest.get("protocol_version") == config["protocol_version"]
        and manifest.get("obsid") == config["obsid"]
        and manifest.get("XMM_pixels_inspected") is False
        and manifest.get("sas", {}).get("archive_sha256")
        and manifest.get("heasoft", {}).get("explicit_lock_path")
        and manifest.get("ccf", {}).get("manifest_sha256")
    )
    command_gate = all(item["exit_code"] == 0 for item in runtime.values())
    version_gate = bool(
        command_gate
        and "22.1.0" in runtime["sasversion"]["stdout"]
        and "6.36" in runtime["ftversion"]["stdout"]
    )
    ccf_tokens = runtime["ccf"]["stdout"].splitlines()
    ccf_count = int(ccf_tokens[0]) if len(ccf_tokens) == 2 and ccf_tokens[0].isdigit() else 0
    ccf_bytes = int(ccf_tokens[1]) if len(ccf_tokens) == 2 and ccf_tokens[1].isdigit() else 0
    ccf_gate = bool(
        runtime["ccf"]["exit_code"] == 0
        and ccf_count >= config["ccf"]["minimum_constituent_files"]
        and ccf_bytes >= config["ccf"]["minimum_total_bytes"]
    )
    environment_gate = protocol_gate and manifest_gate and version_gate and ccf_gate

    report = {
        "report_version": "R1B3-RXJ2129-XMM-environment-audit-0.1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "selection_blind": True,
        "obsid": config["obsid"],
        "XMM_pixels_inspected": False,
        "runtime": runtime,
        "ccf_snapshot": {
            "date": config["ccf"]["snapshot_date_utc"],
            "analysis_date": config["ccf"]["analysis_date"],
            "files": ccf_count,
            "bytes": ccf_bytes,
        },
        "gates": {
            "environment_protocol_frozen_before_install_and_task_execution": protocol_gate,
            "software_and_CCF_manifest_complete": manifest_gate,
            "exact_SAS_HEASoft_XSPEC_commands_passed": version_gate,
            "dated_CCF_snapshot_threshold_and_hash_manifest_passed": ccf_gate,
            "R1B3_XMM_reduction_environment_gate_passed": environment_gate,
            "R1B3_XMM_event_processing_gate_passed": False,
        },
        "decision": (
            "authorize_separate_XMM_event_processing_protocol_freeze"
            if environment_gate
            else "stop_before_XMM_tasks_and_complete_environment"
        ),
        "authorization": {
            "freeze_XMM_event_processing_commands": environment_gate,
            "run_cifbuild_or_odfingest": False,
            "inspect_XMM_event_arrays": False,
            "run_EPIC_reduction": False,
            "infer_gas_profile": False,
            "measure_HST_arc_pixels": False,
            "infer_dynamical_or_Weyl_response": False,
            "fit_new_force_or_action": False,
        },
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    print(json.dumps(build_report(), indent=2))
