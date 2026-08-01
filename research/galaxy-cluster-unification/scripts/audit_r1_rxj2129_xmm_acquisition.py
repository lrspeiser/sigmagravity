#!/usr/bin/env python3
"""Audit the complete RX J2129 XMM archive acquisition without pixel access."""

from __future__ import annotations

import json
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw/r1_rxj2129_xmm"
PROVENANCE = RAW / "provenance.json"
REPORT = ROOT / "results/r1_rxj2129_xmm_acquisition/report.json"


def command_output(command: str) -> str:
    result = subprocess.run(
        ["wsl.exe", "-e", "bash", "-lc", command],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip()


def build_report() -> dict:
    provenance = json.loads(PROVENANCE.read_text(encoding="utf-8"))
    counts = Counter()
    files_pass = True
    partials = []
    for record in provenance["records"]:
        path = ROOT / record["path"]
        files_pass &= bool(path.exists() and path.stat().st_size == record["bytes"])
        parts = Path(record["path"]).parts
        obs_index = parts.index(provenance["obsid"])
        counts[parts[obs_index + 1]] += 1
    for path in RAW.rglob("*.part"):
        partials.append(str(path.relative_to(ROOT)).replace("\\", "/"))
    expected_counts = {"4XMM": 276, "ODF": 284, "om_mosaic": 9, "PPS": 2255}
    etag_semantics_pass = all(
        "-" in record["archive_s3_etag"] or record["archive_s3_etag_verified_as_md5"]
        for record in provenance["records"]
    )
    multipart_etags = sum("-" in record["archive_s3_etag"] for record in provenance["records"])
    manifest_gate = bool(
        provenance["selection_frozen_before_download"]
        and provenance["XMM_pixels_inspected"] is False
        and provenance["local_files"] == 2824
        and provenance["local_bytes"] == 645765075
        and dict(counts) == expected_counts
        and not partials
        and files_pass
        and etag_semantics_pass
    )
    environment = {
        "wsl_distribution": command_output("lsb_release -ds 2>/dev/null || true"),
        "sasversion": command_output("command -v sasversion >/dev/null && sasversion 2>/dev/null || true"),
        "heasoft_ftversion": command_output("command -v ftversion >/dev/null && ftversion 2>/dev/null || true"),
        "docker": command_output("command -v docker || true"),
        "podman": command_output("command -v podman || true"),
    }
    sas_ready = bool(environment["sasversion"] and environment["heasoft_ftversion"])
    report = {
        "report_version": "R1B3-RXJ2129-XMM-acquisition-0.1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "selection_blind": True,
        "obsid": provenance["obsid"],
        "XMM_pixels_inspected": False,
        "local_files": provenance["local_files"],
        "local_bytes": provenance["local_bytes"],
        "category_counts": dict(counts),
        "partial_files": partials,
        "multipart_ETags_with_local_SHA256_provenance": multipart_etags,
        "environment": environment,
        "gates": {
            "archive_manifest_ETag_size_and_SHA_provenance_passed": manifest_gate,
            "raw_ODF_and_PPS_classes_present": counts["ODF"] > 0 and counts["PPS"] > 0,
            "exact_SAS_HEASoft_environment_present": sas_ready,
            "R1B3_XMM_acquisition_gate_passed": manifest_gate,
            "R1B3_XMM_reduction_environment_gate_passed": sas_ready,
        },
        "decision": "authorize_SAS_HEASoft_CCF_environment_protocol_freeze" if manifest_gate and not sas_ready else "authorize_XMM_execution_protocol_freeze" if manifest_gate else "stop_XMM_acquisition_integrity_failure",
        "next_action": "Install the official SAS 22.1.0 Ubuntu 24.04 binary in an isolated path with a compatible HEASoft environment and a dated CCF snapshot; record checksums and versions before cifbuild or odfingest." if manifest_gate and not sas_ready else "Freeze the exact XMM event-processing commands before reading arrays.",
        "authorization": {
            "freeze_and_install_SAS_HEASoft_CCF_environment": manifest_gate and not sas_ready,
            "run_cifbuild_or_odfingest": False,
            "inspect_XMM_event_arrays": False,
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
