from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_rxj2129_XMM_environment_protocol_is_frozen_before_task_execution() -> None:
    config = json.loads(
        (ROOT / "configs/r1_rxj2129_xmm_environment_protocol.json").read_text()
    )
    assert config["obsid"] == "0093030201"
    assert config["status"] == (
        "corrected_and_refrozen_after_SAS_command_interface_audit_and_before_successful_environment_verification_or_XMM_task_execution"
    )
    assert config["sas"]["version"] == "22.1.0"
    assert config["sas"]["archive_bytes"] == 1534501526
    assert config["sas"]["archive_sha256"] == (
        "f8085d71b0231b9b21fc7a4545746c948f4bc5920d7fa7cce953b2b560bed88c"
    )
    assert config["paths"]["sas_prefix"].endswith(
        "/xmmsas_22.1.0-a8f2c2afa-20250304"
    )
    assert config["sas"]["runtime_bindings"]["SAS_PERL"] == "/usr/bin/perl"
    assert config["sas"]["runtime_bindings"]["HEASoft_initialization"].endswith(
        "/bin/heainit.sh"
    )
    assert config["sas"]["verification_commands"][-1] == "epproc -v"
    assert config["sas"]["bundled_python_requirement_mapping"]["PyQt5"] == "pyqt"
    assert config["sas"]["bundled_python_requirement_mapping"]["pyds9"] == (
        "pip:pyds9==1.8.1@sha256:b4f198f5d29b749f721c491f8384f6293e43ec417bd0492be36bffb5c3904b2a"
    )
    assert config["heasoft"]["version"] == "6.36"
    assert "xspec-data" in config["heasoft"]["packages"]
    assert config["ccf"]["analysis_date"] == "2026-07-27"
    assert config["ccf"]["snapshot_policy"].startswith("Mirror once")
    assert config["authorization"]["download_and_install_exact_SAS_archive"] is True
    assert config["authorization"]["mirror_and_hash_dated_CCF_snapshot"] is True
    assert config["authorization"]["run_cifbuild_or_odfingest"] is False
    assert config["authorization"]["inspect_XMM_event_arrays"] is False
    assert config["authorization"]["fit_new_force_or_action"] is False


def test_rxj2129_XMM_environment_gate_passed_without_event_processing() -> None:
    manifest = json.loads(
        (ROOT / "data/derived/r1_rxj2129_xmm_environment_manifest.json").read_text()
    )
    report = json.loads(
        (ROOT / "results/r1_rxj2129_xmm_environment/report.json").read_text()
    )
    assert manifest["XMM_pixels_inspected"] is False
    assert manifest["sas"]["archive_bytes"] == 1534501526
    assert manifest["sas"]["all_declared_task_versions_and_python_imports_passed"] is True
    assert manifest["heasoft"]["heasoft_version"] == "6.36"
    assert manifest["heasoft"]["xspec_data_version"] == "6.36"
    assert manifest["heasoft"]["package_gate_passed"] is True
    assert manifest["ccf"]["constituent_files"] == 1815
    assert manifest["ccf"]["constituent_bytes"] == 9260611200
    assert manifest["ccf"]["all_constituent_hashes_reverified"] is True
    assert manifest["gates"]["R1B3_XMM_reduction_environment_gate_passed"] is True
    assert manifest["gates"]["R1B3_XMM_event_processing_gate_passed"] is False
    assert report["gates"]["R1B3_XMM_reduction_environment_gate_passed"] is True
    assert report["authorization"]["run_cifbuild_or_odfingest"] is False
    assert report["authorization"]["fit_new_force_or_action"] is False
