#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import run_sigma_v19cs_v19x2_independent_ciao_probe as v19cs

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "sigma_v19cu_astropy_base_environment_remediation.json"


def conda_inventory(config: dict[str, Any], output: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    remediation = config["environment_remediation"]
    command = [remediation["conda_executable"], "list", "-n", remediation["environment_name"], "--json"]
    execution = v19cs.run_logged(command, output)
    if execution["returncode"] != 0:
        raise RuntimeError(f"V19CU conda inventory failed: {execution}")
    inventory = json.loads(output.read_text(encoding="utf-8"))
    return inventory, execution


def inventory_map(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {row["name"]: row for row in rows}


def execute(config: dict[str, Any]) -> dict[str, Any]:
    parents = config["parents"]
    ct_path = ROOT / parents["v19ct_failure_report"]["path"]
    cs_config_path = ROOT / parents["v19cs_config"]["path"]
    cs_runner_path = ROOT / parents["v19cs_runner"]["path"]
    import_path = ROOT / parents["astropy_import_source"]["path"]
    ct = v19cs.load_json(ct_path)
    preflight = {
        "v19ct_failure_exact": (
            v19cs.sha256(ct_path) == parents["v19ct_failure_report"]["sha256"]
            and ct["status"] == parents["v19ct_failure_report"]["required_status"]
            and parents["v19ct_failure_report"]["required_exception_fragment"] in ct["exception"]
        ),
        "v19cs_execution_contract_exact": (
            v19cs.sha256(cs_config_path) == parents["v19cs_config"]["sha256"]
            and v19cs.sha256(cs_runner_path) == parents["v19cs_runner"]["sha256"]
        ),
        "astropy_import_requirement_exact": (
            v19cs.sha256(import_path) == parents["astropy_import_source"]["sha256"]
            and parents["astropy_import_source"]["required_import"] in import_path.read_text(encoding="utf-8")
        ),
        "minimal_install_authorized": (
            config["authorization"]["install_pinned_minimal_runtime_dependency"]
            and not config["authorization"]["install_full_astropy_metapackage"]
            and not config["authorization"]["change_x2_config_runner_scientific_rules_or_data"]
            and not config["authorization"]["run_v19bs_or_derive_action"]
            and not config["authorization"]["open_lensing_halo_gravity_holdout_or_solar_optimization"]
        ),
    }
    if not all(preflight.values()):
        raise RuntimeError(f"V19CU preflight failed: {preflight}")

    outputs = config["effective_outputs"]
    before, before_exec = conda_inventory(config, ROOT / outputs["environment_before_log"])
    before_map = inventory_map(before)
    required_absent = config["environment_remediation"]["required_absent_before"]
    if any(name in before_map for name in required_absent):
        raise RuntimeError(f"V19CU required package unexpectedly present before install: {required_absent}")

    install_exec = v19cs.run_logged(config["environment_remediation"]["command"], ROOT / outputs["install_log"])
    if install_exec["returncode"] != 0:
        raise RuntimeError(f"V19CU minimal dependency install failed: {install_exec}")
    after, after_exec = conda_inventory(config, ROOT / outputs["environment_after_log"])
    after_map = inventory_map(after)
    required_after = config["environment_remediation"]["required_present_after"]
    package_gate = all(name in after_map and after_map[name]["version"] == version for name, version in required_after.items())
    if not package_gate:
        raise RuntimeError(f"V19CU installed package gate failed: required={required_after}")
    changes = {
        name: {"before": before_map.get(name), "after": after_map.get(name)}
        for name in sorted(set(before_map) | set(after_map))
        if before_map.get(name) != after_map.get(name)
    }

    effective = v19cs.load_json(cs_config_path)
    effective["outputs"] = {
        "report": outputs["report"],
        "environment_probe_log": outputs["environment_probe_log"],
        "v19x2_log": outputs["v19x2_log"],
        "v19br_log": outputs["v19br_log"],
    }
    source_result = v19cs.execute(effective)
    source_result.update({
        "protocol_version": config["protocol_version"],
        "status": "astropy_base_environment_remediated_and_source_chain_disposed",
        "generated_utc": datetime.now(UTC).isoformat(),
        "config_sha256": v19cs.sha256(DEFAULT_CONFIG),
        "environment_remediation_preflight": preflight,
        "environment_before_execution": before_exec,
        "environment_after_execution": after_exec,
        "install_execution": install_exec,
        "installed_package_gate": package_gate,
        "environment_package_changes": changes,
        "claim_boundary": config["claim_boundary"],
    })
    return source_result


def main() -> None:
    config = v19cs.load_json(DEFAULT_CONFIG)
    output = ROOT / config["effective_outputs"]["report"]
    try:
        report = execute(config)
    except Exception as exc:
        report = {
            "protocol_version": config["protocol_version"], "status": "astropy_base_environment_remediation_failed_closed",
            "exception": f"{type(exc).__name__}: {exc}", "generated_utc": datetime.now(UTC).isoformat(),
            "authorization_boundary": {"v19bs_run": False, "action_derived": False, "target_or_gravity_opened": False, "solar_optimized": False},
            "claim_boundary": config["claim_boundary"],
        }
    v19cs.atomic_json(output, report)
    print(json.dumps({key: report.get(key) for key in ("status", "decision", "exception")}, indent=2, sort_keys=True))
    if report["status"] != "astropy_base_environment_remediated_and_source_chain_disposed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
