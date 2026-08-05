#!/usr/bin/env python3
from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from run_sigma_v19cr_v19x2_ciao_launch_remediation import (
    atomic_json,
    load_json,
    probe_environment,
    run_logged,
    scratch_audit,
    sha256,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "sigma_v19cs_v19x2_independent_ciao_probe.json"


def validate_preflight(config: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    parents = config["failure_parents"]
    cr_path = ROOT / parents["v19cr_report"]["path"]
    fail_path = ROOT / parents["v19x2_failure_report"]["path"]
    x2_config_path = ROOT / parents["v19x2_config"]["path"]
    cd_config_path = ROOT / parents["v19cd_environment_config"]["path"]
    w5_path = ROOT / parents["v19w5_report"]["path"]
    cr, fail, x2, cd_config, w5 = map(load_json, (cr_path, fail_path, x2_config_path, cd_config_path, w5_path))
    scratch = scratch_audit(config)
    execution = config["unchanged_execution"]
    environment = config["environment"]
    checks = {
        "v19cr_preflight_failure_exact": (
            sha256(cr_path) == parents["v19cr_report"]["sha256"]
            and cr["status"] == parents["v19cr_report"]["required_status"]
            and parents["v19cr_report"]["required_exception_fragment"] in cr["exception"]
        ),
        "v19x2_environment_failure_exact": (
            sha256(fail_path) == parents["v19x2_failure_report"]["sha256"]
            and fail["status"] == parents["v19x2_failure_report"]["required_status"]
            and fail["execution_exception"] == parents["v19x2_failure_report"]["required_exception"]
            and fail["gates"] == {"execution_completed": False}
        ),
        "corrected_x2_config_exact": (
            sha256(x2_config_path) == parents["v19x2_config"]["sha256"]
            and x2["runtime_authorization"]["required_completed_cells"] == parents["v19x2_config"]["required_completed_cells"]
            and x2["execution"]["response_archives"]["v19w5_recovery"] == parents["v19x2_config"]["required_recovery_root"]
        ),
        "launch_contract_matches_frozen_v19cd_config": (
            sha256(cd_config_path) == parents["v19cd_environment_config"]["sha256"]
            and cd_config["environment"]["conda_executable"] == environment["conda_executable"]
            and cd_config["environment"]["environment_name"] == environment["environment_name"]
            and cd_config["environment"]["environment_prefix"] == environment["environment_prefix"]
        ),
        "v19w5_pass_exact": (
            sha256(w5_path) == parents["v19w5_report"]["sha256"]
            and w5["status"] == parents["v19w5_report"]["required_status"]
            and all(w5["gates"].values())
        ),
        "unchanged_runners_exact": (
            sha256(ROOT / execution["v19x2_runner"]) == execution["v19x2_runner_sha256"]
            and sha256(ROOT / execution["v19br_config"]) == execution["v19br_config_sha256"]
            and sha256(ROOT / execution["v19br_runner"]) == execution["v19br_runner_sha256"]
            and sha256(ROOT / config["implementation"]["reused_environment_helpers"]) == config["implementation"]["reused_environment_helpers_sha256"]
        ),
        "scratch_precedes_combination_and_fit": scratch["only_permitted_files"] and not scratch["combined_or_fit_products"],
        "authorization_probe_only": (
            config["authorization"]["remove_invalid_historical_terminal_pass_requirement"]
            and config["authorization"]["require_independent_live_environment_probe"]
            and not config["authorization"]["change_x2_config_runner_or_scientific_rules"]
            and not config["authorization"]["run_v19bs_or_derive_action"]
            and not config["authorization"]["open_lensing_halo_gravity_holdout_or_solar_optimization"]
        ),
    }
    if not all(checks.values()):
        raise RuntimeError(f"V19CS preflight failed: {checks}; scratch={scratch}")
    return checks, scratch


def execute(config: dict[str, Any]) -> dict[str, Any]:
    preflight, scratch = validate_preflight(config)
    environment_probe, probe_execution = probe_environment(config)
    parents = config["failure_parents"]
    execution = config["unchanged_execution"]
    x2_config_path = ROOT / parents["v19x2_config"]["path"]
    x2_sha = sha256(x2_config_path)
    x2_command = config["environment"]["launch_prefix"] + [
        str(ROOT / execution["v19x2_runner"]),
        "--config", str(x2_config_path),
        "--output", str(ROOT / execution["v19x2_output"]),
        "--scratch", execution["v19x2_scratch"],
        "--response-report", str(ROOT / parents["v19w5_report"]["path"]),
    ]
    x2_exec = run_logged(x2_command, ROOT / config["outputs"]["v19x2_log"])
    x2_report = load_json(ROOT / execution["v19x2_output"] / "report.json")
    passing = "unified_spectral_combination_commissioning_passed_and_full_regional_fits_authorized"
    failing = "unified_spectral_combination_commissioning_gate_failed"
    scientific = x2_report["status"] in {passing, failing}

    br_exec = None
    br_report = None
    if x2_report["status"] == passing:
        br_command = config["environment"]["launch_prefix"] + [
            str(ROOT / execution["v19br_runner"]), "--config", str(ROOT / execution["v19br_config"]), "--execute"
        ]
        br_exec = run_logged(br_command, ROOT / config["outputs"]["v19br_log"])
        br_report = load_json(ROOT / "results" / "sigma_v19br_target_sealed_terminal_chain" / "report.json")

    gates = {
        "preflight_exact": all(preflight.values()),
        "x2_config_byte_identical": sha256(x2_config_path) == x2_sha == parents["v19x2_config"]["sha256"],
        "live_ciao_probe_passed": all(environment_probe["checks"].values()),
        "v19x2_reaches_registered_scientific_disposition": scientific and x2_report["runner_sha256"] == execution["v19x2_runner_sha256"],
        "if_v19x2_passes_v19br_reaches_source_decision": (
            x2_report["status"] != passing
            or (br_exec is not None and br_exec["returncode"] == 0 and br_report is not None and br_report["status"] == "terminal_chain_complete" and br_report.get("source_decision") is not None)
        ),
        "target_and_gravity_sealed": (
            not x2_report["replacement_cluster_lensing_target_opened"]
            and not x2_report["gravity_formula_or_parameter_changed"]
            and (br_report is None or not br_report["lensing_halo_action_gravity_or_holdout_payload_opened"])
        ),
    }
    decision = (
        "v19x2_valid_scientific_gate_failure_no_full_source_chain" if x2_report["status"] == failing
        else "run_frozen_v19bs_disposition_next" if br_report is not None and br_report.get("status") == "terminal_chain_complete"
        else "v19x2_or_source_chain_execution_incomplete"
    )
    if not all(gates.values()):
        raise RuntimeError(f"V19CS post-execution gates failed: {gates}; decision={decision}; x2_status={x2_report['status']}")
    return {
        "protocol_version": config["protocol_version"], "status": "v19x2_independent_ciao_probe_completed", "decision": decision,
        "generated_utc": datetime.now(UTC).isoformat(), "config_sha256": sha256(DEFAULT_CONFIG),
        "preflight": preflight, "preexecution_scratch_audit": scratch,
        "environment_probe": environment_probe, "environment_probe_execution": probe_execution,
        "v19x2_config_sha256_before_and_after": x2_sha, "v19x2_execution": x2_exec, "v19x2_report": x2_report,
        "v19br_execution": br_exec,
        "v19br_summary": None if br_report is None else {
            "status": br_report["status"], "source_decision": br_report.get("source_decision"),
            "executed_stages": br_report.get("executed_stages"), "already_passing_stages": br_report.get("already_passing_stages"),
        },
        "gate_results": gates,
        "authorization_boundary": {"v19bs_run": False, "action_derived": False, "target_or_gravity_opened": False, "solar_optimized": False},
        "claim_boundary": config["claim_boundary"],
    }


def main() -> None:
    config = load_json(DEFAULT_CONFIG)
    output = ROOT / config["outputs"]["report"]
    try:
        report = execute(config)
    except Exception as exc:
        report = {
            "protocol_version": config["protocol_version"], "status": "v19x2_independent_ciao_probe_failed_closed",
            "exception": f"{type(exc).__name__}: {exc}", "generated_utc": datetime.now(UTC).isoformat(),
            "authorization_boundary": {"v19bs_run": False, "action_derived": False, "target_or_gravity_opened": False, "solar_optimized": False},
            "claim_boundary": config["claim_boundary"],
        }
    atomic_json(output, report)
    print(json.dumps({key: report.get(key) for key in ("status", "decision", "exception")}, indent=2, sort_keys=True))
    if report["status"] != "v19x2_independent_ciao_probe_completed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
