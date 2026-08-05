#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "sigma_v19cr_v19x2_ciao_launch_remediation.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def run_logged(command: list[str], path: Path) -> dict[str, Any]:
    completed = subprocess.run(command, cwd=ROOT, check=False, capture_output=True, text=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(completed.stdout + ("\n" if completed.stdout and completed.stderr else "") + completed.stderr, encoding="utf-8")
    return {
        "command": command,
        "returncode": completed.returncode,
        "log": path.relative_to(ROOT).as_posix(),
        "log_sha256": sha256(path),
        "log_bytes": path.stat().st_size,
    }


def probe_environment(config: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    environment = config["environment"]
    code = (
        "import importlib,json,shutil,sys;"
        f"commands={environment['required_executables']!r};"
        f"modules={environment['required_python_modules']!r};"
        "print(json.dumps({'python':sys.executable,'commands':{x:shutil.which(x) for x in commands},"
        "'modules':{x:bool(importlib.import_module(x)) for x in modules}},sort_keys=True))"
    )
    command = environment["launch_prefix"] + ["-c", code]
    execution = run_logged(command, ROOT / config["outputs"]["environment_probe_log"])
    lines = (ROOT / config["outputs"]["environment_probe_log"]).read_text(encoding="utf-8").splitlines()
    if execution["returncode"] != 0 or not lines:
        raise RuntimeError(f"V19CR environment probe failed: {execution}")
    payload = json.loads(lines[-1])
    expected_bin = PurePosixPath(environment["environment_prefix"]) / "bin"
    checks = {
        "python_from_declared_environment": PurePosixPath(payload["python"]) == expected_bin / "python",
        "all_required_executables_resolve": all(payload["commands"].values()),
        "all_executables_from_declared_environment": all(
            PurePosixPath(value).parent == expected_bin for value in payload["commands"].values() if value
        ),
        "all_required_python_modules_import": all(payload["modules"].values()),
    }
    if not all(checks.values()):
        raise RuntimeError(f"V19CR environment probe gates failed: {checks}; payload={payload}")
    return {"checks": checks, **payload}, execution


def scratch_audit(config: dict[str, Any]) -> dict[str, Any]:
    execution = config["unchanged_execution"]
    scratch = Path(execution["v19x2_scratch"])
    files = sorted(path.relative_to(scratch).as_posix() for path in scratch.rglob("*") if path.is_file()) if scratch.is_dir() else []
    permitted = sorted(execution["permitted_preexisting_scratch_files"])
    combined_or_fit = [
        name for name in files
        if name.endswith((".pi", ".arf", ".rmf", ".json")) or "fit" in PurePosixPath(name).name.lower()
    ]
    return {
        "scratch": str(scratch),
        "files": files,
        "permitted_files": permitted,
        "only_permitted_files": files == permitted,
        "combined_or_fit_products": combined_or_fit,
    }


def validate_preflight(config: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    parents = config["failure_parents"]
    cq_path = ROOT / parents["v19cq_report"]["path"]
    fail_path = ROOT / parents["v19x2_failure_report"]["path"]
    x2_config_path = ROOT / parents["v19x2_config"]["path"]
    cd_config_path = ROOT / parents["v19cd_environment_config"]["path"]
    cd_report_path = ROOT / parents["v19cd_environment_report"]["path"]
    w5_path = ROOT / parents["v19w5_report"]["path"]
    cq, fail, x2, cd_report, w5 = map(load_json, (cq_path, fail_path, x2_config_path, cd_report_path, w5_path))
    scratch = scratch_audit(config)
    execution = config["unchanged_execution"]
    checks = {
        "v19cq_failure_exact": (
            sha256(cq_path) == parents["v19cq_report"]["sha256"]
            and cq["status"] == parents["v19cq_report"]["required_status"]
            and parents["v19cq_report"]["required_exception_fragment"] in cq["exception"]
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
        "prior_environment_contract_exact": (
            sha256(cd_config_path) == parents["v19cd_environment_config"]["sha256"]
            and sha256(cd_report_path) == parents["v19cd_environment_report"]["sha256"]
            and cd_report["status"] == parents["v19cd_environment_report"]["required_status"]
            and all(cd_report["environment"]["checks"].values())
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
        ),
        "scratch_precedes_combination_and_fit": scratch["only_permitted_files"] and not scratch["combined_or_fit_products"],
        "authorization_environment_only": (
            config["authorization"]["change_launch_environment_only"]
            and not config["authorization"]["change_x2_config_runner_or_scientific_rules"]
            and not config["authorization"]["run_v19bs_or_derive_action"]
            and not config["authorization"]["open_lensing_halo_gravity_holdout_or_solar_optimization"]
        ),
    }
    if not all(checks.values()):
        raise RuntimeError(f"V19CR preflight failed: {checks}; scratch={scratch}")
    return checks, scratch


def execute(config: dict[str, Any]) -> dict[str, Any]:
    preflight, scratch = validate_preflight(config)
    environment_probe, probe_execution = probe_environment(config)
    parents = config["failure_parents"]
    execution = config["unchanged_execution"]
    x2_config_path = ROOT / parents["v19x2_config"]["path"]
    x2_config_sha_before = sha256(x2_config_path)
    x2_command = config["environment"]["launch_prefix"] + [
        str(ROOT / execution["v19x2_runner"]),
        "--config", str(x2_config_path),
        "--output", str(ROOT / execution["v19x2_output"]),
        "--scratch", execution["v19x2_scratch"],
        "--response-report", str(ROOT / parents["v19w5_report"]["path"]),
    ]
    x2_exec = run_logged(x2_command, ROOT / config["outputs"]["v19x2_log"])
    x2_report = load_json(ROOT / execution["v19x2_output"] / "report.json")
    passing_status = "unified_spectral_combination_commissioning_passed_and_full_regional_fits_authorized"
    failing_status = "unified_spectral_combination_commissioning_gate_failed"
    x2_scientific = x2_report["status"] in {passing_status, failing_status}

    br_exec = None
    br_report = None
    if x2_report["status"] == passing_status:
        br_command = config["environment"]["launch_prefix"] + [
            str(ROOT / execution["v19br_runner"]), "--config", str(ROOT / execution["v19br_config"]), "--execute"
        ]
        br_exec = run_logged(br_command, ROOT / config["outputs"]["v19br_log"])
        br_report = load_json(ROOT / "results" / "sigma_v19br_target_sealed_terminal_chain" / "report.json")

    gates = {
        "failure_and_environment_parents_exact": all(preflight.values()),
        "corrected_x2_config_exact_and_unchanged": sha256(x2_config_path) == x2_config_sha_before == parents["v19x2_config"]["sha256"],
        "ciao_python_modules_and_executables_exact": all(environment_probe["checks"].values()),
        "partial_scratch_contains_no_combined_or_fit_product": preflight["scratch_precedes_combination_and_fit"],
        "v19x2_reaches_registered_scientific_disposition": x2_scientific and x2_report["runner_sha256"] == execution["v19x2_runner_sha256"],
        "if_v19x2_passes_unchanged_v19br_reaches_source_decision": (
            x2_report["status"] != passing_status
            or (br_exec is not None and br_exec["returncode"] == 0 and br_report is not None and br_report["status"] == "terminal_chain_complete" and br_report.get("source_decision") is not None)
        ),
        "no_lensing_halo_action_gravity_holdout_or_solar_access": (
            not config["authorization"]["open_lensing_halo_gravity_holdout_or_solar_optimization"]
            and not x2_report["replacement_cluster_lensing_target_opened"]
            and not x2_report["gravity_formula_or_parameter_changed"]
            and (br_report is None or not br_report["lensing_halo_action_gravity_or_holdout_payload_opened"])
        ),
    }
    decision = (
        "v19x2_valid_scientific_gate_failure_no_full_source_chain" if x2_report["status"] == failing_status
        else "run_frozen_v19bs_disposition_next" if br_report is not None and br_report.get("status") == "terminal_chain_complete"
        else "v19x2_or_source_chain_execution_incomplete"
    )
    if not all(gates.values()):
        raise RuntimeError(f"V19CR post-execution gates failed: {gates}; decision={decision}; x2_status={x2_report['status']}")
    return {
        "protocol_version": config["protocol_version"],
        "status": "v19x2_ciao_launch_remediation_completed",
        "decision": decision,
        "generated_utc": datetime.now(UTC).isoformat(),
        "config_sha256": sha256(DEFAULT_CONFIG),
        "preflight": preflight,
        "preexecution_scratch_audit": scratch,
        "environment_probe": environment_probe,
        "environment_probe_execution": probe_execution,
        "v19x2_config_sha256_before_and_after": x2_config_sha_before,
        "v19x2_execution": x2_exec,
        "v19x2_report": x2_report,
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
            "protocol_version": config["protocol_version"],
            "status": "v19x2_ciao_launch_remediation_failed_closed",
            "exception": f"{type(exc).__name__}: {exc}",
            "generated_utc": datetime.now(UTC).isoformat(),
            "authorization_boundary": {"v19bs_run": False, "action_derived": False, "target_or_gravity_opened": False, "solar_optimized": False},
            "claim_boundary": config["claim_boundary"],
        }
    atomic_json(output, report)
    print(json.dumps({key: report.get(key) for key in ("status", "decision", "exception")}, indent=2, sort_keys=True))
    if report["status"] != "v19x2_ciao_launch_remediation_completed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
