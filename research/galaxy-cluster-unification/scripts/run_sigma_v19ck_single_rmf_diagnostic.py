#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "sigma_v19ck_single_rmf_diagnostic.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def preflight(config_path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    config = load_json(config_path)
    boundary = config["failure_boundary"]
    scratch = Path(boundary["scratch"])
    current = Path(boundary["current_partial"])
    attempt1 = Path(boundary["preserved_attempt1"])
    prefix = Path(config["environment"]["prefix"])
    parent_checks = {
        key: sha256(ROOT / spec["path"]) == spec["sha256"]
        for key, spec in config["parent"].items()
    }
    parent_report = load_json(ROOT / config["parent"]["v19ci_report"]["path"])
    current_log = current / boundary["failed_log_relative"]
    attempt1_log = attempt1 / boundary["failed_log_relative"]
    source = current / "e" / "s.fits"
    background = current / "e" / "b.fits"
    completed = list((scratch / "completed").glob("*/cell_report.json"))
    gates = {
        "parents_and_ciao_executables_hash_exact": (
            all(parent_checks.values())
            and parent_report["status"] == config["parent"]["v19ci_report"]["required_status"]
            and sha256(prefix / "bin" / "specextract") == config["environment"]["specextract_sha256"]
            and sha256(prefix / "bin" / "mkacisrmf") == config["environment"]["mkacisrmf_sha256"]
        ),
        "two_failed_logs_byte_identical_and_exact": (
            current_log.read_bytes() == attempt1_log.read_bytes()
            and sha256(current_log) == boundary["failed_log_sha256_both_attempts"]
            and b"ERROR Failed to create RMF" in current_log.read_bytes()
        ),
        "current_copied_subsets_hash_exact": (
            sha256(source) == boundary["current_source_subset_sha256"]
            and sha256(background) == boundary["current_background_subset_sha256"]
        ),
        "recovery_has_383_completed_and_failed_checkpoint_absent": (
            len(completed) == boundary["completed_recovery_checkpoints"]
            and not (scratch / "completed" / boundary["cell"] / "cell_report.json").exists()
        ),
        "diagnostic_workspace_absent_before_single_run": not Path(config["diagnostic"]["workspace"]).exists(),
        "diagnostic_inputs_copied_without_modifying_recovery": True,
        "verbose_error_and_direct_mkacisrmf_disposition_recorded": True,
        "no_final_retry_target_action_or_gravity_access": (
            not config["authorization"]["modify_recovery_scratch_or_completed_checkpoints"]
            and not config["authorization"]["authorize_final_recovery_retry"]
            and not config["authorization"]["resume_v19br"]
            and not config["authorization"]["open_lensing_halo_action_gravity_or_holdout"]
            and not config["authorization"]["derive_or_select_action_or_change_gravity_parameter"]
        ),
    }
    if set(gates) != set(config["required_gates"]):
        raise RuntimeError("implemented and declared V19CK gates differ")
    return {
        "config": config,
        "gates": gates,
        "parent_checks": parent_checks,
        "source": source,
        "background": background,
        "current_log": current_log,
        "attempt1_log": attempt1_log,
    }


def run_logged(command: list[str], log_path: Path, env: dict[str, str]) -> dict[str, Any]:
    completed = subprocess.run(command, check=False, capture_output=True, text=True, env=env)
    combined = completed.stdout + ("\n" if completed.stdout and completed.stderr else "") + completed.stderr
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(combined, encoding="utf-8")
    diagnostic_lines = [
        line for line in combined.splitlines()
        if any(token in line.lower() for token in ("error", "warning", "caldb", "chip", "wmap", "rmf", "range", "outside", "invalid"))
    ]
    return {
        "command": command,
        "returncode": completed.returncode,
        "log": log_path.relative_to(ROOT).as_posix(),
        "log_sha256": sha256(log_path),
        "log_bytes": log_path.stat().st_size,
        "diagnostic_lines": diagnostic_lines[-120:],
    }


def execute(config_path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    state = preflight(config_path)
    if not all(state["gates"].values()):
        raise RuntimeError(f"V19CK preflight failed: {state['gates']}")
    config = state["config"]
    workspace = Path(config["diagnostic"]["workspace"])
    event_dir = workspace / "e"
    product_dir = workspace / "products"
    temp_dir = workspace / "tmp"
    pfiles_dir = workspace / "pfiles"
    for path in (event_dir, product_dir, temp_dir, pfiles_dir):
        path.mkdir(parents=True, exist_ok=False)
    shutil.copy2(state["source"], event_dir / "s.fits")
    shutil.copy2(state["background"], event_dir / "b.fits")

    prefix = Path(config["environment"]["prefix"])
    env = os.environ.copy()
    env["PFILES"] = f"{pfiles_dir};{prefix / 'param'}"
    env["ASCDS_WORK_PATH"] = str(temp_dir)
    env["ASCDS_TMP"] = str(temp_dir)
    settings = config["diagnostic"]["response_settings"]
    outroot = product_dir / config["failure_boundary"]["cell"]
    source_filter = f"{event_dir / 's.fits'}[sky=region({settings['fov']})]"
    background_filter = f"{event_dir / 'b.fits'}[sky=region({settings['fov']})]"
    command = [
        str(prefix / "bin" / "specextract"),
        f"infile={source_filter}", f"outroot={outroot}", f"bkgfile={background_filter}",
        f"asp=@{settings['aspect']}", f"mskfile={settings['mask']}", f"badpixfile={settings['badpix']}",
        "dafile=CALDB", "bkgresp=no", f"weight={settings['weight']}", f"weight_rmf={settings['weight_rmf']}",
        "resp_pos=CENTROID", f"refcoord={settings['refcoord']}", "correctpsf=no", "combine=no",
        "grouptype=NONE", "binspec=NONE", "bkg_grouptype=NONE", "bkg_binspec=NONE",
        f"energy={settings['energy']}", f"energy_wmap={settings['energy_wmap']}", f"binwmap={settings['binwmap']}",
        "binarfwmap=1", "parallel=no", "nproc=1", f"tmpdir={temp_dir}", "clobber=no", "verbose=5", "mode=h",
    ]
    speclog = ROOT / config["outputs"]["specextract_log"]
    spec_result = run_logged(command, speclog, env)

    pha = Path(f"{outroot}.pi")
    rmf = Path(f"{outroot}.rmf")
    direct_result: dict[str, Any] | None = None
    if spec_result["returncode"] != 0 and pha.exists() and not rmf.exists():
        direct_out = workspace / "direct_mkacisrmf.rmf"
        direct_command = [
            str(prefix / "bin" / "mkacisrmf"), "infile=CALDB", f"outfile={direct_out}",
            f"wmap={pha}[WMAP]", f"energy={settings['energy']}", "channel=1:1024:1", "chantype=PI",
            f"ccd_id={settings['ccd_id']}", "chipx=", "chipy=", "gain=CALDB", "obsfile=", "asolfile=",
            "logfile=", "clobber=no", "verbose=5", "mode=h",
        ]
        directlog = ROOT / config["outputs"]["mkacisrmf_log"]
        direct_result = run_logged(direct_command, directlog, env)

    recovery_hashes_after = {
        "current_source_subset": sha256(state["source"]),
        "current_background_subset": sha256(state["background"]),
        "current_failed_log": sha256(state["current_log"]),
        "attempt1_failed_log": sha256(state["attempt1_log"]),
    }
    direct_code = None if direct_result is None else direct_result["returncode"]
    if spec_result["returncode"] == 0:
        decision = "verbose_diagnostic_did_not_reproduce_final_retry_still_not_authorized"
    elif direct_code == 0:
        decision = "specextract_orchestration_failure_isolated_remediation_preregistration_required"
    elif direct_code is not None:
        decision = "direct_mkacisrmf_failure_captured_remediation_preregistration_required"
    else:
        decision = "rmf_failure_reproduced_without_direct_diagnostic_remediation_not_authorized"

    return {
        "protocol_version": config["protocol_version"],
        "status": "completed_single_rmf_diagnostic",
        "decision": decision,
        "generated_utc": datetime.now(UTC).isoformat(),
        "config": config_path.relative_to(ROOT).as_posix(),
        "config_sha256": sha256(config_path),
        "preflight_gate_results": state["gates"],
        "parent_checks": state["parent_checks"],
        "diagnostic_workspace": str(workspace),
        "copied_input_hashes": {
            "source": sha256(event_dir / "s.fits"),
            "background": sha256(event_dir / "b.fits"),
        },
        "specextract": spec_result,
        "direct_mkacisrmf": direct_result,
        "workspace_products": [str(path.relative_to(workspace)) for path in sorted(workspace.rglob("*")) if path.is_file()],
        "recovery_hashes_after": recovery_hashes_after,
        "recovery_unchanged": (
            recovery_hashes_after["current_source_subset"] == config["failure_boundary"]["current_source_subset_sha256"]
            and recovery_hashes_after["current_background_subset"] == config["failure_boundary"]["current_background_subset_sha256"]
            and recovery_hashes_after["current_failed_log"] == config["failure_boundary"]["failed_log_sha256_both_attempts"]
            and recovery_hashes_after["attempt1_failed_log"] == config["failure_boundary"]["failed_log_sha256_both_attempts"]
        ),
        "authorization_boundary": {
            "final_retry_authorized": False,
            "cell_drop_or_response_change_authorized": False,
            "v19br_resume_authorized": False,
            "target_action_or_gravity_accessed": False,
        },
        "claim_boundary": config["claim_boundary"],
    }


def main() -> None:
    config = load_json(DEFAULT_CONFIG)
    output = ROOT / config["outputs"]["report"]
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        report = execute()
    except Exception as exc:
        report = {
            "protocol_version": config["protocol_version"],
            "status": "single_rmf_diagnostic_failed_closed",
            "exception": f"{type(exc).__name__}: {exc}",
            "generated_utc": datetime.now(UTC).isoformat(),
            "config_sha256": sha256(DEFAULT_CONFIG),
            "authorization_boundary": {"final_retry_authorized": False, "target_action_or_gravity_accessed": False},
            "claim_boundary": config["claim_boundary"],
        }
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: report.get(key) for key in ("status", "decision", "exception")}, indent=2, sort_keys=True))
    if report["status"] != "completed_single_rmf_diagnostic":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
