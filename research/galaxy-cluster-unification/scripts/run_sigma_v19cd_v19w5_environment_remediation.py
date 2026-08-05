#!/usr/bin/env python3
"""Repair the V19W5 launch environment and resume the frozen V19BR chain.

This successor is intentionally operational.  It may run only after the exact
pre-cell ``dmkeypar`` launch failure registered in V19W5.  It executes the
byte-identical frozen V19W5 runner in a fresh scratch directory through the
declared CIAO conda environment, then resumes the unchanged V19BR chain.
"""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import hashlib
import json
import os
from pathlib import Path
from pathlib import PurePosixPath
import shlex
import subprocess
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "sigma_v19cd_v19w5_environment_remediation.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def validate_parent_hashes(config: dict[str, Any]) -> dict[str, str]:
    observed: dict[str, str] = {}
    for name, spec in config["parents"].items():
        path = ROOT / spec["path"]
        actual = sha256(path)
        if actual != spec["sha256"]:
            raise RuntimeError(f"V19CD parent changed: {name}: {actual}")
        observed[name] = actual
    runner = ROOT / config["implementation"]["runner"]["path"]
    if runner.resolve() != Path(__file__).resolve():
        raise RuntimeError("V19CD config identifies another runner")
    if sha256(runner) != config["implementation"]["runner"]["sha256"]:
        raise RuntimeError("V19CD frozen runner changed")
    return observed


def task_workspace_state(path: Path) -> dict[str, Any]:
    def directory_count(name: str) -> int:
        root = path / name
        return len(list(root.iterdir())) if root.is_dir() else 0

    completed_reports = (
        len(list((path / "completed").glob("*/cell_report.json")))
        if (path / "completed").is_dir()
        else 0
    )
    return {
        "path": str(path),
        "exists": path.exists(),
        "completed_cell_reports": completed_reports,
        "failed_attempt_directories": directory_count("failed_attempts"),
        "partial_attempt_directories": directory_count("partial"),
        "quarantine_directories": directory_count("quarantine"),
        "geometry_prep_artifacts_allowed": True,
    }


def validate_failure_boundary(config: dict[str, Any]) -> dict[str, Any]:
    expected = config["required_failure_boundary"]
    v19w5 = load_json(ROOT / config["parents"]["v19w5_failure_report"]["path"])
    v19br = load_json(ROOT / config["parents"]["v19br_failure_report"]["path"])
    base = load_json(ROOT / config["parents"]["v19w_base_terminal_report"]["path"])
    workspace = task_workspace_state(Path(config["failed_workspace"]["path"]))
    checks = {
        "v19w5_status_exact": v19w5.get("status")
        == expected["v19w5_status"],
        "v19w5_exception_exact": v19w5.get("exception")
        == expected["v19w5_exception"],
        "v19w5_base_not_modified": v19w5.get(
            "base_v19w_archive_modified_by_protocol"
        )
        is False,
        "v19w5_no_combination_or_fit": v19w5.get("spectrum_combined_or_fitted")
        is False,
        "v19w5_no_target_or_gravity_access": v19w5.get(
            "lensing_halo_or_gravity_payload_opened"
        )
        is False
        and v19w5.get("gravity_formula_or_parameter_changed") is False,
        "v19br_status_exact": v19br.get("status")
        == expected["v19br_status"],
        "v19br_exception_exact": v19br.get("exception")
        == expected["v19br_exception"],
        "v19br_target_sealed": v19br.get(
            "lensing_halo_action_gravity_or_holdout_payload_opened"
        )
        is False,
        "base_terminal_state_exact": base.get("status")
        == expected["base_status"]
        and int(base.get("completed_cells", -1)) == expected["base_completed_cells"]
        and int(base.get("expected_cells", -1)) == expected["base_expected_cells"],
        "failure_preceded_every_recovery_cell": workspace[
            "completed_cell_reports"
        ]
        == workspace["failed_attempt_directories"]
        == workspace["partial_attempt_directories"]
        == workspace["quarantine_directories"]
        == 0,
    }
    if not all(checks.values()):
        raise RuntimeError(f"V19CD failure boundary changed: {checks}")
    return {
        "checks": checks,
        "failed_workspace": workspace,
        "base_missing_cells": expected["base_expected_cells"]
        - expected["base_completed_cells"],
    }


def probe_environment(config: dict[str, Any]) -> dict[str, Any]:
    environment = config["environment"]
    conda_text = environment["conda_executable"]
    prefix_text = environment["environment_prefix"]
    if os.name != "nt" and (
        not Path(conda_text).is_file() or not Path(prefix_text).is_dir()
    ):
        raise RuntimeError("V19CD declared conda executable or environment is absent")
    probe_code = (
        "import importlib, json, shutil, sys; "
        f"commands={environment['required_executables']!r}; "
        f"modules={environment['required_python_modules']!r}; "
        "paths={name: shutil.which(name) for name in commands}; "
        "loaded={name: bool(importlib.import_module(name)) for name in modules}; "
        "print(json.dumps({'python': sys.executable, 'commands': paths, "
        "'modules': loaded}, sort_keys=True))"
    )
    command = [
        conda_text,
        "run",
        "--no-capture-output",
        "-n",
        environment["environment_name"],
        "python",
        "-c",
        probe_code,
    ]
    host_command = (
        ["wsl", "-e", "bash", "-lc", shlex.join(command)]
        if os.name == "nt"
        else command
    )
    completed = subprocess.run(
        host_command, check=False, capture_output=True, text=True
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "V19CD environment probe failed: " + completed.stdout + completed.stderr
        )
    try:
        payload = json.loads(completed.stdout.strip().splitlines()[-1])
    except (IndexError, json.JSONDecodeError) as exc:
        raise RuntimeError("V19CD environment probe returned invalid JSON") from exc
    expected_bin = PurePosixPath(prefix_text) / "bin"
    checks = {
        "python_from_declared_environment": PurePosixPath(payload["python"])
        == expected_bin / "python",
        "all_required_executables_resolve": all(payload["commands"].values()),
        "all_executables_from_declared_environment": all(
            PurePosixPath(value).parent == expected_bin
            for value in payload["commands"].values()
            if value
        ),
        "all_required_python_modules_import": all(payload["modules"].values()),
    }
    if not all(checks.values()):
        raise RuntimeError(f"V19CD environment gate failed: {checks}")
    return {"checks": checks, **payload}


def build_command(config: dict[str, Any], stage: str) -> list[str]:
    environment = config["environment"]
    remediation = config["remediation"]
    prefix = [
        environment["conda_executable"],
        "run",
        "--no-capture-output",
        "-n",
        environment["environment_name"],
        "python",
    ]
    if stage == "v19w5":
        return prefix + [
            str(ROOT / remediation["v19w5_runner"]),
            "--config",
            str(ROOT / remediation["v19w5_config"]),
            "--output",
            str(ROOT / remediation["v19w5_output"]),
            "--scratch",
            remediation["fresh_recovery_scratch"],
            "--base-scratch",
            remediation["protected_base_scratch"],
        ]
    if stage == "v19br":
        return prefix + [
            str(ROOT / remediation["v19br_runner"]),
            "--config",
            str(ROOT / remediation["v19br_config"]),
            "--execute",
        ]
    raise ValueError(f"unknown V19CD stage: {stage}")


def run_logged(command: list[str], log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8", buffering=1) as handle:
        print(f"{datetime.now(UTC).isoformat()} command={command}", file=handle)
        completed = subprocess.run(
            command,
            cwd=ROOT,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
        print(
            f"{datetime.now(UTC).isoformat()} returncode={completed.returncode}",
            file=handle,
        )
    return completed.returncode


def validate_v19w5_pass(config: dict[str, Any]) -> dict[str, Any]:
    report = load_json(ROOT / config["remediation"]["v19w5_report"])
    checks = {
        "status_passes": report.get("status")
        == "ccd7_hardened_unified_5082_response_archive_passed",
        "all_gates_pass": bool(report.get("gates"))
        and all(report["gates"].values()),
        "unified_5082": report.get("unified_cells") == 5082,
        "products_20328": report.get("unified_product_files") == 20328,
        "base_unchanged": report.get("base_v19w_archive_modified") is False,
        "target_sealed": report.get("lensing_halo_or_gravity_payload_opened")
        is False,
        "gravity_unchanged": report.get("gravity_formula_or_parameter_changed")
        is False,
    }
    if not all(checks.values()):
        raise RuntimeError(f"V19CD V19W5 successor pass invalid: {checks}")
    return {"checks": checks, "report_sha256": sha256(ROOT / config["remediation"]["v19w5_report"])}


def validate_v19br_pass(config: dict[str, Any]) -> dict[str, Any]:
    report = load_json(ROOT / config["remediation"]["v19br_report"])
    checks = {
        "terminal_chain_complete": report.get("status") == "terminal_chain_complete",
        "no_active_base_pids": report.get("active_base_pids") == [],
        "every_stage_passed": bool(report.get("stages"))
        and all(row.get("state") == "passed" for row in report["stages"]),
        "target_sealed": report.get(
            "lensing_halo_action_gravity_or_holdout_payload_opened"
        )
        is False,
    }
    if not all(checks.values()):
        raise RuntimeError(f"V19CD resumed V19BR pass invalid: {checks}")
    return {
        "checks": checks,
        "report_sha256": sha256(ROOT / config["remediation"]["v19br_report"]),
        "source_decision": report.get("source_decision"),
    }


def execute(config_path: Path) -> dict[str, Any]:
    config_path = config_path.resolve()
    config = load_json(config_path)
    parent_hashes = validate_parent_hashes(config)
    failure = validate_failure_boundary(config)
    environment = probe_environment(config)
    fresh = Path(config["remediation"]["fresh_recovery_scratch"])
    if fresh.exists():
        raise RuntimeError("V19CD fresh recovery scratch already exists")
    if fresh == Path(config["remediation"]["protected_base_scratch"]):
        raise RuntimeError("V19CD fresh scratch equals protected base")

    w5_returncode = run_logged(
        build_command(config, "v19w5"),
        ROOT / config["outputs"]["v19w5_log"],
    )
    if w5_returncode != 0:
        raise RuntimeError(f"V19CD remediated V19W5 exited {w5_returncode}")
    w5 = validate_v19w5_pass(config)

    br_returncode = run_logged(
        build_command(config, "v19br"),
        ROOT / config["outputs"]["v19br_log"],
    )
    if br_returncode != 0:
        raise RuntimeError(f"V19CD resumed V19BR exited {br_returncode}")
    br = validate_v19br_pass(config)
    return {
        "status": "v19w5_environment_remediated_and_v19br_terminal_chain_complete",
        "parent_hashes": parent_hashes,
        "failure_boundary": failure,
        "environment": environment,
        "v19w5": w5,
        "v19br": br,
        "fresh_recovery_scratch": str(fresh),
        "initial_failure_report_preserved": (
            ROOT / config["parents"]["v19w5_failure_report"]["path"]
        ).is_file(),
        "lensing_halo_action_gravity_or_holdout_payload_opened": False,
        "gravity_formula_or_parameter_changed": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.preflight_only == args.execute:
        raise SystemExit("select exactly one of --preflight-only or --execute")
    config_path = args.config.resolve()
    config = load_json(config_path)
    if args.preflight_only:
        result = {
            "parent_hashes": validate_parent_hashes(config),
            "failure_boundary": validate_failure_boundary(config),
            "environment": probe_environment(config),
            "v19w5_command": build_command(config, "v19w5"),
            "v19br_command": build_command(config, "v19br"),
        }
        print(json.dumps(result, indent=2, sort_keys=True))
        return
    try:
        result = execute(config_path)
    except Exception as exc:  # noqa: BLE001 - persist exact fail-closed state
        result = {
            "status": "v19w5_environment_remediation_failed_closed",
            "exception": f"{type(exc).__name__}: {exc}",
            "lensing_halo_action_gravity_or_holdout_payload_opened": False,
            "gravity_formula_or_parameter_changed": False,
        }
    report = {
        "protocol_version": config["protocol_version"],
        "generated_utc": datetime.now(UTC).isoformat(),
        "config_sha256": sha256(config_path),
        "runner_sha256": sha256(Path(__file__).resolve()),
        **result,
        "claim_boundary": config["claim_boundary"],
    }
    output = ROOT / config["outputs"]["report"]
    atomic_json(output, report)
    print(output)
    print(report["status"])
    if report["status"] != (
        "v19w5_environment_remediated_and_v19br_terminal_chain_complete"
    ):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
