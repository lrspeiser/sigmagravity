#!/usr/bin/env python3
"""Retry one frozen V19W5 RMF cell, then resume the unchanged V19BR chain."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "sigma_v19ci_single_rmf_retry.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
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


def verify_static_hashes(config: dict[str, Any]) -> dict[str, str]:
    observed: dict[str, str] = {}
    for name, spec in config["parents"].items():
        path = ROOT / spec["path"]
        actual = sha256(path)
        if actual != spec["sha256"]:
            raise RuntimeError(f"V19CI parent changed: {name}: {actual}")
        observed[name] = actual
    runner = ROOT / config["implementation"]["runner"]["path"]
    if runner.resolve() != Path(__file__).resolve():
        raise RuntimeError("V19CI config identifies another runner")
    if sha256(runner) != config["implementation"]["runner"]["sha256"]:
        raise RuntimeError("V19CI frozen runner changed")
    return observed


def active_recovery_processes() -> list[str]:
    completed = subprocess.run(
        ["ps", "-eo", "pid=,cmd="],
        check=True,
        capture_output=True,
        text=True,
    )
    needles = (
        "run_sigma_v19cd_v19w5_environment_remediation.py",
        "run_sigma_v19w5_ccd7_hardened_response_recovery.py",
    )
    return [
        line.strip()
        for line in completed.stdout.splitlines()
        if any(needle in line for needle in needles)
    ]


def workspace_counts(scratch: Path) -> dict[str, Any]:
    def directories(name: str) -> list[str]:
        path = scratch / name
        return sorted(row.name for row in path.iterdir() if row.is_dir()) if path.is_dir() else []

    completed = directories("completed")
    partial = directories("partial")
    failed = directories("failed_attempts")
    return {
        "scratch": str(scratch),
        "completed_directories": len(completed),
        "completed_cell_reports": len(
            list((scratch / "completed").glob("*/cell_report.json"))
        ),
        "partial_directories": partial,
        "failed_attempt_directories": failed,
    }


def inspect_pre_retry_boundary(config: dict[str, Any]) -> dict[str, Any]:
    boundary = config["initial_failure_boundary"]
    progress_path = ROOT / boundary["progress_path"]
    progress = load_json(progress_path)
    failure = load_json(ROOT / boundary["v19w5_failure_path"])
    v19cd = load_json(ROOT / boundary["v19cd_failure_path"])
    scratch = Path(config["retry"]["scratch"])
    workspace = workspace_counts(scratch)
    partial = scratch / "partial" / boundary["failed_token"]
    log = partial / "logs" / "specextract.log"
    checks = {
        "no_prior_recovery_process_remains": not active_recovery_processes(),
        "v19cd_failed_only_at_remediated_v19w5": (
            v19cd.get("status") == boundary["v19cd_status"]
            and v19cd.get("exception") == boundary["v19cd_exception"]
        ),
        "v19w5_failure_is_exact_single_rmf_cell": (
            failure.get("status") == boundary["v19w5_status"]
            and boundary["failed_cell"] in failure.get("exception", "")
            and "specextract failed" in failure.get("exception", "")
        ),
        "progress_records_383_successes_and_one_failure": (
            progress.get("recovery_completed_cells") == 383
            and progress.get("missing_cells_at_launch") == 384
            and set(progress.get("recovery_failures", {}))
            == {boundary["failed_cell"]}
            and progress.get("base_archive_modified") is False
            and progress.get("gravity_formula_or_parameter_changed") is False
        ),
        "workspace_has_383_completed_and_one_exact_partial": (
            workspace["completed_directories"]
            == workspace["completed_cell_reports"]
            == 383
            and workspace["partial_directories"] == [boundary["failed_token"]]
            and workspace["failed_attempt_directories"] == []
        ),
        "failed_partial_log_is_exact_rmf_creation_failure": (
            log.is_file()
            and sha256(log) == boundary["failed_log_sha256"]
            and "ERROR Failed to create RMF" in log.read_text(
                encoding="utf-8", errors="replace"
            )
        ),
        "failed_cell_has_no_completed_checkpoint": not (
            scratch / "completed" / boundary["failed_cell"] / "cell_report.json"
        ).exists(),
    }
    if not all(checks.values()):
        raise RuntimeError(f"V19CI initial boundary changed: {checks}")
    return {"checks": checks, "workspace": workspace}


def build_command(config: dict[str, Any], stage: str) -> list[str]:
    environment = config["environment"]
    prefix = [
        environment["conda_executable"],
        "run",
        "--no-capture-output",
        "-n",
        environment["environment_name"],
        "python",
    ]
    retry = config["retry"]
    if stage == "v19w5":
        return prefix + [
            str(ROOT / retry["v19w5_runner"]),
            "--config",
            str(ROOT / retry["v19w5_config"]),
            "--output",
            str(ROOT / retry["v19w5_output"]),
            "--scratch",
            retry["scratch"],
            "--base-scratch",
            retry["protected_base_scratch"],
        ]
    if stage == "v19br":
        return prefix + [
            str(ROOT / retry["v19br_runner"]),
            "--config",
            str(ROOT / retry["v19br_config"]),
            "--execute",
        ]
    raise ValueError(f"unknown V19CI stage: {stage}")


def run_logged(command: list[str], log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8", buffering=1) as stream:
        print(f"{datetime.now(UTC).isoformat()} command={command}", file=stream)
        completed = subprocess.run(
            command,
            cwd=ROOT,
            stdout=stream,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
        print(
            f"{datetime.now(UTC).isoformat()} returncode={completed.returncode}",
            file=stream,
        )
    return completed.returncode


def validate_v19w5_pass(config: dict[str, Any]) -> dict[str, Any]:
    report_path = ROOT / config["retry"]["v19w5_report"]
    report = load_json(report_path)
    checks = {
        "terminal_status_passes": report.get("status")
        == "ccd7_hardened_unified_5082_response_archive_passed",
        "every_gate_passes": bool(report.get("gates"))
        and all(report["gates"].values()),
        "all_cells_and_products_present": report.get("unified_cells") == 5082
        and report.get("unified_product_files") == 20328,
        "exact_384_recovery_cells": report.get("recovered_cells") == 384,
        "protected_base_unchanged": report.get("base_v19w_archive_modified")
        is False,
        "target_and_gravity_remained_sealed": report.get(
            "lensing_halo_or_gravity_payload_opened"
        )
        is False
        and report.get("gravity_formula_or_parameter_changed") is False,
    }
    if not all(checks.values()):
        raise RuntimeError(f"V19CI V19W5 pass invalid: {checks}")
    return {"checks": checks, "report_sha256": sha256(report_path)}


def validate_v19br_pass(config: dict[str, Any]) -> dict[str, Any]:
    report_path = ROOT / config["retry"]["v19br_report"]
    report = load_json(report_path)
    checks = {
        "terminal_chain_complete": report.get("status") == "terminal_chain_complete",
        "no_active_base_pids": report.get("active_base_pids") == [],
        "every_stage_passed": bool(report.get("stages"))
        and all(row.get("state") == "passed" for row in report["stages"]),
        "target_remained_sealed": report.get(
            "lensing_halo_action_gravity_or_holdout_payload_opened"
        )
        is False,
    }
    if not all(checks.values()):
        raise RuntimeError(f"V19CI V19BR pass invalid: {checks}")
    return {
        "checks": checks,
        "report_sha256": sha256(report_path),
        "source_decision": report.get("source_decision"),
    }


def execute(config_path: Path) -> dict[str, Any]:
    config = load_json(config_path)
    parent_hashes = verify_static_hashes(config)
    boundary = inspect_pre_retry_boundary(config)
    scratch = Path(config["retry"]["scratch"])
    token = config["initial_failure_boundary"]["failed_token"]
    source = scratch / "partial" / token
    destination = scratch / "failed_attempts" / f"{token}_rmf_attempt1"
    if destination.exists():
        raise RuntimeError("V19CI failed-attempt destination already exists")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(source), str(destination))

    w5_returncode = run_logged(
        build_command(config, "v19w5"), ROOT / config["outputs"]["v19w5_log"]
    )
    if w5_returncode != 0:
        raise RuntimeError(f"V19CI single-cell V19W5 retry exited {w5_returncode}")
    v19w5 = validate_v19w5_pass(config)

    br_returncode = run_logged(
        build_command(config, "v19br"), ROOT / config["outputs"]["v19br_log"]
    )
    if br_returncode != 0:
        raise RuntimeError(f"V19CI resumed V19BR exited {br_returncode}")
    v19br = validate_v19br_pass(config)
    return {
        "status": "single_rmf_retry_passed_and_v19br_terminal_chain_complete",
        "parent_hashes": parent_hashes,
        "initial_failure_boundary": boundary,
        "failed_attempt_preserved_at": str(destination),
        "v19w5": v19w5,
        "v19br": v19br,
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
        payload = {
            "parent_hashes": verify_static_hashes(config),
            "initial_failure_boundary": inspect_pre_retry_boundary(config),
            "v19w5_command": build_command(config, "v19w5"),
            "v19br_command": build_command(config, "v19br"),
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
        return
    try:
        result = execute(config_path)
    except Exception as exc:  # noqa: BLE001 - preserve the exact fail-closed state
        result = {
            "status": "single_rmf_retry_failed_closed",
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
        "single_rmf_retry_passed_and_v19br_terminal_chain_complete"
    ):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
