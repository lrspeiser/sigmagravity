#!/usr/bin/env python3
"""Execute the frozen V19W5-to-V19BQ chain in fail-closed order."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "sigma_v19br_target_sealed_terminal_chain.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_static(config: dict[str, Any]) -> None:
    if config.get("freeze_state") != "frozen_before_terminal_v19w5_or_source_results":
        raise RuntimeError("V19BR terminal chain is not in its frozen pre-result state")
    for section in ("parents", "implementation"):
        for name, spec in config[section].items():
            path = ROOT / spec["path"]
            if not path.is_file() or sha256(path) != spec["sha256"]:
                raise RuntimeError(f"V19BR {section[:-1]} changed: {name}")
    runner = ROOT / config["implementation"]["runner"]["path"]
    if runner.resolve() != Path(__file__).resolve():
        raise RuntimeError("V19BR configuration names another runner")


def running_base_processes() -> list[int]:
    scripts = ROOT / "scripts"
    if str(scripts) not in sys.path:
        sys.path.insert(0, str(scripts))
    import run_sigma_v19w3_full_response_recovery as v19w3

    return [int(pid) for pid in v19w3.running_base_processes()]


def artifact_state(stage: dict[str, Any]) -> dict[str, Any]:
    path = ROOT / stage["artifact"]
    failure_paths = [ROOT / value for value in stage.get("failure_artifacts", [])]
    if not path.is_file():
        failures = [value.relative_to(ROOT).as_posix() for value in failure_paths if value.is_file()]
        return {
            "state": "failed_closed" if failures else "pending",
            "artifact": stage["artifact"],
            "failure_artifacts": failures,
        }
    try:
        payload = load_json(path)
    except (OSError, json.JSONDecodeError) as exc:
        return {
            "state": "failed_closed",
            "artifact": stage["artifact"],
            "error": f"{type(exc).__name__}: {exc}",
        }
    key = stage["state_key"]
    expected = stage.get("expected_values", [stage.get("expected_value")])
    checks = {"expected_state_value": payload.get(key) in expected}
    if stage.get("require_all_gates"):
        checks["all_gates_pass"] = bool(payload.get("gates")) and all(
            payload["gates"].values()
        )
    for flag in stage.get("required_true_flags", []):
        checks[f"{flag}_true"] = payload.get(flag) is True
    for flag in stage.get("required_false_flags", []):
        checks[f"{flag}_false"] = payload.get(flag) is False
    for required_key in stage.get("required_keys", []):
        checks[f"{required_key}_present"] = required_key in payload
    return {
        "state": "passed" if all(checks.values()) else "failed_closed",
        "artifact": stage["artifact"],
        "artifact_sha256": sha256(path),
        "observed_value": payload.get(key),
        "checks": checks,
    }


def snapshot(config: dict[str, Any], active_pids: list[int] | None = None) -> dict[str, Any]:
    validate_static(config)
    pids = running_base_processes() if active_pids is None else active_pids
    stages = [
        {"id": stage["id"], **artifact_state(stage)} for stage in config["stages"]
    ]
    next_stage = next((row["id"] for row in stages if row["state"] != "passed"), None)
    return {
        "status": "terminal_chain_complete" if next_stage is None else "terminal_chain_pending",
        "active_base_pids": pids,
        "stages": stages,
        "next_stage": next_stage,
        "lensing_halo_action_gravity_or_holdout_payload_opened": False,
    }


def run_stage(stage: dict[str, Any]) -> None:
    command = [sys.executable, str(ROOT / stage["command"]["script"])]
    command.extend(str(value) for value in stage["command"].get("args", []))
    completed = subprocess.run(command, cwd=ROOT, check=False)
    if completed.returncode != 0:
        raise RuntimeError(
            f"V19BR stage {stage['id']} exited {completed.returncode}; chain stopped"
        )


def execute(config: dict[str, Any]) -> dict[str, Any]:
    validate_static(config)
    if not config["authorization"]["execute_only_after_base_process_exits"]:
        raise RuntimeError("V19BR execution authorization changed")
    initial = snapshot(config)
    if initial["active_base_pids"]:
        raise RuntimeError(
            f"V19BR refuses while base PIDs remain: {initial['active_base_pids']}"
        )
    executed: list[str] = []
    skipped: list[str] = []
    for stage in config["stages"]:
        before = artifact_state(stage)
        if before["state"] == "passed":
            skipped.append(stage["id"])
            continue
        if before["state"] == "failed_closed":
            raise RuntimeError(
                f"V19BR stage {stage['id']} has terminal or corrupt failure evidence"
            )
        run_stage(stage)
        after = artifact_state(stage)
        if after["state"] != "passed":
            raise RuntimeError(
                f"V19BR stage {stage['id']} did not produce its required pass"
            )
        executed.append(stage["id"])
    final = snapshot(config, active_pids=[])
    if final["status"] != "terminal_chain_complete":
        raise RuntimeError("V19BR terminal chain ended without every stage passing")
    return {
        **final,
        "executed_stages": executed,
        "already_passing_stages": skipped,
        "source_decision": load_json(
            ROOT / config["stages"][-1]["artifact"]
        ).get("aggregate_decision"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--status-only", action="store_true")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.status_only == args.execute:
        raise SystemExit("select exactly one of --status-only or --execute")
    config_path = args.config.resolve()
    config = load_json(config_path)
    if args.status_only:
        print(json.dumps(snapshot(config), indent=2))
        return
    try:
        result = execute(config)
    except Exception as exc:  # noqa: BLE001 - persist the terminal stop reason
        result = {
            "status": "terminal_chain_failed_closed",
            "exception": f"{type(exc).__name__}: {exc}",
            "lensing_halo_action_gravity_or_holdout_payload_opened": False,
        }
    report = {
        "protocol_version": config["protocol_version"],
        "generated_utc": datetime.now(UTC).isoformat(),
        "config_sha256": sha256(config_path),
        "runner_sha256": sha256(Path(__file__).resolve()),
        **result,
        "claim_boundary": config["claim_boundary"],
    }
    output = ROOT / config["outputs"]["terminal_report"]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(output)
    print(report["status"])
    if report["status"] != "terminal_chain_complete":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
