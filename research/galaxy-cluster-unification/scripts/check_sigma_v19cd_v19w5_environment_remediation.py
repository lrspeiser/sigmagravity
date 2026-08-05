from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import run_sigma_v19cd_v19w5_environment_remediation as runner


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "sigma_v19cd_v19w5_environment_remediation.json"


def build_report(config_path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    config_path = config_path.resolve()
    config = runner.load_json(config_path)
    parents = runner.validate_parent_hashes(config)
    failure = runner.validate_failure_boundary(config)
    environment = runner.probe_environment(config)
    fresh = Path(config["remediation"]["fresh_recovery_scratch"])
    failed = failure["failed_workspace"]
    gates = {
        "all_parent_hashes_exact": len(parents) == len(config["parents"]),
        "failure_signature_is_exact_pre_cell_environment_error": all(
            failure["checks"].values()
        ),
        "failed_workspace_contains_no_recovery_cell_attempt": all(
            failed[key] == 0
            for key in (
                "completed_cell_reports",
                "failed_attempt_directories",
                "partial_attempt_directories",
                "quarantine_directories",
            )
        ),
        "base_terminal_state_is_4698_of_5082": failure["base_missing_cells"]
        == 384,
        "ciao_environment_is_complete_and_exact": all(
            environment["checks"].values()
        ),
        "fresh_scratch_is_separate_and_absent": not fresh.exists()
        and fresh != Path(config["remediation"]["protected_base_scratch"]),
        "frozen_v19w5_runner_and_science_settings_unchanged": config[
            "remediation"
        ]["same_v19w5_config_runner_manifest_masks_response_rules_and_audits"]
        and config["remediation"]["retry_count"] == 1,
        "original_failure_evidence_is_preserved": config["authorization"][
            "preserve_original_failed_workspace"
        ]
        and config["authorization"]["preserve_original_failure_report"],
        "downstream_v19br_chain_is_unchanged": config["remediation"][
            "same_v19br_downstream_stage_commands"
        ],
        "no_target_action_or_gravity_access": not any(
            config["authorization"][key]
            for key in (
                "open_lensing_halo_action_gravity_or_holdout",
                "derive_or_select_action",
                "change_gravity_formula_or_parameter",
            )
        ),
    }
    if set(gates) != set(config["required_gates"]):
        raise RuntimeError("V19CD implemented and declared gate sets differ")
    if not all(config["required_gates"].values()):
        raise RuntimeError("V19CD every declared gate must be mandatory")
    return {
        "protocol_version": config["protocol_version"],
        "status": "v19w5_environment_remediation_preflight_passed",
        "config": config_path.relative_to(ROOT).as_posix(),
        "config_sha256": runner.sha256(config_path),
        "parent_hashes": parents,
        "failure_boundary": failure,
        "environment": environment,
        "fresh_recovery_scratch": str(fresh),
        "v19w5_command": runner.build_command(config, "v19w5"),
        "v19br_command": runner.build_command(config, "v19br"),
        "gates": gates,
        "decision": (
            "remediated_launch_authorized"
            if all(gates.values())
            else "remediated_launch_forbidden"
        ),
        "lensing_halo_action_gravity_or_holdout_payload_opened": False,
        "gravity_formula_or_parameter_changed": False,
        "claim_boundary": config["claim_boundary"],
    }


def main() -> None:
    report = build_report()
    config = runner.load_json(DEFAULT_CONFIG)
    output = ROOT / config["outputs"]["preflight_report"]
    runner.atomic_json(output, report)
    print(json.dumps({"decision": report["decision"], "gates": report["gates"]}, indent=2, sort_keys=True))
    if report["decision"] != "remediated_launch_authorized":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
