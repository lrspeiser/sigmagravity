from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = ROOT / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import check_sigma_v19br_target_sealed_terminal_chain as checker
import run_sigma_v19br_target_sealed_terminal_chain as runner

CONFIG = ROOT / "configs" / "sigma_v19br_target_sealed_terminal_chain.json"
REPORT = ROOT / "results" / "sigma_v19br_target_sealed_terminal_chain" / "preflight_report.json"


def test_frozen_preflight_report_is_current_and_target_sealed() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    frozen = json.loads(REPORT.read_text(encoding="utf-8"))
    rebuilt = checker.execute(config, CONFIG)
    assert frozen == rebuilt
    assert all(frozen["gates"].values())
    assert not frozen["terminal_gas_stellar_or_source_result_opened"]
    assert not frozen["lensing_halo_action_gravity_or_holdout_payload_opened"]


def test_status_snapshot_is_read_only_and_stops_at_v19w5(monkeypatch) -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    monkeypatch.setattr(
        runner,
        "artifact_state",
        lambda stage: {"state": "pending", "artifact": stage["artifact"]},
    )
    status = runner.snapshot(config, active_pids=[101, 202])
    assert status["status"] == "terminal_chain_pending"
    assert status["active_base_pids"] == [101, 202]
    assert status["next_stage"] == "V19W5_RESPONSE_RECOVERY"
    assert all(stage["state"] == "pending" for stage in status["stages"])


def test_source_falsification_is_a_valid_terminal_decision(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "root"
    report_path = root / "results" / "source.json"
    report_path.parent.mkdir(parents=True)
    stage = {
        "artifact": "results/source.json",
        "failure_artifacts": ["results/source.json"],
        "state_key": "status",
        "expected_values": [
            "observed_source_invariant_gates_passed_action_derivation_authorized",
            "observed_source_invariant_gates_failed_no_action_authorized",
        ],
        "required_keys": ["aggregate_decision"],
        "required_false_flags": [
            "lensing_halo_action_or_gravity_payload_opened",
            "gravity_formula_or_parameter_changed",
        ],
    }
    report_path.write_text(
        json.dumps(
            {
                "status": "observed_source_invariant_gates_failed_no_action_authorized",
                "aggregate_decision": {"action_derivation_authorized": False},
                "lensing_halo_action_or_gravity_payload_opened": False,
                "gravity_formula_or_parameter_changed": False,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(runner, "ROOT", root)
    assert runner.artifact_state(stage)["state"] == "passed"
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    payload["status"] = "v19bq_observed_source_invariant_execution_failed_closed"
    report_path.write_text(json.dumps(payload), encoding="utf-8")
    assert runner.artifact_state(stage)["state"] == "failed_closed"


def test_execute_refuses_while_base_process_is_active(monkeypatch) -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    monkeypatch.setattr(runner, "running_base_processes", lambda: [582953])
    with pytest.raises(RuntimeError, match="refuses while base PIDs remain"):
        runner.execute(config)
