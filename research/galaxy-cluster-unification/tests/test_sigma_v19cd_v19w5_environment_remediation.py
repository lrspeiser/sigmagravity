from __future__ import annotations

import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = ROOT / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import check_sigma_v19cd_v19w5_environment_remediation as checker
import run_sigma_v19cd_v19w5_environment_remediation as runner


CONFIG = ROOT / "configs" / "sigma_v19cd_v19w5_environment_remediation.json"
REPORT = (
    ROOT
    / "results"
    / "sigma_v19cd_v19w5_environment_remediation"
    / "preflight_report.json"
)


def test_v19cd_preflight_passes_exact_environment_failure_boundary() -> None:
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    assert report["decision"] == "remediated_launch_authorized"
    assert all(report["gates"].values())
    assert report["failure_boundary"]["base_missing_cells"] == 384
    assert report["failure_boundary"]["failed_workspace"][
        "completed_cell_reports"
    ] == 0


def test_v19cd_uses_fresh_scratch_and_byte_identical_frozen_runner() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    command = runner.build_command(config, "v19w5")
    assert command[0] == config["environment"]["conda_executable"]
    assert "sigma-ciao-4.18" in command
    assert str(ROOT / config["remediation"]["v19w5_runner"]) in command
    assert config["remediation"]["fresh_recovery_scratch"] in command
    assert config["remediation"]["protected_base_scratch"] in command
    assert config["remediation"]["fresh_recovery_scratch"] != config[
        "failed_workspace"
    ]["path"]


def test_v19cd_resumes_unchanged_v19br_and_opens_no_target() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    command = runner.build_command(config, "v19br")
    assert str(ROOT / config["remediation"]["v19br_runner"]) in command
    assert command[-1] == "--execute"
    assert not config["authorization"][
        "open_lensing_halo_action_gravity_or_holdout"
    ]
    assert not config["authorization"]["derive_or_select_action"]
    assert not config["authorization"]["change_gravity_formula_or_parameter"]


def test_v19cd_committed_preflight_is_hash_bound_and_self_consistent() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    assert report["config_sha256"] == runner.sha256(CONFIG)
    assert set(report["parent_hashes"]) == set(config["parents"])
    assert all(report["environment"]["checks"].values())
    assert report["failure_boundary"]["failed_workspace"]["exists"]
    assert not report["lensing_halo_action_gravity_or_holdout_payload_opened"]
    assert not report["gravity_formula_or_parameter_changed"]
