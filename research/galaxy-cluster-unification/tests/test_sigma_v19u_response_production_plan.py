from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "sigma_v19u_response_production_plan.json"
RUNNER = ROOT / "scripts" / "plan_sigma_v19u_response_production.py"
REPORT = ROOT / "results" / "sigma_v19u_response_production_plan" / "report.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_v19u_freezes_bounded_production_before_pilot_outcomes() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    for key, value in config["parents"].items():
        if key.endswith("_sha256"):
            continue
        expected = config["parents"].get(f"{key}_sha256")
        if expected is not None:
            assert sha256(ROOT / value) == expected
    assert config["workload"]["expected_task_count_total"] == 5082
    assert config["workload"]["batch_size"] == 64
    assert config["workload"]["maximum_concurrent_cells"] == 4
    assert config["throughput_pilot"]["expected_new_cell_count"] == 4
    assert config["throughput_pilot"]["full_production_is_authorized_at_freeze"] is False
    assert config["integrity"]["production_manifest_constructed_at_freeze"] is False
    assert config["integrity"]["pilot_cells_selected_at_freeze"] is False
    assert config["integrity"]["pilot_or_production_response_executed_at_freeze"] is False
    assert config["integrity"]["gravity_formula_or_parameter_changed"] is False


def test_v19u_plan_passes_and_only_authorizes_the_pilot() -> None:
    if not REPORT.exists():
        pytest.skip("V19U frozen planner has not been executed yet")
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    assert report["config_sha256"] == sha256(CONFIG)
    assert report["runner_sha256"] == sha256(RUNNER)
    assert report["status"] == "production_plan_passed_and_throughput_pilot_authorized"
    assert report["production_manifest"]["task_count"] == 5082
    assert report["production_manifest"]["batch_count"] == 80
    assert report["production_manifest"]["final_batch_size"] == 26
    assert len(report["pilot_cells"]) == 4
    assert all(report["gates"].values())
    assert report["throughput_pilot_authorized"] is True
    assert report["full_production_authorized"] is False
    assert report["response_or_spectrum_constructed"] is False
    assert report["gravity_formula_or_parameter_changed"] is False
