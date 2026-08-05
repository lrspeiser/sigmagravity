from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "sigma_v19w_full_response_production.json"
RUNNER = ROOT / "scripts" / "run_sigma_v19w_full_response_production.py"
REPORT = ROOT / "results" / "sigma_v19w_full_response_production" / "report.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_v19w_freezes_exact_checkpointed_workload_before_production() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    for key, value in config["parents"].items():
        if key.endswith("_sha256"):
            continue
        expected = config["parents"].get(f"{key}_sha256")
        if expected is not None:
            assert sha256(ROOT / value) == expected
    assert config["workload"]["expected_task_count"] == 5082
    assert config["workload"]["expected_batch_count"] == 80
    assert config["workload"]["maximum_concurrent_cells"] == 4
    assert config["workload"]["maximum_total_attempts_per_cell"] == 2
    edge = config["known_positive_exposure_edge_case"]
    assert edge["bin_id"] == 24
    assert edge["zero_exposure_event_count"] == 1
    assert edge["bin_passed_v19m_region_admission"] is False
    assert edge["task_exists_in_v19u_manifest"] is False
    assert config["integrity"]["production_cell_started_at_freeze"] is False
    assert config["integrity"]["gravity_formula_or_parameter_changed"] is False


def test_v19w_requires_all_cells_before_regional_fitting() -> None:
    if not REPORT.exists():
        pytest.skip("V19W production has not completed yet")
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    assert report["config_sha256"] == sha256(CONFIG)
    assert report["runner_sha256"] == sha256(RUNNER)
    assert report["status"] == (
        "all_response_cells_passed_and_regional_spectral_fitting_authorized"
    )
    assert report["completed_cells"] == 5082
    assert report["product_index"]["rows"] == 5082
    assert all(report["gates"].values())
    assert report["regional_spectral_fitting_authorized"] is True
    assert report["temperature_density_mach_or_speed_fitted"] is False
    assert report["gravity_formula_or_parameter_changed"] is False
