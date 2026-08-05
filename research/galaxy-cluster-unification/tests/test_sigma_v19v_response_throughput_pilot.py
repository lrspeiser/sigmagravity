from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "sigma_v19v_response_throughput_pilot.json"
RUNNER = ROOT / "scripts" / "run_sigma_v19v_response_throughput_pilot.py"
REPORT = ROOT / "results" / "sigma_v19v_response_throughput_pilot" / "report.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_v19v_freezes_cross_cluster_concurrent_pilot_before_execution() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    for key, value in config["parents"].items():
        if key.endswith("_sha256"):
            continue
        expected = config["parents"].get(f"{key}_sha256")
        if expected is not None:
            assert sha256(ROOT / value) == expected
    assert len(config["pilot_cells"]) == 4
    assert {row["cluster"] for row in config["pilot_cells"]} == {"BULLET", "ABELL2146"}
    assert {row["quantile"] for row in config["pilot_cells"]} == {0.25, 0.75}
    assert config["execution"]["maximum_concurrent_cells"] == 4
    assert config["execution"]["maximum_attempts_per_cell"] == 2
    assert config["integrity"]["pilot_pha_arf_or_rmf_existed_at_freeze"] is False
    assert config["integrity"]["pilot_runtime_or_storage_observed_at_freeze"] is False
    assert config["integrity"]["gravity_formula_or_parameter_changed"] is False


def test_v19v_pilot_passes_before_full_production_is_authorized() -> None:
    if not REPORT.exists():
        pytest.skip("V19V frozen pilot has not been executed yet")
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    assert report["config_sha256"] == sha256(CONFIG)
    assert report["runner_sha256"] == sha256(RUNNER)
    assert report["status"] == "throughput_pilot_passed_and_full_response_production_authorized"
    assert len(report["cells"]) == 4
    assert all(report["gates"].values())
    assert report["observed_maximum_concurrency"] >= 2
    assert report["pilot_wall_seconds"] <= 600
    assert report["full_response_production_authorized"] is True
    assert report["additional_temperature_density_mach_or_speed_fitted"] is False
    assert report["gravity_formula_or_parameter_changed"] is False
    assert len(report["frozen_snapshots"]) == 4
    for cell in report["frozen_snapshots"]:
        assert set(cell["products"]) == {
            "source_pha",
            "background_pha",
            "arf",
            "rmf",
            "specextract_log",
            "cell_report",
        }
        for product in cell["products"].values():
            path = ROOT / product["relative_path"]
            assert path.stat().st_size == product["bytes"]
            assert sha256(path) == product["sha256"]
