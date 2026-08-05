from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
V19N_CONFIG = ROOT / "configs" / "sigma_v19n_regional_response_workload.json"
V19N_RUNNER = ROOT / "scripts" / "run_sigma_v19n_regional_response_workload.py"
V19N_REPORT = ROOT / "results" / "sigma_v19n_regional_response_workload" / "report.json"
V19O_CONFIG = ROOT / "configs" / "sigma_v19o_fov_filtered_response_workload.json"
V19O_RUNNER = ROOT / "scripts" / "run_sigma_v19o_fov_filtered_response_workload.py"
V19O_REPORT = (
    ROOT / "results" / "sigma_v19o_fov_filtered_response_workload" / "report.json"
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_parents(config: dict) -> None:
    for key, value in config["parents"].items():
        if key.endswith("_sha256"):
            continue
        expected = config["parents"].get(f"{key}_sha256")
        if expected is not None:
            assert sha256(ROOT / value) == expected


def test_v19n_is_frozen_and_fails_closed_on_one_bullet_event() -> None:
    config = json.loads(V19N_CONFIG.read_text(encoding="utf-8"))
    report = json.loads(V19N_REPORT.read_text(encoding="utf-8"))
    validate_parents(config)
    assert config["integrity"]["event_row_read_at_freeze"] is False
    assert report["config_sha256"] == sha256(V19N_CONFIG)
    assert report["runner_sha256"] == sha256(V19N_RUNNER)
    assert report["status"] == "regional_response_workload_gate_failed"
    clusters = {row["cluster"]: row for row in report["clusters"]}
    assert clusters["BULLET"]["science_count_delta"] == 1.0
    assert clusters["ABELL2146"]["science_count_delta"] == 0.0
    assert clusters["BULLET"]["response_task_count"] == 3812
    assert clusters["ABELL2146"]["response_task_count"] == 1270
    assert report["response_extraction_authorized"] is False
    assert report["spectrum_or_response_constructed"] is False
    assert report["lensing_target_opened"] is False


def test_v19o_is_frozen_and_rejects_wrong_fov_equivalence() -> None:
    config = json.loads(V19O_CONFIG.read_text(encoding="utf-8"))
    report = json.loads(V19O_REPORT.read_text(encoding="utf-8"))
    validate_parents(config)
    assert config["integrity"]["fov_filtered_event_read_at_freeze"] is False
    assert report["config_sha256"] == sha256(V19O_CONFIG)
    assert report["runner_sha256"] == sha256(V19O_RUNNER)
    assert report["status"] == "fov_filtered_regional_response_workload_gate_failed"
    clusters = {row["cluster"]: row for row in report["clusters"]}
    assert clusters["BULLET"]["science_count_delta"] == 1.0
    assert clusters["ABELL2146"]["science_count_delta"] == -4255.0
    assert clusters["BULLET"]["tasks_removed_by_fov_filter"] == 0
    assert clusters["ABELL2146"]["tasks_removed_by_fov_filter"] == 0
    assert all(
        row["gates"]["all_translated_fov_hashes_finite_and_unique"]
        for row in clusters.values()
    )
    assert report["response_extraction_authorized"] is False
    assert report["spectrum_or_response_constructed"] is False
    assert report["temperature_density_mach_or_speed_fitted"] is False
    assert report["gravity_formula_or_parameter_changed"] is False
