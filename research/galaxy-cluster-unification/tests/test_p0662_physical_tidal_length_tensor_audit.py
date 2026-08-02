from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "p0662_physical_tidal_length_tensor_audit"


def report() -> dict:
    return json.loads((RESULTS / "report.json").read_text(encoding="utf-8"))


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_frozen_result_fails_only_registered_resolution_stability():
    result = report()
    assert result["status"] == "fail"
    assert result["candidate_advanced_to_real_map_field_solves"] is False
    failed = [name for name, passed in result["gate_results"].items() if not passed]
    assert failed == ["resolution_stability"]


def test_physical_estimator_passes_controlled_scale_and_resolution_tests():
    metrics = report()["metrics"]
    assert metrics["physical_length_scale_covariance_relative_error"] < 1e-12
    assert metrics["synthetic_resolution_median_change_fraction"] < 0.021


def test_domain_separation_remains_large_and_mass_robust():
    metrics = report()["metrics"]
    assert metrics["registered_cluster_to_galaxy_nominal_median_sigma_ratio"] > 60.0
    assert metrics["minimum_mass_sensitivity_cluster_to_galaxy_ratio"] > 54.0


def test_real_galaxy_resolution_not_cluster_resolution_causes_failure():
    summaries = {row["domain"]: row for row in report()["resolution_summary"]}
    assert summaries["registered_cluster_baryons_only"][
        "median_absolute_fractional_change"
    ] < 0.12
    assert summaries["registered_galaxy_baryons_only"][
        "median_absolute_fractional_change"
    ] > 0.50


def test_sources_blindness_and_figure_are_preserved():
    result = report()
    assert result["protocol_sha256"] == digest(
        ROOT / "configs/p0662_physical_tidal_length_tensor_audit.json"
    )
    assert result["source_sha256"] == digest(
        ROOT / "scripts/run_p0662_physical_tidal_length_tensor_audit.py"
    )
    assert result["activation_source_sha256"] == digest(
        ROOT / "src/voidscreen/physical_tensor_activation.py"
    )
    assert result["sealed_P0633_kinematics_opened"] is False
    assert result["sealed_P0640_lensing_constraints_opened"] is False
    assert (RESULTS / "p0662_physical_tidal_length_tensor.png").stat().st_size > 20000
