from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "p0666_zero_slip_3d_photon_deflection"


def report() -> dict:
    return json.loads((RESULTS / "report.json").read_text(encoding="utf-8"))


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_frozen_result_fails_only_radial_activation_null():
    result = report()
    assert result["status"] == "fail"
    assert result["all_progression_gates_pass"] is False
    assert result["candidate_advanced_to_spent_RXJ2129_map_build"] is False
    failed = [name for name, passed in result["gate_results"].items() if not passed]
    assert failed == ["radial_activation_null"]


def test_zero_slip_point_mass_normalization_is_accurate():
    metrics = report()["metrics"]
    assert metrics["point_mass_GR_deflection_median_relative_error"] < 0.016
    assert metrics["point_mass_GR_deflection_p95_relative_error"] < 0.03
    assert metrics["deflection_rotation_covariance_relative_RMS_error"] < 5e-15
    assert metrics["normalized_deflection_curl_RMS"] < 2e-16


def test_offset_activation_is_large_but_radial_artifact_exceeds_null():
    metrics = report()["metrics"]
    assert metrics["offset_sigma_mass_weighted_mean"] > 0.068
    assert metrics["radial_sigma_mass_weighted_mean"] > 2e-5
    assert metrics["minimum_constitutive_eigenvalue_proxy"] > 0.0


def test_sources_blindness_and_figure_are_preserved():
    result = report()
    assert result["protocol_sha256"] == digest(
        ROOT / "configs/p0666_zero_slip_3d_photon_deflection.json"
    )
    assert result["source_sha256"] == digest(
        ROOT / "scripts/run_p0666_zero_slip_3d_photon_deflection.py"
    )
    assert result["metric_source_sha256"] == digest(
        ROOT / "src/voidscreen/metric_lensing_3d.py"
    )
    assert result["spent_RXJ2129_lensing_outcomes_opened"] is False
    assert result["sealed_P0640_lensing_constraints_opened"] is False
    assert (RESULTS / "p0666_zero_slip_photon_deflection.png").stat().st_size > 20000
