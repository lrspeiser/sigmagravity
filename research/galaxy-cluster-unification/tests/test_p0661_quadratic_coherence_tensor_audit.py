from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "p0661_quadratic_coherence_tensor_audit"


def report() -> dict:
    return json.loads((RESULTS / "report.json").read_text(encoding="utf-8"))


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_frozen_result_fails_only_resolution_stability():
    result = report()
    assert result["status"] == "fail"
    assert result["all_progression_gates_pass"] is False
    assert result["candidate_advanced_to_real_map_field_solves"] is False
    failed = [name for name, passed in result["gate_results"].items() if not passed]
    assert failed == ["resolution_stability"]


def test_quadratic_kernel_has_the_frozen_analytic_behavior():
    metrics = report()["metrics"]
    assert metrics["coherence_power"] == 2.0
    assert 1.998 < metrics["short_path_log_slope"] < 2.0
    assert metrics["long_path_survival"] > 0.98


def test_domain_separation_is_large_and_mass_robust():
    metrics = report()["metrics"]
    assert metrics["registered_galaxy_nominal_median_sigma"] < 0.0013
    assert metrics["registered_cluster_nominal_median_sigma"] > 0.074
    assert metrics["registered_cluster_to_galaxy_nominal_median_sigma_ratio"] > 61.0
    assert metrics["minimum_mass_sensitivity_cluster_to_galaxy_ratio"] > 55.0


def test_resolution_failure_is_in_galaxies_not_clusters():
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
        ROOT / "configs/p0661_quadratic_coherence_tensor_audit.json"
    )
    assert result["source_sha256"] == digest(
        ROOT / "scripts/run_p0661_quadratic_coherence_tensor_audit.py"
    )
    assert result["sealed_P0633_kinematics_opened"] is False
    assert result["sealed_P0640_lensing_constraints_opened"] is False
    assert (RESULTS / "p0661_quadratic_coherence_tensor.png").stat().st_size > 20000
