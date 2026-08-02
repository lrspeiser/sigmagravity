from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "p0660_exact_tensor_activation_audit"


def report() -> dict:
    return json.loads((RESULTS / "report.json").read_text(encoding="utf-8"))


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_frozen_result_fails_only_exact_domain_separation():
    result = report()
    assert result["status"] == "fail"
    assert result["all_progression_gates_pass"] is False
    assert result["candidate_advanced_to_real_map_field_solves"] is False
    failed = [name for name, passed in result["gate_results"].items() if not passed]
    assert failed == ["registered_domain_separation"]


def test_exact_transverse_activation_is_substantial_but_below_frozen_ratio():
    metrics = report()["metrics"]
    assert 0.0089 < metrics["registered_galaxy_nominal_median_sigma"] < 0.0090
    assert 0.0732 < metrics["registered_cluster_nominal_median_sigma"] < 0.0733
    assert 8.21 < metrics["registered_cluster_to_galaxy_nominal_median_sigma_ratio"] < 8.22
    assert min(metrics["mass_sensitivity_cluster_to_galaxy_ratios"].values()) > 7.1


def test_structure_symmetry_ellipticity_and_resolution_gates_pass():
    metrics = report()["metrics"]
    assert metrics["radial_cocentered_sigma_weighted_mean"] < 1e-14
    assert metrics["rotation_covariance_relative_error"] < 1e-14
    assert metrics["direction_reversal_tensor_relative_error"] == 0.0
    assert metrics["minimum_constitutive_eigenvalue_proxy"] > 1e-5
    assert metrics["maximum_domain_median_resolution_change_fraction"] < 0.35


def test_all_registered_maps_and_sensitivities_are_covered():
    result = report()
    scores = pd.read_csv(RESULTS / "registered_map_activation_scores.csv")
    assert result["coverage"]["registered_galaxies"] == 13
    assert result["coverage"]["registered_clusters"] == 4
    assert set(scores.scenario) == {"low", "nominal", "high"}
    assert set(scores.resolution) == {"primary", "check"}
    assert len(scores) == 102


def test_protocol_sources_blindness_and_figure_are_preserved():
    result = report()
    assert result["protocol_sha256"] == digest(
        ROOT / "configs/p0660_exact_tensor_activation_audit.json"
    )
    assert result["source_sha256"] == digest(
        ROOT / "scripts/run_p0660_exact_tensor_activation_audit.py"
    )
    assert result["activation_source_sha256"] == digest(
        ROOT / "src/voidscreen/tensor_activation.py"
    )
    assert result["sealed_P0633_kinematics_opened"] is False
    assert result["sealed_P0640_lensing_constraints_opened"] is False
    assert (RESULTS / "p0660_exact_tensor_activation.png").stat().st_size > 20000
