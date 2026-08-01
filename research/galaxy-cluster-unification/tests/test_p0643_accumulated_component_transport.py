from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "p0643_accumulated_component_transport"


def report() -> dict:
    return json.loads((RESULTS / "report.json").read_text(encoding="utf-8"))


def test_all_primary_gates_pass_without_target_leakage():
    result = report()
    assert result["status"] == "pass"
    assert result["all_primary_gates_pass"] is True
    assert len(result["gate_results"]) == 10
    assert all(result["gate_results"].values())
    assert result["sealed_P0633_kinematics_opened"] is False
    assert result["sealed_P0640_lensing_constraints_opened"] is False


def test_primary_candidate_is_universal_and_fixed():
    candidate = report()["candidate"]
    assert report()["candidate_advanced"] is True
    assert candidate == {
        "base_geometry": "component_cancellation",
        "Lc_kpc": 10.0,
        "q": 1.0,
        "spatial_gravity_parameters": 0,
        "universal_new_constants": 1,
    }


def test_primary_candidate_separates_registered_domains():
    metrics = report()["primary_metrics"]
    assert metrics["registered_galaxy_median_activation"] <= 0.01
    assert metrics["registered_cluster_median_activation"] >= 0.01
    assert metrics["registered_cluster_to_galaxy_ratio"] >= 4.0
    assert min(metrics["mass_sensitivity_cluster_to_galaxy_ratios"].values()) > 1.0


def test_synthetic_path_accumulates_and_radial_case_is_null():
    metrics = report()["primary_metrics"]
    assert metrics["radial_activation"] <= 1e-8
    assert metrics["large_over_small_ratio"] >= 4.0
    assert metrics["rotation_covariance_relative_error"] <= 0.03
    assert metrics["translation_covariance_relative_error"] <= 0.03


def test_sensitivity_grid_contains_primary_and_does_not_select_it_posthoc():
    sensitivity = pd.read_csv(RESULTS / "sensitivity_summary.csv")
    assert len(sensitivity) == 12
    primary = sensitivity[np.isclose(sensitivity.Lc_kpc, 10.0) & np.isclose(sensitivity.q, 1.0)]
    assert len(primary) == 1
    assert np.isclose(
        primary.iloc[0].cluster_to_galaxy_ratio,
        report()["primary_metrics"]["registered_cluster_to_galaxy_ratio"],
    )


def test_registered_map_coverage_and_outputs():
    coverage = report()["coverage"]
    assert coverage["registered_galaxies"] == 13
    assert coverage["registered_clusters"] == 4
    assert coverage["per_object_gravity_parameters"] == 0
    scores = pd.read_csv(RESULTS / "registered_map_accumulation_scores.csv")
    assert len(scores) == 17 * 3 * 12
    assert np.isfinite(scores.select_dtypes(include=[float, int])).all().all()
    assert (RESULTS / "accumulated_transport_screen.png").stat().st_size > 20000
