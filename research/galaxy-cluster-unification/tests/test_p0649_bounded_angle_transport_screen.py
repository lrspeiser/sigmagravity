from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "p0649_bounded_angle_transport_screen"


def report() -> dict:
    return json.loads((RESULTS / "report.json").read_text(encoding="utf-8"))


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_primary_passes_every_frozen_gate_without_target_leakage():
    result = report()
    assert result["status"] == "pass"
    assert result["all_primary_gates_pass"] is True
    assert result["candidate_advanced"] is True
    assert len(result["gate_results"]) == 11
    assert all(result["gate_results"].values())
    assert result["sealed_P0633_kinematics_opened"] is False
    assert result["sealed_P0640_lensing_constraints_opened"] is False


def test_primary_has_no_unbounded_or_per_object_amplitude():
    result = report()
    assert result["primary_mode"] == "linear_chord_mix"
    assert result["coverage"]["unbounded_amplitude_parameters"] == 0
    assert result["coverage"]["per_object_gravity_parameters"] == 0
    assert result["coverage"]["registered_galaxies"] == 13
    assert result["coverage"]["registered_clusters"] == 4


def test_registered_domain_separation_survives_mass_sensitivity():
    metrics = report()["primary_metrics"]
    assert metrics["registered_galaxy_median_activation"] <= 0.10
    assert metrics["registered_cluster_median_activation"] >= 0.05
    assert metrics["registered_cluster_to_galaxy_ratio"] >= 4.0
    assert min(metrics["mass_sensitivity_cluster_to_galaxy_ratios"].values()) > 7.0


def test_bounded_radial_solar_and_covariance_controls():
    metrics = report()["primary_metrics"]
    assert metrics["radial_activation"] <= 1e-8
    assert metrics["large_over_small_ratio"] >= 4.0
    assert 0.0 <= metrics["activation_global_minimum"]
    assert metrics["activation_global_maximum"] <= 1.0
    assert metrics["solar_one_component_activation"] == 0.0
    assert metrics["rotation_covariance_relative_error"] <= 0.03
    assert metrics["translation_covariance_relative_error"] <= 0.03


def test_all_predeclared_modes_and_system_rows_are_saved():
    scores = pd.read_csv(RESULTS / "registered_map_mode_scores.csv")
    assert set(scores["mode"]) == {
        "quadratic_cancellation",
        "linear_chord_mix",
        "oriented_cross_mix",
    }
    assert len(scores) == 17 * 3 * 3
    assert np.isfinite(scores.select_dtypes(include=[float, int])).all().all()


def test_hashes_and_visual_artifact_are_reproducible():
    result = report()
    assert result["protocol_sha256"] == digest(
        ROOT / "configs/p0649_bounded_angle_transport_screen.json"
    )
    assert result["source_sha256"] == digest(
        ROOT / "scripts/run_p0649_bounded_angle_transport_screen.py"
    )
    assert (RESULTS / "bounded_angle_transport_screen.png").stat().st_size > 20000
