from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "p0664_registered_tensor_field_solve"


def report() -> dict:
    return json.loads((RESULTS / "report.json").read_text(encoding="utf-8"))


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_every_frozen_field_gate_passes():
    result = report()
    assert result["status"] == "pass"
    assert result["all_progression_gates_pass"] is True
    assert result["candidate_advanced_to_spent_lensing_topology"] is True
    assert len(result["gate_results"]) == 17
    assert all(result["gate_results"].values())


def test_all_registered_scalar_and_tensor_solves_converge():
    scores = pd.read_csv(RESULTS / "registered_field_scores.csv")
    assert len(scores) == 17
    assert scores.scalar_converged.all()
    assert scores.tensor_converged.all()
    assert scores.scalar_normalized_residual_RMS.max() < 1e-5
    assert scores.tensor_normalized_residual_RMS.max() < 1e-5


def test_tensor_preserves_galaxies_but_changes_every_cluster():
    metrics = report()["metrics"]
    assert metrics["registered_galaxy_median_tensor_effect"] < 0.0005
    assert metrics["registered_galaxy_maximum_tensor_effect"] < 0.0035
    assert metrics["registered_cluster_median_tensor_effect"] > 0.030
    assert metrics["registered_cluster_minimum_tensor_effect"] > 0.015
    assert metrics["cluster_to_galaxy_median_tensor_effect_ratio"] > 68.0


def test_recovery_symmetry_ellipticity_and_conservation_are_strong():
    metrics = report()["metrics"]
    assert metrics["constant_mu_sigma_zero_newtonian_recovery_relative_RMS"] < 1.2e-11
    assert metrics["rotation_covariance_tensor_effect_relative_error"] < 2.3e-12
    assert metrics["minimum_constitutive_eigenvalue"] > 8e-5
    assert metrics["maximum_normalized_acceleration_curl_RMS"] < 3e-16


def test_sources_blindness_and_figure_are_preserved():
    result = report()
    assert result["protocol_sha256"] == digest(
        ROOT / "configs/p0664_registered_tensor_field_solve.json"
    )
    assert result["source_sha256"] == digest(
        ROOT / "scripts/run_p0664_registered_tensor_field_solve.py"
    )
    assert result["field_source_sha256"] == digest(
        ROOT / "src/voidscreen/registered_tensor_field.py"
    )
    assert result["sealed_P0633_kinematics_opened"] is False
    assert result["sealed_P0640_lensing_constraints_opened"] is False
    assert (RESULTS / "p0664_registered_tensor_fields.png").stat().st_size > 20000
