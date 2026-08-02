from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "p0659_tensor_aqual_solver"


def report() -> dict:
    return json.loads((RESULTS / "report.json").read_text(encoding="utf-8"))


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_every_frozen_progression_gate_passes():
    result = report()
    assert result["status"] == "pass"
    assert result["all_progression_gates_pass"] is True
    assert result["candidate_advanced_to_outcome_blind_map_tests"] is True
    assert len(result["gate_results"]) == 14
    assert all(result["gate_results"].values())


def test_manufactured_accuracy_and_order_pass():
    result = report()
    metrics = result["metrics"]
    convergence = pd.read_csv(RESULTS / "manufactured_convergence.csv")
    assert metrics["manufactured_65_grid_relative_RMS_error"] < 0.00021
    assert metrics["manufactured_convergence_order"] > 2.0
    assert convergence.cells.tolist() == [17, 33, 65]
    assert convergence.relative_RMS_error.is_monotonic_decreasing


def test_symmetries_and_aligned_aqual_limit_are_numerically_exact():
    metrics = report()["metrics"]
    assert metrics["rotation_covariance_relative_RMS_error"] < 1e-14
    assert metrics["direction_reversal_relative_RMS_error"] == 0.0
    assert metrics["aligned_sigma_zero_relative_RMS_difference"] == 0.0


def test_nonlinear_solve_is_elliptic_and_converged():
    metrics = report()["metrics"]
    assert metrics["nonlinear_normalized_residual_RMS"] < 1e-5
    assert metrics["nonlinear_iterations"] == 8
    assert metrics["minimum_constitutive_eigenvalue"] > 0.0


def test_registered_lever_and_solar_proxy_pass_without_new_parameters():
    result = report()
    metrics = result["metrics"]
    coverage = result["coverage"]
    assert metrics["registered_cluster_to_galaxy_activation_ratio"] > 18.0
    assert metrics["solar_1au_constitutive_anisotropy"] < 2.1e-8
    assert coverage["new_universal_constants"] == 0
    assert coverage["per_object_gravity_parameters"] == 0


def test_blindness_hashes_and_figure_are_preserved():
    result = report()
    assert result["sealed_P0633_kinematics_opened"] is False
    assert result["sealed_P0640_lensing_constraints_opened"] is False
    assert result["protocol_sha256"] == digest(
        ROOT / "configs/p0659_tensor_aqual_solver.json"
    )
    assert result["source_sha256"] == digest(
        ROOT / "scripts/run_p0659_tensor_aqual_solver.py"
    )
    assert result["solver_source_sha256"] == digest(
        ROOT / "src/voidscreen/tensor_aqual.py"
    )
    assert (RESULTS / "tensor_aqual_solver.png").stat().st_size > 20000
