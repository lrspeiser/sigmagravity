from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "p0665_tensor_aqual_3d_solver"


def report() -> dict:
    return json.loads((RESULTS / "report.json").read_text(encoding="utf-8"))


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_every_frozen_3d_solver_gate_passes():
    result = report()
    assert result["status"] == "pass"
    assert result["all_progression_gates_pass"] is True
    assert result["candidate_advanced_to_zero_slip_photon_deflection"] is True
    assert len(result["gate_results"]) == 15
    assert all(result["gate_results"].values())


def test_manufactured_accuracy_is_second_order():
    result = report()
    convergence = pd.read_csv(RESULTS / "manufactured_convergence.csv")
    assert result["metrics"]["manufactured_25_grid_relative_RMS_error"] < 0.0015
    assert result["metrics"]["manufactured_convergence_order"] > 2.0
    assert convergence.cells.tolist() == [9, 17, 25]
    assert convergence.relative_RMS_error.is_monotonic_decreasing


def test_symmetry_nonlinearity_and_density_lift_pass():
    metrics = report()["metrics"]
    assert metrics["rotation_covariance_relative_RMS_error"] < 9e-15
    assert metrics["direction_reversal_relative_RMS_error"] == 0.0
    assert metrics["sigma_zero_scalar_graph_AQUAL_relative_RMS_difference"] == 0.0
    assert metrics["nonlinear_normalized_residual_RMS"] < 8e-6
    assert metrics["nonlinear_iterations"] == 8
    assert metrics["surface_density_lift_column_mass_relative_error"] < 1.4e-16


def test_sources_blindness_and_figure_are_preserved():
    result = report()
    assert result["protocol_sha256"] == digest(
        ROOT / "configs/p0665_tensor_aqual_3d_solver.json"
    )
    assert result["source_sha256"] == digest(
        ROOT / "scripts/run_p0665_tensor_aqual_3d_solver.py"
    )
    assert result["solver_source_sha256"] == digest(
        ROOT / "src/voidscreen/tensor_aqual_3d.py"
    )
    assert result["spent_RXJ2129_lensing_outcomes_opened"] is False
    assert result["sealed_P0640_lensing_constraints_opened"] is False
    assert (RESULTS / "p0665_tensor_aqual_3d.png").stat().st_size > 20000
