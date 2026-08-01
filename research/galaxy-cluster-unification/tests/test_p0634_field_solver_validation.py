from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "p0634_field_solver_validation"


def report() -> dict:
    return json.loads((RESULTS / "report.json").read_text(encoding="utf-8"))


def test_every_preregistered_solver_gate_passes():
    result = report()
    assert result["status"] == "pass"
    assert result["all_gates_pass"] is True
    assert len(result["gate_results"]) == 12
    assert all(result["gate_results"].values())
    assert result["target_observables_opened"] is False


def test_poisson_solver_is_second_order_and_resolves_plummer_force():
    metrics = report()["metrics"]["newtonian_poisson"]
    assert metrics["grid_convergence_order"] >= 1.8
    assert metrics["median_relative_error"] <= 0.01
    assert metrics["p95_relative_error"] <= 0.03
    assert metrics["normalized_residual_RMS"] <= 1e-5


def test_aqual_and_qumond_are_field_solutions_not_algebraic_placeholders():
    result = report()
    assert result["metrics"]["QUMOND"]["normalized_residual_RMS"] <= 1e-5
    assert result["metrics"]["AQUAL"]["normalized_residual_RMS"] <= 1e-5
    assert result["metrics"]["QUMOND"]["median_relative_error"] <= 0.01
    assert result["metrics"]["AQUAL"]["median_relative_error"] <= 0.02
    assert result["metrics"]["AQUAL"]["nonlinear_iterations"] > 1


def test_machine_readable_outputs_are_complete():
    metrics = pd.read_csv(RESULTS / "solver_metrics.csv")
    convergence = pd.read_csv(RESULTS / "poisson_grid_convergence.csv")
    assert set(metrics["law"]) == {"newtonian_poisson", "QUMOND", "AQUAL"}
    assert convergence["cells"].tolist() == [17, 25, 33, 49]
    assert (RESULTS / "poisson_grid_convergence.png").stat().st_size > 10000
