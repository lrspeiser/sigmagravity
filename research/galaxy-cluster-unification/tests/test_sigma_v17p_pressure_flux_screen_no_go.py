import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "sigma_v17p_pressure_flux_screen_no_go.json"
RUNNER = ROOT / "scripts" / "audit_sigma_v17p_pressure_flux_screen_no_go.py"
REPORT = ROOT / "results" / "sigma_v17p_pressure_flux_screen_no_go" / "report.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_runner():
    spec = importlib.util.spec_from_file_location("sigma_v17p_flux", RUNNER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_v17p_is_hash_locked_and_does_not_open_holdouts() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    for section in ("parent", "pressure_archetype", "retired_derivative_metric_class"):
        entry = config[section]
        assert entry["sha256"] == _sha256(ROOT / entry["protocol"])
        assert entry["report_sha256"] == _sha256(ROOT / entry["report"])
    diagnostic = config["spent_diagnostic_input"]
    assert diagnostic["sha256"] == _sha256(ROOT / diagnostic["path"])
    assert config["authorization"]["untouched_holdout_opened"] is False
    assert config["authorization"]["empirical_fit_authorized"] is False


@pytest.mark.parametrize("power", [2, 4, 6, 8])
def test_flux_solver_recovers_the_monotone_polynomial_equation(power: int) -> None:
    runner = _load_runner()
    for source in np.logspace(-8, 12, 41):
        field = runner.solve_flux(float(source), power)
        recovered = field * (1.0 + field**power)
        assert recovered == pytest.approx(source, rel=1e-11, abs=1e-12)


def test_v17p_analytic_lower_bound_exceeds_cassini() -> None:
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    theorem = report["analytic_no_go"]

    assert theorem["cluster_x_floor"] == pytest.approx(0.1)
    assert theorem["gamma_potential_lower_bound"] > theorem["cassini_limit"]
    assert theorem["minimum_excess_factor"] > 4.0
    assert theorem["analytic_gate_pass"] is False
    assert theorem["solar_crossing_radius_at_floor_au"] > 2000.0


def test_every_representative_curve_verifies_bound_and_fails_cassini() -> None:
    report = json.loads(REPORT.read_text(encoding="utf-8"))

    assert len(report["representative_curves"]) == 12
    assert all(row["potential_bound_verified"] for row in report["representative_curves"])
    assert not any(row["cassini_pass"] for row in report["representative_curves"])
    assert report["numerics"]["maximum_normalized_flux_residual"] <= 1e-10
    assert report["numerics"]["minimum_numeric_excess_factor"] > 20.0


def test_v17p_retires_only_the_conserved_flux_screen_class() -> None:
    report = json.loads(REPORT.read_text(encoding="utf-8"))

    assert report["status"] == "completed_pressure_flux_screen_no_go_audit"
    assert report["holdout_opened"] is False
    assert report["empirical_fit_performed"] is False
    assert report["selection"]["clash_floor_gate_pass"] is True
    assert report["selection"]["outcome"] == (
        "retire_monotone_shift_symmetric_pressure_flux_screen"
    )
    assert any("not all pressure-sourced" in item for item in report["claim_boundary"])
