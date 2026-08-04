import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "sigma_v17h_susceptibility_screened_pressure.json"
RUNNER = ROOT / "scripts" / "audit_sigma_v17h_susceptibility_screen.py"
REPORT = ROOT / "results" / "sigma_v17h_susceptibility_screen" / "report.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_runner():
    spec = importlib.util.spec_from_file_location("sigma_v17h_screen", RUNNER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_v17h_is_frozen_before_data_and_has_no_object_switch() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))

    assert config["protocol_version"] == (
        "SIGMA-V17H-SUSCEPTIBILITY-SCREENED-PRESSURE-1.0.0"
    )
    assert config["authorization"]["observational_data_opened"] is False
    assert config["authorization"]["empirical_coupling_fit_authorized"] is False
    assert config["authorization"]["holdout_authorized"] is False
    assert config["fixed_theory_choices"]["new_scale_beyond_existing_a_sigma"] is False
    assert config["gates"]["one_metric"] is True
    assert config["gates"]["object_specific_parameters"] == 0
    assert config["gates"]["lensing_only_multiplier"] is False
    assert config["complexity"]["maximum_candidate_constants"] <= 5


def test_v17h_parent_hashes_are_current() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    for path_key, hash_key in (
        ("pressure_metric_gate", "pressure_metric_gate_sha256"),
        ("root_scale_protocol", "root_scale_protocol_sha256"),
        ("reciprocal_metric_protocol", "reciprocal_metric_protocol_sha256"),
    ):
        path = ROOT / config["parents"][path_key]
        assert path.is_file()
        assert config["parents"][hash_key] == _sha256(path)


def test_susceptibility_is_the_square_root_of_normalized_action_slope() -> None:
    runner = _load_runner()
    z = np.geomspace(1e-12, 1e12, 401)
    step = 1e-5
    numerical_derivative = (
        runner.effective_aether_density(z * np.exp(step))
        - runner.effective_aether_density(z * np.exp(-step))
    ) / (2.0 * step * z)
    analytic = runner.susceptibility(z) ** 2 / 2.0

    assert np.allclose(numerical_derivative, analytic, rtol=2e-5, atol=1e-12)
    assert float(runner.susceptibility(0.0)) == 1.0


def test_reduced_acceleration_hessian_is_positive_and_maxwell_floored() -> None:
    runner = _load_runner()
    z = np.r_[0.0, np.geomspace(1e-20, 1e30, 2001)]
    tangential, radial = runner.acceleration_hessian_eigenvalues(z)

    assert np.all(tangential > 0.0)
    assert np.all(radial > 0.0)
    assert np.all(tangential >= radial)
    assert np.min(1.0 + tangential) >= 1.0
    assert np.min(1.0 + radial) >= 1.0


def test_fixed_response_meets_low_and_high_acceleration_selection_gates() -> None:
    runner = _load_runner()

    assert float(runner.homogeneous_response(0.1)) > 0.99
    assert float(runner.homogeneous_response(1e5)) <= 1e-5
    assert float(runner.homogeneous_response(0.0)) == 1.0


def test_uniform_solar_control_is_well_below_cassini_but_not_full_ppn() -> None:
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    solar = report["solar_uniform_sphere_control"]

    assert solar["pressure_weighted_source_chi"] == pytest.approx(
        8.820414196174122e-7, rel=1e-10
    )
    assert solar["effective_gamma_minus_one_upper_bound"] < 1e-9
    assert solar["fraction_of_Cassini_limit"] < 1e-4
    assert solar["is_full_PPN_calculation"] is False
    assert report["gates"]["full_PPN_pass"] is False
    assert report["gates"]["holdout_authorized"] is False


def test_v17h_advances_only_to_exact_variation() -> None:
    report = json.loads(REPORT.read_text(encoding="utf-8"))

    assert report["status"] == "passed_conditional_action_selection"
    assert report["gates"]["analytic_selection_pass"] is True
    assert report["gates"]["full_covariant_variation_pass"] is False
    assert report["decision"]["outcome"] == "advance_to_exact_variation_only"
