import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "sigma_v17k_luminal_aether_pressure_carrier.json"
RUNNER = ROOT / "scripts" / "audit_sigma_v17k_luminal_aether_carrier.py"
REPORT = ROOT / "results" / "sigma_v17k_luminal_aether_pressure_carrier" / "report.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_runner():
    spec = importlib.util.spec_from_file_location("sigma_v17k_carrier", RUNNER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_v17k_is_frozen_theory_only_and_parent_hashes_are_current() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))

    assert config["protocol_version"] == "SIGMA-V17K-LUMINAL-AETHER-PRESSURE-CARRIER-1.0.0"
    assert config["authorization"]["observational_data_opened"] is False
    assert config["authorization"]["empirical_fit_authorized"] is False
    assert config["authorization"]["holdout_authorized"] is False
    assert config["parent"]["sha256"] == _sha256(ROOT / config["parent"]["protocol"])
    assert config["parent"]["report_sha256"] == _sha256(ROOT / config["parent"]["report"])
    assert config["candidate_action"]["born_infeld_acceleration_density_present"] is False
    assert config["complexity"]["maximum_candidate_constants"] <= 5
    assert config["complexity"]["one_physical_metric"] is True
    assert config["complexity"]["per_object_gravity_constants"] == 0


@pytest.mark.parametrize("epsilon", [1e-12, 1e-9, 1e-7, 1e-5, 1e-3, 0.1])
def test_luminal_family_identities_hold_without_tuning(epsilon: float) -> None:
    runner = _load_runner()
    coefficients = runner.carrier_coefficients(epsilon)
    speeds = runner.mode_speeds(coefficients)
    ppn = runner.standard_ppn(coefficients)

    assert coefficients["c_13"] == pytest.approx(0.0, abs=1e-18)
    assert coefficients["c_14"] == pytest.approx(epsilon)
    assert coefficients["c_123"] == pytest.approx(coefficients["c_2"])
    assert np.asarray(list(speeds.values())) == pytest.approx(np.ones(3), abs=1e-12)
    assert ppn["gamma"] == 1.0
    assert ppn["beta"] == 1.0
    assert ppn["alpha_1"] == pytest.approx(-4.0 * epsilon, abs=1e-15)
    assert ppn["alpha_2"] == pytest.approx(0.0, abs=1e-12)


def test_fixed_epsilon_passes_declared_flat_and_weak_field_bounds() -> None:
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    fixed = report["fixed_carrier"]

    assert fixed["epsilon"] == 1e-7
    assert fixed["standard_aether_ppn_at_X_zero"]["alpha_1"] == pytest.approx(-4e-7)
    assert fixed["standard_aether_ppn_at_X_zero"]["alpha_2"] == pytest.approx(0.0)
    assert fixed["energy_sign_proxies"]["spin_1"] > 0.0
    assert fixed["energy_sign_proxies"]["spin_0"] > 0.0
    assert fixed["newtonian_coupling"]["fractional_shift"] < 1e-6
    assert fixed["gates"]["all_selection_gates_pass"] is True


def test_v17k_advances_only_to_tilted_constraint_gate() -> None:
    report = json.loads(REPORT.read_text(encoding="utf-8"))

    assert report["status"] == "passed_carrier_selection"
    assert report["family_identity_scan"]["symbolic_relations_verified_numerically"] is True
    assert report["gates"]["carrier_selection_pass"] is True
    assert report["gates"]["holdout_authorized"] is False
    assert report["limitations"]["full_derivative_metric_constraint_matrix_computed"] is False
    assert report["limitations"]["full_Sigma_PPN_solution_computed"] is False
    assert report["limitations"]["halo_radius_or_lensing_prediction_made"] is False
    assert report["decision"]["outcome"] == "advance_to_localized_tilted_constraint_gate_only"


def test_invalid_epsilon_is_rejected() -> None:
    runner = _load_runner()

    for value in (float("nan"), -1.0, 0.0, 0.5, 1.0):
        with pytest.raises(ValueError):
            runner.carrier_coefficients(value)
