import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "sigma_v17m_active_pressure_kinetic_gate.json"
RUNNER = ROOT / "scripts" / "audit_sigma_v17m_active_pressure_kinetics.py"
REPORT = ROOT / "results" / "sigma_v17m_active_pressure_kinetic_gate" / "report.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_runner():
    spec = importlib.util.spec_from_file_location("sigma_v17m_active_pressure", RUNNER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_v17m_is_target_blind_and_parent_is_hash_locked() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))

    assert config["protocol_version"] == "SIGMA-V17M-ACTIVE-PRESSURE-KINETIC-GATE-1.0.0"
    assert config["authorization"]["observational_data_opened"] is False
    assert config["authorization"]["empirical_fit_authorized"] is False
    assert config["authorization"]["holdout_authorized"] is False
    assert config["parent"]["sha256"] == _sha256(ROOT / config["parent"]["protocol"])
    assert config["parent"]["report_sha256"] == _sha256(ROOT / config["parent"]["report"])


def test_exact_canonical_matter_source_reduces_to_three_p() -> None:
    runner = _load_runner()

    for rho_hat, p_hat in ((1.0, 0.0), (1.0, 1e-5), (2.0, 0.4), (3.0, 3.0)):
        assert runner.reciprocal_source_hat(0.0, rho_hat, p_hat) == pytest.approx(
            3.0 * p_hat, abs=1e-12
        )


def test_exact_transverse_hessian_matches_finite_difference() -> None:
    runner = _load_runner()

    for q, rho_hat, w in ((1e-5, 0.01, 1.0), (1e-5, 1000.0, 1e-5), (1e-3, 1e-4, 1 / 3)):
        p_hat = w * rho_hat
        analytic = -0.5 * q * runner.reciprocal_source_hat(q, rho_hat, p_hat)
        observed = runner.finite_difference_hessian(q, rho_hat, p_hat)
        assert observed == pytest.approx(analytic, rel=1e-3, abs=1e-12)


def test_positive_pressure_creates_a_finite_spin_one_zero_surface() -> None:
    runner = _load_runner()
    epsilon = 1e-7

    threshold = runner.critical_density(1e-5, 1.0, epsilon)
    below = runner.transverse_sector(1e-5, threshold * 0.5, threshold * 0.5, epsilon)
    above = runner.transverse_sector(1e-5, threshold * 2.0, threshold * 2.0, epsilon)

    assert threshold > 0.0
    assert below["c_14_effective"] > 0.0
    assert above["c_14_effective"] < 0.0
    assert above["spin_1_speed_squared"] < 0.0
    assert above["matter_scalar_kinetic"] > 0.0


def test_v17m_report_retires_only_the_derivative_metric_coupling() -> None:
    report = json.loads(REPORT.read_text(encoding="utf-8"))

    assert report["status"] == "failed_active_pressure_kinetic_gate"
    assert report["observational_data_opened"] is False
    assert report["exact_matter_action"]["maximum_normalized_hessian_error"] <= 1e-3
    assert report["grid"]["unstable_backgrounds"] > 0
    assert report["zero_surfaces"]["finite_positive_threshold_count"] > 0
    assert report["gates"]["positive_matter_kinetic_pass"] is True
    assert report["gates"]["positive_c14_everywhere_pass"] is False
    assert report["gates"]["active_pressure_kinetic_pass"] is False
    assert report["decision"]["outcome"] == (
        "retire_acceleration_susceptibility_from_physical_metric"
    )
    assert report["decision"]["holdout_authorized"] is False
