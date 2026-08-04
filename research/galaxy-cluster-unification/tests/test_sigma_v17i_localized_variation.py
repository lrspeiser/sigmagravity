import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "sigma_v17i_localized_variation.json"
RUNNER = ROOT / "scripts" / "audit_sigma_v17i_localized_variation.py"
REPORT = ROOT / "results" / "sigma_v17i_localized_variation" / "report.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_runner():
    spec = importlib.util.spec_from_file_location("sigma_v17i_variation", RUNNER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_v17i_freeze_and_parent_hash() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))

    assert config["protocol_version"] == "SIGMA-V17I-LOCALIZED-VARIATION-1.0.1"
    assert config["authorization"]["observational_data_opened"] is False
    assert config["authorization"]["empirical_fit_authorized"] is False
    assert config["authorization"]["holdout_authorized"] is False
    parent = ROOT / config["parent"]["protocol"]
    assert config["parent"]["sha256"] == _sha256(parent)
    assert config["complexity"]["new_physical_constants_added_by_localization"] == 0


def test_q_derivative_and_pressure_source_are_exact() -> None:
    runner = _load_runner()

    assert runner.perfect_fluid_source(100.0, 0.0) == pytest.approx(0.0, abs=1e-13)
    assert runner.perfect_fluid_source(100.0, 2.5) == pytest.approx(7.5, abs=1e-13)
    assert runner.perfect_fluid_source(1.0e8, 1e-4) == pytest.approx(3e-4, abs=1e-8)


def test_physical_metric_is_lorentzian_in_weak_field_fixture() -> None:
    runner = _load_runner()
    metric = np.diag([-1.0, 1.0, 1.0, 1.0])
    physical, state = runner.physical_metric(
        metric,
        np.array([-1.0, 0.0, 0.0, 0.0]),
        np.array([0.0, 0.5, -0.2, 0.1]),
        1e-5,
        100.0,
        1.0,
    )

    eigenvalues = np.linalg.eigvalsh(physical)
    assert np.count_nonzero(eigenvalues < 0.0) == 1
    assert np.count_nonzero(eigenvalues > 0.0) == 3
    assert state["Z"] > 0.0
    assert 0.0 < state["chi"] < 1.0


def test_executable_variation_report_passes_only_localized_gate() -> None:
    report = json.loads(REPORT.read_text(encoding="utf-8"))

    assert report["status"] == "passed_localized_variation"
    assert report["variation_finite_difference"]["maximum_normalized_error"] <= 2e-6
    assert report["perfect_fluid_source"]["maximum_absolute_error"] <= 1e-12
    assert report["localized_equation_system"]["maximum_Euler_differential_order"] == 2
    assert report["localized_equation_system"]["off_shell_diffeomorphism_identity_derived"]
    assert report["gates"]["localized_variation_pass"] is True
    assert report["gates"]["full_Hamiltonian_health_pass"] is False
    assert report["gates"]["full_causality_pass"] is False
    assert report["gates"]["holdout_authorized"] is False
    assert report["decision"]["outcome"] == "advance_to_Dirac_and_principal_symbol_only"
