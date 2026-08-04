import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "sigma_v17l_localized_luminal_pressure.json"
RUNNER = ROOT / "scripts" / "audit_sigma_v17l_localized_luminal_pressure.py"
REPORT = ROOT / "results" / "sigma_v17l_localized_luminal_pressure" / "report.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_runner():
    spec = importlib.util.spec_from_file_location("sigma_v17l_localized", RUNNER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_v17l_is_frozen_theory_only_and_all_dependencies_are_hash_locked() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))

    assert config["protocol_version"] == "SIGMA-V17L-LOCALIZED-LUMINAL-PRESSURE-1.0.0"
    assert config["authorization"]["observational_data_opened"] is False
    assert config["authorization"]["empirical_fit_authorized"] is False
    assert config["authorization"]["holdout_authorized"] is False
    assert config["parent"]["sha256"] == _sha256(ROOT / config["parent"]["protocol"])
    assert config["parent"]["report_sha256"] == _sha256(ROOT / config["parent"]["report"])
    shared = config["shared_matter_variation_kernel"]
    assert shared["sha256"] == _sha256(ROOT / shared["path"])
    assert config["localized_action"]["born_infeld_acceleration_density_present"] is False


def test_auxiliary_source_vanishes_continuously_for_vacuum_or_dust() -> None:
    runner = _load_runner()

    vacuum = runner.auxiliary_acceleration_source(
        0.0,
        3.0,
        [0.0, 0.1, 0.2, 0.3],
        alpha=2.0,
        chi_z=-0.25,
        a_sigma=1.0,
    )
    dust = runner.auxiliary_acceleration_source(
        0.1,
        0.0,
        [0.0, 0.1, 0.2, 0.3],
        alpha=2.0,
        chi_z=-0.25,
        a_sigma=1.0,
    )
    pressure = runner.auxiliary_acceleration_source(
        0.1,
        3.0,
        [0.0, 0.1, 0.2, 0.3],
        alpha=2.0,
        chi_z=-0.25,
        a_sigma=1.0,
    )

    assert vacuum == pytest.approx([0.0, 0.0, 0.0, 0.0])
    assert dust == pytest.approx([0.0, 0.0, 0.0, 0.0])
    assert pressure != pytest.approx([0.0, 0.0, 0.0, 0.0])


def test_localized_report_preserves_variation_and_pressure_identities() -> None:
    report = json.loads(REPORT.read_text(encoding="utf-8"))

    assert report["status"] == "passed_localized_variation"
    assert report["parent"]["protocol_hash_verified"] is True
    assert report["parent"]["report_hash_verified"] is True
    assert report["parent"]["epsilon_aether_loaded"] == pytest.approx(1e-7)
    assert report["shared_kernel"]["hash_verified"] is True
    assert report["matter_variation"]["maximum_normalized_error"] <= 2e-6
    assert report["perfect_fluid_source"]["maximum_absolute_error"] <= 1e-12
    assert report["perfect_fluid_source"]["dust_value"] == pytest.approx(0.0, abs=1e-13)
    assert report["localized_equations"]["born_infeld_force_removed"] is True
    assert report["localized_equations"]["maximum_Euler_differential_order"] == 2


def test_v17l_advances_only_to_active_pressure_kinetics() -> None:
    report = json.loads(REPORT.read_text(encoding="utf-8"))

    assert report["gates"]["localized_variation_pass"] is True
    assert report["gates"]["active_pressure_kinetic_pass"] is False
    assert report["gates"]["full_Sigma_PPN_pass"] is False
    assert report["gates"]["holdout_authorized"] is False
    assert report["decision"]["outcome"] == "advance_to_active_pressure_kinetic_gate_only"
