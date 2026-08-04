import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "sigma_v17g_pressure_metric_gate.json"
RUNNER = ROOT / "scripts" / "audit_sigma_v17g_pressure_metric_gate.py"
REPORT = ROOT / "results" / "sigma_v17g_pressure_metric_gate" / "report.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_runner():
    spec = importlib.util.spec_from_file_location("sigma_v17g_pressure_gate", RUNNER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_v17g_is_frozen_prefit_and_uses_one_metric_coupling() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))

    assert config["protocol_version"] == "SIGMA-V17G-PRESSURE-METRIC-GATE-1.0.0"
    assert config["authorization"]["astronomical_target_opened"] is False
    assert config["authorization"]["empirical_coupling_fit_authorized"] is False
    assert config["integrity"]["galaxy_cluster_switch"] is False
    assert config["integrity"]["object_specific_halo_radius"] is False
    assert config["integrity"]["one_physical_metric"] is True
    assert config["gates"]["per_object_gravity_parameters_allowed"] == 0
    assert config["gates"]["lensing_only_multiplier_allowed"] is False


def test_v17g_parent_hashes_are_current() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    for path_key, hash_key in (
        ("root_scale_protocol", "root_scale_protocol_sha256"),
        ("reciprocal_metric_protocol", "reciprocal_metric_protocol_sha256"),
    ):
        path = ROOT / config["parents"][path_key]
        assert path.is_file()
        assert config["parents"][hash_key] == _sha256(path)


def test_pressure_compactness_is_monotonic_and_matches_declared_values() -> None:
    runner = _load_runner()
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    constants = config["physical_constants"]
    kwargs = {
        "joule_per_kev": constants["joule_per_keV"],
        "proton_mass_kg": constants["proton_mass_kg"],
        "speed_of_light_m_s": constants["speed_of_light_m_s"],
    }
    pi_6 = runner.cluster_pressure_compactness(6.0, 0.61, **kwargs)
    pi_17 = runner.cluster_pressure_compactness(17.0, 0.61, **kwargs)

    assert pi_17 > pi_6 > 0.0
    assert pi_17 == pytest.approx(8.910694289218627e-5, rel=1e-12)


def test_reciprocal_bound_is_alpha_independent_and_fails_cassini() -> None:
    runner = _load_runner()
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    report = runner.build_report(CONFIG, config)
    bounds = report["reciprocal_bounds"]

    assert bounds["minimum_alpha_for_cluster_gate"] == pytest.approx(
        105.93616037065179, rel=1e-12
    )
    assert bounds["solar_gamma_deviation_at_cluster_alpha"] == pytest.approx(
        0.023820441881440987, rel=1e-12
    )
    assert bounds["maximum_cluster_extra_Weyl_fraction_at_Cassini_alpha"] == (
        pytest.approx(0.0009655572350200517, rel=1e-12)
    )
    assert bounds["ratio_is_independent_of_alpha"] is True
    assert report["gates"]["advance"] is False
    assert report["decision"]["canonical_nonzero_length_pressure_metric_completion"] == (
        "reject_without_additional_screening"
    )


def test_nonzero_root_scale_range_does_not_screen_the_solar_system() -> None:
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    range_control = report["range_control"]

    assert range_control["minimum_Yukawa_transmission_over_solar_control"] > 0.99999999
    assert range_control["finite_range_can_supply_solar_screening"] is False
    assert report["status"] == "failed_pre_fit"
