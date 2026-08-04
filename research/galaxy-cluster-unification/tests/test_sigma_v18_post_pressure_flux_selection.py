import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "sigma_v18_post_pressure_flux_selection.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_v18_selection_is_parent_locked_and_target_blind() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))

    for parent in config["parents"].values():
        assert parent["sha256"] == _sha256(ROOT / parent["path"])

    authorization = config["authorization"]
    assert authorization["observational_target_opened"] is False
    assert authorization["temperature_result_read"] is False
    assert authorization["thermal_stress_map_constructed"] is False
    assert authorization["empirical_coefficient_fit"] is False
    assert authorization["formula_selected"] is False


def test_v18_defines_an_effective_halo_without_dark_matter() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    variables = config["root_variables"]

    assert "nabla_i D^i=4 pi G rho_b" in variables["baryonic_flux"]
    assert "rho_Sigma,eff" in variables["effective_halo_density"]
    assert "not a dark medium" in variables["baryonic_flux"]
    assert config["covariant_completion_gates"]["one_metric"] is True
    assert config["covariant_completion_gates"]["object_specific_gravity_parameters"] == 0
    assert config["covariant_completion_gates"]["lensing_only_parameters"] == 0


def test_v18_branching_cannot_add_a_fourth_pressure_screen_or_private_length() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    branches = " ".join(config["target_blind_branching"])

    assert "do not put gas pressure or temperature into H" in branches
    assert "exactly one universal correlation length" in branches
    assert "do not extend the length grid" in branches
    assert config["covariant_completion_gates"]["no_direct_pressure_only_reciprocal_metric"] is True
    assert config["covariant_completion_gates"]["no_derivative_dependent_matter_metric"] is True


def test_v18_records_the_aqual_and_polarization_prior_art_boundary() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    parents = " ".join(config["prior_art_boundary"]["published_parents"])

    assert "AQUAL" in parents
    assert "QUMOND" in parents
    assert "Refracted Gravity" in parents
    assert "dipolar gravitational polarization" in parents
    assert config["prior_art_boundary"]["not_new"]
