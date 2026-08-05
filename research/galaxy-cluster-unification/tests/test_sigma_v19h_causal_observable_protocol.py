from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "sigma_v19h_causal_observable_protocol.json"


def load() -> dict:
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def test_v19h_is_frozen_before_any_registered_image_or_target_access() -> None:
    config = load()
    assert config["integrity"] == {
        "registered_science_image_inspected_at_freeze": False,
        "merged_image_constructed_at_freeze": False,
        "edge_candidate_known_at_freeze": False,
        "spectrum_or_response_constructed_at_freeze": False,
        "temperature_density_mach_or_speed_fitted_at_freeze": False,
        "member_mixture_fitted_at_freeze": False,
        "projection_or_clock_drawn_at_freeze": False,
        "causal_source_constructed_at_freeze": False,
        "replacement_lensing_target_opened": False,
        "gravity_parameter_changed": False,
    }
    assert config["sample"]["lensing_targets_sealed"] is True


def test_v19h_parent_chain_is_content_addressed() -> None:
    scripts = str(ROOT / "scripts")
    sys.path.insert(0, scripts)
    try:
        import sigma_v19f_chandra_common as common

        common.validate_parent_hashes(load())
    finally:
        sys.path.remove(scripts)


def test_v19h_uses_identical_physical_measurement_rules() -> None:
    config = load()
    assert set(config["coordinates"]["clusters"]) == {"BULLET", "ABELL2146"}
    assert config["sample"]["same_rules_for_both_clusters"] is True
    assert config["projection_clock_ensemble"]["same_construction_for_both_clusters"]
    assert config["registered_inputs"]["all_twenty_observations_required"] is True
    assert config["registered_inputs"]["observation_selection_after_image_creation"] == (
        "forbidden"
    )


def test_v19h_front_is_automatic_and_uncertainty_bearing() -> None:
    config = load()
    edge = config["automated_front_estimator"]
    assert edge["candidate_scales_kpc"] == [8.0, 16.0, 32.0, 64.0]
    assert edge["ridge_linking"]["minimum_single_scale_significance_sigma"] == 5.0
    assert edge["uncertainty"]["poisson_parametric_bootstraps"] == 1000
    assert edge["shock_classification_after_spectra"][
        "published_front_coordinates_used"
    ] is False
    assert edge["gates"]["minimum_confirmed_shocks_per_cluster"] == 1


def test_v19h_thermodynamic_and_projection_gates_are_fixed() -> None:
    config = load()
    thermo = config["adaptive_thermodynamics"]
    assert thermo["spatial_binning"]["target_signal_to_noise"] == 40.0
    assert thermo["spatial_binning"]["minimum_valid_regions_per_cluster"] == 12
    assert thermo["shock_side_spectra"]["target_net_counts_per_side"] == 2500.0
    ensemble = config["projection_clock_ensemble"]
    assert ensemble["draws"] == 4096
    assert ensemble["draw_rules"]["primary_clock"] == "kinematic_clock"
    assert "published-clock branch" in ensemble["draw_rules"]["clock_robustness"]


def test_v19h_does_not_define_or_fit_gravity() -> None:
    config = load()
    assert config["automated_front_estimator"]["profile_fit"][
        "not_gravity_parameters"
    ] is True
    forbidden = config["advance_gate"]["not_authorized"]
    assert "fitting a halo size or amplitude" in forbidden
    assert "choosing a gravity formula from these source measurements" in forbidden
