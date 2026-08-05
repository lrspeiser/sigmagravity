from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "sigma_v19h_causal_observable_protocol.json"
RUNNER = ROOT / "scripts" / "run_sigma_v19h_source_maps.py"
MAP_REPORT = ROOT / "results" / "sigma_v19h_source_maps" / "report.json"
MEMBER_RUNNER = ROOT / "scripts" / "run_sigma_v19h_member_phase.py"
MEMBER_REPORT = ROOT / "results" / "sigma_v19h_member_phase" / "report.json"


def load() -> dict:
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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


def test_v19h_source_map_gate_passes_without_inspection_or_target_access() -> None:
    report = json.loads(MAP_REPORT.read_text(encoding="utf-8"))
    assert report["status"] == "both_clusters_passed_frozen_v19h_source_map_gate"
    assert report["failed_clusters"] == []
    assert report["registered_science_images_visually_inspected"] is False
    assert report["edge_search_run"] is False
    assert report["spectrum_or_response_constructed"] is False
    assert report["lensing_target_opened"] is False
    assert report["gravity_parameter_changed"] is False
    assert {row["cluster"] for row in report["clusters"]} == {
        "BULLET",
        "ABELL2146",
    }
    for row in report["clusters"]:
        assert all(row["gates"].values())
        assert len(row["observations"]) == 10
        assert row["frozen_snapshot"]["files"] == 9
        for product in row["frozen_snapshot"]["products"]:
            path = ROOT / product["relative_path"]
            assert path.stat().st_size == product["bytes"]
            assert sha256(path) == product["sha256"]


def test_v19h_source_map_runner_is_importable_without_ciao() -> None:
    scripts = str(ROOT / "scripts")
    sys.path.insert(0, scripts)
    try:
        spec = importlib.util.spec_from_file_location("sigma_v19h_map_test", RUNNER)
        assert spec and spec.loader
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        config, astrometry, cleaning = module.validate(
            module.DEFAULT_CONFIG,
            module.DEFAULT_ASTROMETRY,
            module.DEFAULT_CLEANING,
        )
        assert config["protocol_version"] == "SIGMA-V19H-CAUSAL-OBSERVABLES-1.0.0"
        assert astrometry["observation_count"] == cleaning["observation_count"] == 20
    finally:
        sys.path.remove(scripts)


def test_v19h_member_phase_gate_records_identifiability_failure() -> None:
    report = json.loads(MEMBER_REPORT.read_text(encoding="utf-8"))
    assert report["status"] == "frozen_v19h_member_phase_gate_failed"
    assert set(report["failed_clusters"]) == {"BULLET", "ABELL2146"}
    assert report["published_subcluster_labels_used_for_fit_or_selection"] is False
    assert report["lensing_target_opened"] is False
    assert report["gravity_parameter_changed"] is False
    for row in report["clusters"]:
        assert row["selected_components"] == 1
        assert row["gates"]["minimum_identifiable_merger_components"] is False
        assert row["gates"]["selected_fit_converged"] is True
        assert row["bootstrap"]["requested_draws"] == 2000
        assert row["bootstrap"]["accepted_draws"] == 2000
        assert row["bootstrap"]["failed_draws"] == 0
        path = ROOT / row["bootstrap"]["draws_file"]
        assert sha256(path) == row["bootstrap"]["draws_sha256"]


def test_v19h_member_mixture_recovers_a_synthetic_two_component_case() -> None:
    scripts = str(ROOT / "scripts")
    sys.path.insert(0, scripts)
    try:
        spec = importlib.util.spec_from_file_location(
            "sigma_v19h_member_test", MEMBER_RUNNER
        )
        assert spec and spec.loader
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        import numpy as np

        generator = np.random.default_rng(19008)
        observed = np.vstack(
            [
                generator.normal([-2.0, 0.0, 0.0], [0.3, 0.4, 0.5], (60, 3)),
                generator.normal([2.0, 0.0, 1.0], [0.4, 0.3, 0.5], (60, 3)),
            ]
        )
        errors = np.zeros_like(observed)
        errors[:, 2] = 0.1
        fits = [
            module.fit_mixture(observed, errors, components, 20 + components)
            for components in (1, 2, 3)
        ]
        assert len(module.select_fit(fits)["weights"]) == 2
    finally:
        sys.path.remove(scripts)
