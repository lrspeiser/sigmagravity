import importlib.util
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "sigma_v19bd_two_cluster_directional_source_uncertainty.json"
SCRIPT = ROOT / "scripts" / "run_sigma_v19bd_two_cluster_directional_source_uncertainty.py"
SPEC = importlib.util.spec_from_file_location("sigma_v19bd", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_weighted_morphology_is_scale_free():
    xy = np.asarray([[-2.0, 0.0], [1.0, -1.0], [2.0, 0.5], [0.0, 3.0]])
    luminosity = np.asarray([1.0, 2.0, 1.5, 0.75])
    beta = np.asarray([-0.004, -0.001, 0.002, 0.005])
    first = MODULE.compute_morphology(xy, luminosity, beta)
    second = MODULE.compute_morphology(7.0 * xy + np.asarray([100.0, -30.0]), luminosity, beta)
    assert np.isclose(first["normalized_second_offset"], second["normalized_second_offset"])
    assert np.isclose(
        first["normalized_current_separation"], second["normalized_current_separation"]
    )
    assert np.isclose(first["current_axis_alignment_cos2"], second["current_axis_alignment_cos2"])
    assert np.isclose(
        first["second_current_axis_alignment_abs_cos"],
        second["second_current_axis_alignment_abs_cos"],
    )
    assert np.isclose(
        second["luminosity_rms_radius_arcsec"], 7.0 * first["luminosity_rms_radius_arcsec"]
    )


def test_weighted_morphology_obeys_mathematical_bounds():
    xy = np.asarray([[-2.0, 0.0], [1.0, -1.0], [2.0, 0.5], [0.0, 3.0]])
    result = MODULE.compute_morphology(
        xy,
        np.asarray([1.0, 2.0, 1.5, 0.75]),
        np.asarray([-0.004, -0.001, 0.002, 0.005]),
    )
    assert result["luminosity_rms_radius_arcsec"] > 0.0
    assert 0.0 <= result["luminosity_axis_ellipticity"] <= 1.0
    assert result["normalized_second_offset"] > 0.0
    assert result["normalized_current_separation"] > 0.0
    assert -1.0 <= result["current_axis_alignment_cos2"] <= 1.0
    assert 0.0 <= result["second_current_axis_alignment_abs_cos"] <= 1.0


def test_protocol_is_source_only_and_scale_free():
    config = json.loads(CONFIG.read_text())
    assert config["status"].startswith("frozen_before")
    assert config["cluster_inputs"]["BULLET"]["expected_draws"] == 8192
    assert config["cluster_inputs"]["ABELL2146"]["expected_draws"] == 8192
    assert "R_L" in config["per_draw_statistics"]["normalized_current_separation"]
    assert not config["authorization"]["impute_missing_luminosity_or_transverse_velocity"]
    assert not config["authorization"]["compare_cross_filter_luminosity_amplitudes"]
    assert not config["authorization"]["apply_or_fit_long_wave_operator"]
    assert not config["authorization"]["read_lensing_halo_or_gas_response_payload"]
    assert not config["authorization"]["change_gravity_physics"]


def test_frozen_runner_hash_is_exact():
    config = json.loads(CONFIG.read_text())
    assert config["implementation"]["runner"] == str(SCRIPT.relative_to(ROOT)).replace("\\", "/")
    assert MODULE.sha256(SCRIPT) == config["implementation"]["runner_sha256"]
