import numpy as np
import pandas as pd
import pytest

from voidscreen.path_completion import (
    MASS_PATH_MODELS,
    PATH_MODELS,
    mass_path_completion_profile,
    path_completion_profile,
    predict_path_completion_frame,
)


def test_every_path_law_is_bounded_between_local_and_maximum_gravity():
    radius = np.geomspace(0.1, 1000.0, 100)
    gbar = 1.0e-10 * np.power(radius, -1.0)
    parameters = {
        "distance_path": [0.1, 1.0, 1.0],
        "tidal_path": [0.1, 1.0, -30.0, 1.0],
        "matter_path": [0.1, 1.0, -10.0, 1.0],
        "hybrid_path": [0.1, 1.0, -30.0, -10.0],
    }
    for model in PATH_MODELS:
        result = path_completion_profile(radius, gbar, model, parameters[model])
        assert np.all(result["completion_fraction"] >= 0.1)
        assert np.all(result["completion_fraction"] <= 1.0)
        assert np.all(result["enhancement_relative_to_local_G"] >= 1.0)
        assert np.all(result["enhancement_relative_to_local_G"] <= 10.0)
        assert np.all(np.diff(result["completion_fraction"]) >= 0.0)


def test_distance_logistic_can_create_a_temporary_inverse_radius_force():
    radius = np.geomspace(0.01, 10.0, 1000)
    gbar = np.power(radius, -2.0) * 1.0e-12
    result = path_completion_profile(radius, gbar, "distance_path", [0.01, 0.0, 1.0])
    exponent = -np.gradient(
        np.log(result["predicted_acceleration_m_s2"]), np.log(radius)
    )
    assert np.min(np.abs(exponent - 1.0)) < 0.02
    assert exponent[-1] > 1.8


def test_strong_tidal_field_suppresses_path_recovery():
    radius = np.asarray([1.0, 10.0])
    weak = path_completion_profile(
        radius, [1.0e-14, 1.0e-14], "tidal_path", [0.1, 0.0, -30.0, 1.0]
    )
    strong = path_completion_profile(
        radius, [1.0e-7, 1.0e-7], "tidal_path", [0.1, 0.0, -30.0, 1.0]
    )
    assert weak["completion_fraction"][-1] > strong["completion_fraction"][-1]


def test_frame_prediction_preserves_input_order_and_system_boundaries():
    frame = pd.DataFrame(
        {
            "system": ["b", "a", "b", "a"],
            "radius_kpc": [2.0, 2.0, 1.0, 1.0],
            "gbar_m_s2": [1.0e-11, 1.0e-11, 2.0e-11, 2.0e-11],
        }
    )
    result = predict_path_completion_frame(frame, "distance_path", [0.2, 1.0, 1.0])
    assert result["completion_fraction"].shape == (4,)
    assert result["completion_fraction"][0] == pytest.approx(
        result["completion_fraction"][1]
    )
    assert result["completion_fraction"][2] == pytest.approx(
        result["completion_fraction"][3]
    )
    assert result["completion_fraction"][0] > result["completion_fraction"][2]


def test_mass_path_laws_favor_a_larger_source_but_remain_bounded():
    radius = np.geomspace(1.0, 100.0, 100)
    low_mass_gbar = 1.0e-12 * np.power(radius / 10.0, -2.0)
    high_mass_gbar = 1.0e-9 * np.power(radius / 10.0, -2.0)
    for model in MASS_PATH_MODELS:
        low = mass_path_completion_profile(
            radius, low_mass_gbar, model, [0.1, 1.0, 11.0, 1.0]
        )
        high = mass_path_completion_profile(
            radius, high_mass_gbar, model, [0.1, 1.0, 11.0, 1.0]
        )
        assert high["completion_fraction"][-1] > low["completion_fraction"][-1]
        assert np.all(low["completion_fraction"] >= 0.1)
        assert np.all(high["completion_fraction"] <= 1.0)
        assert np.all(np.diff(high["mass_history_solar"]) >= 0.0)
