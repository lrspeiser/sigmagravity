import math

import numpy as np
import pandas as pd

from voidscreen.phenomenology import fixed_rar_enhancement
from voidscreen.sparc_refit import effective_prediction, negative_log_posterior


def _frame(bulge_velocity=0.0):
    return pd.DataFrame(
        {
            "radius_catalog_kpc": [1.0, 2.0],
            "velocity_observed_catalog_kms": [40.0, 50.0],
            "velocity_error_catalog_kms": [2.0, 2.0],
            "gas_velocity_component_km_s": [10.0, 12.0],
            "disk_velocity_unit_ml_km_s": [30.0, 35.0],
            "bulge_velocity_unit_ml_km_s": [bulge_velocity, bulge_velocity],
            "distance_fractional_error": [0.1, 0.1],
            "inclination_catalog_deg": [60.0, 60.0],
            "inclination_error_deg": [3.0, 3.0],
            "disk_scale_kpc": [2.0, 2.0],
            "disk_surface_brightness": [100.0, 50.0],
            "bulge_surface_brightness": [0.0, 0.0],
            "HI_mass_billion_solar": [1.0, 1.0],
            "HI_radius_kpc": [8.0, 8.0],
            "disk_luminosity_fit_solar": [2.0e9, 2.0e9],
            "bulge_luminosity_fit_solar": [0.0, 0.0],
            "bulge_scale_fit_kpc": [math.nan, math.nan],
        }
    )


def _settings():
    return {
        "disk_mass_to_light_prior": 0.5,
        "bulge_mass_to_light_prior": 0.7,
        "log_mass_to_light_prior_sigma": 0.25,
        "velocity_error_floor_km_s": 2.0,
        "rar_acceleration_m_s2": 1.2e-10,
        "mond_acceleration_m_s2": 1.2e-10,
        "coherence_gate_power": 2.0,
        "hubble_km_s_mpc": 70.0,
        "nfw_v200_prior_km_s": 100.0,
        "nfw_log_v200_sigma": 1.0,
        "nfw_concentration_prior": 10.0,
        "nfw_log_concentration_sigma": 0.6,
        "screened_tail_parameter": 10.5,
        "screened_tail_reference_radius_kpc": 200.0,
        "screened_tail_a0_m_s2": 1.2e-10,
    }


def _geometry():
    return {
        "disk_hz_over_Rdisk": 0.2,
        "gas_RHI_divisor": 3.2,
        "gas_hz_over_Rgas": 0.1,
    }


def test_bulgeless_candidate_recovers_fixed_rar_exactly():
    frame = _frame()
    theta = np.zeros(4)
    candidate = effective_prediction(
        frame,
        theta,
        model="candidate",
        settings=_settings(),
        candidate_parameters=np.asarray([0.13, -23.8, 0.43]),
        density_geometry=_geometry(),
    )
    rar = effective_prediction(frame, theta, model="rar", settings=_settings())
    assert np.allclose(candidate["coherence"], 1.0)
    assert np.allclose(
        candidate["velocity_predicted_km_s"], rar["velocity_predicted_km_s"]
    )
    assert np.allclose(
        candidate["velocity_RAR_same_nuisance_km_s"],
        candidate["velocity_predicted_km_s"],
    )


def test_fixed_rar_prediction_matches_reference_enhancement():
    frame = _frame()
    result = effective_prediction(frame, np.zeros(4), model="rar", settings=_settings())
    enhancement = fixed_rar_enhancement(result["g_bar_m_s2"], 1.2e-10)
    expected = np.sqrt(
        result["g_bar_m_s2"]
        * enhancement
        * result["radius_adjusted_kpc"]
        * 3.085677581491367e19
    ) / 1000.0
    assert np.allclose(result["velocity_predicted_km_s"], expected)


def test_nfw_prediction_and_objective_are_finite():
    frame = _frame(bulge_velocity=15.0)
    theta = np.asarray([0.0, 0.0, 0.0, 0.0, math.log(100.0), math.log(10.0)])
    result = effective_prediction(frame, theta, model="nfw", settings=_settings())
    objective = negative_log_posterior(
        theta, frame, model="nfw", settings=_settings()
    )
    assert np.all(np.isfinite(result["velocity_predicted_km_s"]))
    assert np.all(result["velocity_predicted_km_s"] > 0.0)
    assert math.isfinite(objective)


def test_solar_screened_tail_matches_source_mass_law():
    frame = _frame()
    settings = _settings()
    result = effective_prediction(
        frame,
        np.zeros(4),
        model="solar_screened_isothermal",
        settings=settings,
    )
    source_mass = 1.33e9 + 0.5 * 2.0e9
    screen = 1.2e-10 / (1.2e-10 + result["g_bar_m_s2"])
    expected_extra_v2 = 10.5 * 4.300917270036279e-6 * source_mass / 200.0 * screen
    expected_v2 = (
        result["g_bar_m_s2"]
        * result["radius_adjusted_kpc"]
        * 3.085677581491367e19
        / 1.0e6
        + expected_extra_v2
    )
    assert np.allclose(result["source_baryonic_mass_solar"], source_mass)
    assert np.allclose(result["screened_tail_factor"], screen)
    assert np.allclose(np.square(result["velocity_predicted_km_s"]), expected_v2)
