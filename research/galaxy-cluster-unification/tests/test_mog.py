import numpy as np
import pytest

from voidscreen.mog import (
    chameleon_density_power,
    chameleon_metric_enhancement,
    environmental_mog_dynamic_enhancement,
    matched_mog_enhancement,
    mean_enclosed_density_kg_m3,
    mog_extra_acceleration_log_slope,
    spherical_vector_acceleration_m_s2,
    unscreened_ppn_gamma_minus_one,
    vector_light_dynamics_gamma_minus_one,
    yukawa_transition,
)
from voidscreen.unified import G_SI


def test_yukawa_transition_is_stable_and_has_correct_limits():
    x = np.asarray([1e-10, 1e-6, 1e-3, 1.0, 100.0])
    transition = yukawa_transition(x)
    assert transition[0] == pytest.approx(0.5e-20, rel=1e-10)
    assert transition[1] == pytest.approx(0.5e-12, rel=1e-6)
    assert np.all(np.diff(transition) > 0.0)
    assert transition[-1] == pytest.approx(1.0)


def test_matched_force_cancels_at_short_distance_and_attracts_at_large_distance():
    alpha = 9.0
    x = np.asarray([1e-7, 1.0, 100.0])
    enhancement = matched_mog_enhancement(x, alpha)
    assert enhancement[0] == pytest.approx(1.0, abs=5e-14)
    assert 1.0 < enhancement[1] < 1.0 + alpha
    assert enhancement[-1] == pytest.approx(1.0 + alpha)
    direct = environmental_mog_dynamic_enhancement(x, 1.0 + alpha, alpha)
    np.testing.assert_allclose(direct, enhancement, rtol=2e-15)


def test_chameleon_minimum_is_monotonic_and_parameterized_by_positive_n():
    assert chameleon_density_power(1.0) == pytest.approx(0.5)
    density = np.asarray([1e-24, 1e-22, 1e-20])
    enhancement = chameleon_metric_enhancement(
        density,
        reference_density_kg_m3=1e-22,
        z_reference=0.5,
        power=0.5,
    )
    assert np.all(np.diff(enhancement) < 0.0)
    with pytest.raises(ValueError):
        chameleon_density_power(0.0)


def test_mean_enclosed_density_inverts_spherical_acceleration():
    mass = 3.2e40
    radius = 7.1e20
    gbar = G_SI * mass / radius**2
    density = mean_enclosed_density_kg_m3(gbar, radius)
    expected = mass / (4.0 * np.pi * radius**3 / 3.0)
    assert density == pytest.approx(expected)


def test_unscreened_ppn_gamma_has_gr_limit_and_correct_sign():
    assert unscreened_ppn_gamma_minus_one(0.0) == pytest.approx(0.0)
    assert -1.0 < unscreened_ppn_gamma_minus_one(0.1) < 0.0


def test_long_range_vector_creates_light_dynamics_gamma_mismatch():
    alpha = 0.2
    metric = 1.0 + alpha
    assert vector_light_dynamics_gamma_minus_one(alpha, metric) == pytest.approx(
        2.0 * alpha
    )


def test_extra_acceleration_slope_crosses_one_over_r_only_in_transition():
    slope = mog_extra_acceleration_log_slope(np.asarray([1e-4, 1.0, 100.0]))
    assert slope[0] == pytest.approx(0.0, abs=2e-4)
    assert slope[0] > slope[1] > slope[2]
    assert slope[2] == pytest.approx(-2.0)


def test_exact_spherical_vector_kernel_recovers_point_source():
    mass = 2.3e41
    radius = 9.0e20
    range_m = 4.0e21
    alpha = 3.4
    x = radius / range_m
    expected = (
        alpha * G_SI * mass * (1.0 + x) * np.exp(-x) / radius**2
    )
    actual = spherical_vector_acceleration_m_s2(
        radius,
        np.asarray([0.0]),
        np.asarray([mass]),
        alpha=alpha,
        range_m=range_m,
    )
    assert actual[0] == pytest.approx(expected, rel=1e-13)


def test_massless_limit_restores_zero_force_inside_spherical_shell():
    shell_radius = 1e20
    acceleration = spherical_vector_acceleration_m_s2(
        shell_radius / 2.0,
        np.asarray([shell_radius]),
        np.asarray([1e40]),
        alpha=1.0,
        range_m=1e35,
    )
    exterior_scale = G_SI * 1e40 / shell_radius**2
    assert abs(acceleration[0]) / exterior_scale < 1e-25
