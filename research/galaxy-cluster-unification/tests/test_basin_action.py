from __future__ import annotations

import numpy as np
import pytest

from voidscreen.basin_action import (
    G_SI,
    KPC_M,
    M_SUN_KG,
    fit_effective_yukawa_from_extras,
    fractional_linear_point_source_scaling,
    metric_couplings_from_effective,
    point_mass_circular_speed_log_slope,
    point_mass_observable_accelerations_m_s2,
    point_mass_yukawa_acceleration_m_s2,
    positive_spectral_circular_speed_log_slope,
    reciprocal_dust_couplings,
)


def test_reciprocity_reduces_metric_couplings_to_effective_combinations() -> None:
    coupling = reciprocal_dust_couplings(alpha=-0.3, beta=-0.6)
    assert coupling.source_d == pytest.approx(0.3)
    assert coupling.dynamics_amplitude == pytest.approx(0.18)
    assert coupling.lensing_amplitude == pytest.approx(0.18)
    assert coupling.lensing_to_dynamics_ratio == pytest.approx(1.0)

    reconstructed = metric_couplings_from_effective(0.18, 1.0)
    flipped = metric_couplings_from_effective(0.18, 1.0, field_sign=-1)
    assert reconstructed.dynamics_amplitude == pytest.approx(0.18)
    assert reconstructed.lensing_to_dynamics_ratio == pytest.approx(1.0)
    assert flipped.alpha == pytest.approx(-reconstructed.alpha)
    assert flipped.beta == pytest.approx(-reconstructed.beta)


def test_dynamics_blind_metric_limit_is_also_dust_source_blind() -> None:
    coupling = reciprocal_dust_couplings(alpha=1.0, beta=1.0)
    assert coupling.source_d == 0.0
    assert coupling.dynamics_amplitude == 0.0
    assert coupling.lensing_amplitude == 0.0
    assert np.isnan(coupling.lensing_to_dynamics_ratio)


def test_point_mass_yukawa_formula_and_metric_ratio() -> None:
    radius = np.asarray([2.0, 4.0]) * KPC_M
    mass = 1.0e11 * M_SUN_KG
    range_m = 10.0 * KPC_M
    unit = point_mass_yukawa_acceleration_m_s2(radius, mass, range_m)
    x = radius / range_m
    expected = G_SI * mass * (1.0 + x) * np.exp(-x) / np.square(radius)
    np.testing.assert_allclose(unit, expected)

    baryonic, dynamics, lensing = point_mass_observable_accelerations_m_s2(
        radius,
        mass,
        range_m,
        dynamics_amplitude=1.7,
        lensing_to_dynamics_ratio=2.2,
    )
    np.testing.assert_allclose((lensing - baryonic) / (dynamics - baryonic), 2.2)


def test_attractive_yukawa_and_positive_spectral_mixtures_are_never_flat() -> None:
    x = np.geomspace(1.0e-4, 1.0e4, 1000)
    slope = point_mass_circular_speed_log_slope(x, dynamics_amplitude=1.0e6)
    assert np.all(slope <= -0.5)

    radius = np.geomspace(0.01, 1000.0, 1000) * KPC_M
    mixture_slope = positive_spectral_circular_speed_log_slope(
        radius,
        np.asarray([0.3, 3.0, 30.0, 300.0]) * KPC_M,
        np.asarray([0.2, 2.0, 5.0, 20.0]),
    )
    assert np.all(mixture_slope <= -0.5)


def test_fractional_linear_flat_kernel_has_wrong_btfr_mass_scaling() -> None:
    scaling = fractional_linear_point_source_scaling(1.5)
    assert scaling["flat_rotation_curve"]
    assert scaling["acceleration_radial_exponent"] == pytest.approx(-1.0)
    assert scaling["circular_speed_fourth_power_mass_exponent"] == pytest.approx(2.0)


def test_ideal_joint_data_recover_all_three_effective_parameters() -> None:
    radius = np.geomspace(0.2, 8.0, 80) * 24.0 * KPC_M
    mass = 8.0e10 * M_SUN_KG
    amplitude = 1.8
    range_m = 24.0 * KPC_M
    ratio = 2.4
    unit = point_mass_yukawa_acceleration_m_s2(radius, mass, range_m)
    result = fit_effective_yukawa_from_extras(
        radius,
        mass,
        amplitude * unit,
        ratio * amplitude * unit,
        initial_dynamics_amplitude=0.2,
        initial_range_m=3.0 * KPC_M,
        initial_lensing_ratio=0.4,
    )
    assert result.success
    assert result.dynamics_amplitude == pytest.approx(amplitude, rel=1.0e-8)
    assert result.range_m == pytest.approx(range_m, rel=1.0e-8)
    assert result.lensing_to_dynamics_ratio == pytest.approx(ratio, rel=1.0e-8)
    assert result.maximum_absolute_log_residual < 1.0e-9
    assert result.jacobian_condition_number < 100.0
