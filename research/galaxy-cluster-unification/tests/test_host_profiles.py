from __future__ import annotations

import numpy as np

from voidscreen.data import KPC_M
from voidscreen.host_profiles import (
    hernquist_local_density_from_total_mass,
    nfw_mass_function,
    nfw_overdensity_conversion,
    potential_chi_from_mass,
    prugniel_simien_local_density_from_enclosed_mass,
    sersic_deprojected_potential_factor,
    spherical_profile_potential_factor,
    truncated_nfw_potential_factor,
)
from voidscreen.unified import C_M_S, G_SI, M_SUN_KG


def test_hernquist_local_density_matches_analytic_profile() -> None:
    mass = 5.0e11
    radius = 20.0
    effective_radius = 30.0
    value = hernquist_local_density_from_total_mass(
        mass, radius, effective_radius
    )
    scale = 0.551 * effective_radius
    expected_msun_kpc3 = mass * scale / (
        2.0 * np.pi * radius * (radius + scale) ** 3
    )
    conversion = 1.98847e30 * 1.0e3 / (3.085677581491367e19 * 100.0) ** 3
    assert np.isclose(value, expected_msun_kpc3 * conversion, rtol=2.0e-5)


def test_uniform_sphere_potential_factor_has_known_limits() -> None:
    density = lambda radius: 1.0
    assert np.isclose(spherical_profile_potential_factor(density, 0.0), 1.5)
    assert np.isclose(spherical_profile_potential_factor(density, 1.0), 1.0)


def test_nfw_overdensity_conversion_preserves_both_definitions() -> None:
    radius_ratio, mass_ratio = nfw_overdensity_conversion(4.0)
    assert 0.0 < radius_ratio < 1.0
    assert np.isclose(mass_ratio, 2.5 * radius_ratio**3)
    expected = nfw_mass_function(4.0 * radius_ratio) / nfw_mass_function(4.0)
    assert np.isclose(mass_ratio, expected)


def test_truncated_nfw_factor_is_unity_at_outer_radius() -> None:
    assert np.isclose(truncated_nfw_potential_factor(3.0, 1.0), 1.0)
    assert truncated_nfw_potential_factor(3.0, 0.0) > 1.0


def test_sersic_completion_only_adds_exterior_potential() -> None:
    factors = sersic_deprojected_potential_factor(
        np.asarray([0.25, 1.0, 4.0]), np.asarray([1.0, 4.0, 8.0])
    )
    assert np.all(np.isfinite(factors))
    assert np.all(factors > 1.0)


def test_mass_potential_scale_matches_definition() -> None:
    expected = G_SI * 1e13 * M_SUN_KG / (500.0 * KPC_M * C_M_S**2)
    assert np.isclose(potential_chi_from_mass(1e13, 500.0), expected)


def test_prugniel_simien_density_recovers_supplied_enclosed_mass() -> None:
    from scipy.integrate import quad

    mass = 3.2e11
    radius = 8.0
    effective_radius = 10.0
    index = 4.0
    density_at_radius = float(
        prugniel_simien_local_density_from_enclosed_mass(
            mass, radius, effective_radius, index
        )
    )
    p = 1.0 - 0.6097 / index + 0.05463 / index**2
    b = 2.0 * index - 1.0 / 3.0 + 0.009876 / index

    def density_g_cm3(value_kpc: float) -> float:
        ratio = value_kpc / radius
        return density_at_radius * ratio ** (-p) * np.exp(
            -b
            * (
                (value_kpc / effective_radius) ** (1.0 / index)
                - (radius / effective_radius) ** (1.0 / index)
            )
        )

    recovered_g = quad(
        lambda value: 4.0
        * np.pi
        * density_g_cm3(value)
        * (value * KPC_M * 100.0) ** 2
        * (KPC_M * 100.0),
        0.0,
        radius,
    )[0]
    assert np.isclose(recovered_g / (M_SUN_KG * 1.0e3), mass, rtol=2.0e-8)


def test_prugniel_simien_density_vectorizes_and_rejects_bad_inputs() -> None:
    density = prugniel_simien_local_density_from_enclosed_mass(
        [1.0e10, 1.0e11], [2.0, 5.0], [3.0, 8.0], [1.0, 4.0]
    )
    assert density.shape == (2,)
    assert np.all(np.isfinite(density))
    assert np.all(density > 0.0)
    with np.testing.assert_raises(ValueError):
        prugniel_simien_local_density_from_enclosed_mass(1.0e11, 0.0, 8.0, 4.0)
