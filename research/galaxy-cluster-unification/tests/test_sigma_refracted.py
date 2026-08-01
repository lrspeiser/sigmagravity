from __future__ import annotations

import numpy as np

from voidscreen.axisymmetric_permittivity import logistic_permittivity
from voidscreen.sigma_refracted import (
    additive_susceptibility_enhancement,
    coherence_partitioned_spherical_enhancement,
    coherence_weight,
    naive_product_enhancement,
    nbp0_sharpness_from_rg,
    refracted_permittivity,
    sigma_enhancement,
    sigma_h,
)


PARAMETERS = {
    "response_amplitude": 1.17,
    "minimum_permittivity": 0.089,
    "critical_density": 10.0**-24.25,
    "rg_sharpness": 0.47,
}


def test_sigma_asymptotes_have_locked_slopes() -> None:
    g_dagger = 9.60e-11
    low = np.logspace(-8, -4, 30) * g_dagger
    high = np.logspace(4, 8, 30) * g_dagger
    low_slope = np.polyfit(np.log(low), np.log(sigma_h(low)), 1)[0]
    high_slope = np.polyfit(np.log(high), np.log(sigma_h(high)), 1)[0]
    assert abs(low_slope + 0.5) < 1.0e-3
    assert abs(high_slope + 1.5) < 1.0e-3


def test_rg_formula_matches_nbp0_logistic_at_zero_smoothing() -> None:
    density = np.logspace(-28, -21, 101)
    rg_q = 0.47
    expected = logistic_permittivity(
        density,
        minimum_permittivity=0.089,
        critical_density=10.0**-24.25,
        sharpness=nbp0_sharpness_from_rg(rg_q),
    )
    actual = refracted_permittivity(
        density,
        minimum_permittivity=0.089,
        critical_density=10.0**-24.25,
        rg_sharpness=rg_q,
    )
    np.testing.assert_allclose(actual, expected, rtol=1.0e-14, atol=1.0e-14)


def test_coherence_weight_is_bounded_monotonic_and_has_exact_endpoints() -> None:
    coherence = np.linspace(0.0, 1.0, 101)
    weight = coherence_weight(coherence)
    assert weight[0] == 0.0
    assert weight[-1] == 1.0
    assert np.all(np.diff(weight) >= 0.0)
    assert np.all((weight >= 0.0) & (weight <= 1.0))


def test_partitioned_model_recovers_both_parent_spherical_limits() -> None:
    acceleration = np.logspace(-13, -8, 20)
    density = np.logspace(-27, -22, 20)
    epsilon = refracted_permittivity(
        density,
        minimum_permittivity=PARAMETERS["minimum_permittivity"],
        critical_density=PARAMETERS["critical_density"],
        rg_sharpness=PARAMETERS["rg_sharpness"],
    )
    rg_endpoint = coherence_partitioned_spherical_enhancement(
        acceleration, density, 0.0, **PARAMETERS
    )
    sigma_endpoint = coherence_partitioned_spherical_enhancement(
        acceleration, density, 1.0, **PARAMETERS
    )
    np.testing.assert_allclose(rg_endpoint, 1.0 / epsilon)
    np.testing.assert_allclose(
        sigma_endpoint,
        sigma_enhancement(acceleration, PARAMETERS["response_amplitude"]),
    )


def test_partition_avoids_naive_double_boost_at_intermediate_coherence() -> None:
    acceleration = np.asarray([1.0e-12, 1.0e-11, 1.0e-10])
    density = np.asarray([1.0e-27, 1.0e-25, 1.0e-23])
    partitioned = coherence_partitioned_spherical_enhancement(
        acceleration, density, 0.5, **PARAMETERS
    )
    product = naive_product_enhancement(
        acceleration, density, **PARAMETERS
    )
    additive = additive_susceptibility_enhancement(
        acceleration, density, **PARAMETERS
    )
    assert np.all(partitioned < product)
    assert np.all(partitioned < additive)


def test_high_density_high_acceleration_limit_is_newtonian() -> None:
    enhancement = coherence_partitioned_spherical_enhancement(
        1.0e-3, 1.0e-10, 0.37, **PARAMETERS
    )
    assert abs(float(enhancement) - 1.0) < 1.0e-9
