import numpy as np
import pytest

from voidscreen.phenomenology import (
    dimensionless_baryonic_potential,
    fixed_rar_enhancement,
    response_enhancement,
    simple_mond_enhancement,
)


def test_fixed_acceleration_references_are_newtonian_at_high_acceleration():
    gbar = np.array([1.0e-8, 1.0e-7])
    assert np.all(fixed_rar_enhancement(gbar, 1.2e-10) > 1.0)
    assert np.allclose(simple_mond_enhancement(gbar, 1.2e-10), 1.0, atol=0.02)


def test_dimensionless_potential_scales_with_radius():
    potential = dimensionless_baryonic_potential([1.0e-10, 1.0e-10], [10.0, 20.0])
    assert potential[1] == pytest.approx(2.0 * potential[0])


@pytest.mark.parametrize(
    ("model", "parameters"),
    [
        ("RG", [0.2, -25.0, 1.0]),
        ("RG_acceleration_threshold", [0.2, -25.0, 1.0, 0.5]),
        ("RG_potential_threshold", [0.2, -25.0, 1.0, 0.5]),
        ("RG_acceleration_floor", [0.2, -25.0, 1.0, 0.5]),
        ("RG_Sigma_additive", [0.2, -25.0, 1.0, 1.0]),
        ("RG_Sigma_quadrature", [0.2, -25.0, 1.0, 1.0]),
        ("RG_Sigma_product", [0.2, -25.0, 1.0, 1.0]),
        ("RG_density_gated_Sigma", [0.2, -25.0, 1.0, 1.0, 2.0]),
        ("RAR_RG_additive", [0.2, -25.0, 1.0]),
        ("RAR_RG_quadrature", [0.2, -25.0, 1.0]),
        ("RAR_RG_product", [0.2, -25.0, 1.0]),
        ("RAR_potential_gated_RG", [0.2, -25.0, 1.0, -5.0, 2.0]),
        ("RAR_fixed_potential_gated_RG", [0.2, -25.0, 1.0]),
        ("RAR_coherence_gated_RG", [0.2, -25.0, 1.0]),
        ("RAR_sharp_coherence_gated_RG", [0.2, -25.0, 1.0]),
    ],
)
def test_response_variants_are_finite_and_at_least_newtonian(model, parameters):
    result = response_enhancement(
        model,
        [1.0e-9, 1.0e-11],
        [1.0e-23, 1.0e-27],
        [5.0, 500.0],
        parameters,
    )
    assert np.all(np.isfinite(result))
    assert np.all(result >= 1.0)


def test_acceleration_threshold_reduces_to_rg_when_coupling_zero():
    args = ([1.0e-9, 1.0e-11], [1.0e-24, 1.0e-27], [5.0, 500.0])
    baseline = response_enhancement("RG", *args, [0.2, -25.0, 1.0])
    coupled = response_enhancement(
        "RG_acceleration_threshold", *args, [0.2, -25.0, 1.0, 0.0]
    )
    assert np.allclose(baseline, coupled)


def test_coherence_gate_recovers_rar_at_unity_and_additive_at_zero():
    gbar = np.array([1.0e-10, 1.0e-11])
    rho = np.array([1.0e-26, 1.0e-27])
    radius = np.array([10.0, 100.0])
    parameters = [0.2, -25.0, 1.0]
    high = response_enhancement(
        "RAR_coherence_gated_RG", gbar, rho, radius, parameters, coherence=1.0
    )
    low = response_enhancement(
        "RAR_coherence_gated_RG", gbar, rho, radius, parameters, coherence=0.0
    )
    rar = fixed_rar_enhancement(gbar, 1.2e-10)
    assert np.allclose(high, rar)
    assert np.all(low > high)


def test_sharp_coherence_gate_leaks_less_at_intermediate_coherence():
    args = ([1.0e-11], [1.0e-27], [20.0], [0.2, -25.0, 1.0])
    linear = response_enhancement(
        "RAR_coherence_gated_RG", *args, coherence=0.8
    )
    sharp = response_enhancement(
        "RAR_sharp_coherence_gated_RG", *args, coherence=0.8, coherence_gate_power=2.0
    )
    rar = fixed_rar_enhancement(args[0], 1.2e-10)
    assert np.all(sharp < linear)
    assert np.all(sharp > rar)
