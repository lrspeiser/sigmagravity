from __future__ import annotations

import numpy as np
import pytest

from voidscreen.sigma_v9a_bounded_alignment import (
    aether_rest_vector_kinetic_eigenvalues,
    alignment_fluxes,
    alignment_interaction_density,
    alignment_invariants,
    angle_required_for_amplification,
    centered_finite_difference_static_hessian,
    find_one_sided_rank_surface,
    maximum_perpendicular_amplification,
    scan_saturated_static_principal_symbol,
    static_principal_hessian,
    synthetic_multisource_misalignment,
)


def test_gram_invariant_is_nonnegative_and_rotation_invariant() -> None:
    scalar = np.array([0.7, -1.2, 0.3])
    aether = np.array([-0.4, 0.2, 1.1])
    angle = 0.73
    rotation = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    original = alignment_invariants(scalar, aether)
    rotated = alignment_invariants(rotation @ scalar, rotation @ aether)
    assert original.gram >= 0.0
    assert original.gram == pytest.approx(rotated.gram, rel=1.0e-14)
    assert original.sine_squared == pytest.approx(
        rotated.sine_squared, rel=1.0e-14
    )
    for variant in ("one_sided", "saturated"):
        assert alignment_interaction_density(
            scalar, aether, a_sigma=1.3, eta=0.4, variant=variant
        ) == pytest.approx(
            alignment_interaction_density(
                rotation @ scalar,
                rotation @ aether,
                a_sigma=1.3,
                eta=0.4,
                variant=variant,
            ),
            rel=1.0e-14,
        )


@pytest.mark.parametrize("variant", ["one_sided", "saturated"])
def test_aligned_fields_have_exactly_zero_density_and_first_variation(variant: str) -> None:
    scalar = np.array([1.0, -2.0, 0.5])
    aether = -3.0 * scalar
    density = alignment_interaction_density(
        scalar, aether, a_sigma=0.8, eta=2.0 / 3.0, variant=variant
    )
    fluxes = alignment_fluxes(
        scalar, aether, a_sigma=0.8, eta=2.0 / 3.0, variant=variant
    )
    assert density == pytest.approx(0.0, abs=2.0e-14)
    assert np.max(np.abs(fluxes.scalar_gradient_flux)) < 2.0e-13
    assert np.max(np.abs(fluxes.aether_acceleration_flux)) < 2.0e-13


def test_analytic_fluxes_match_independent_finite_differences() -> None:
    scalar = np.array([0.8, -0.2, 0.4])
    aether = np.array([-0.1, 0.7, 0.3])
    step = 1.0e-6
    for variant in ("one_sided", "saturated"):
        exact = alignment_fluxes(
            scalar, aether, a_sigma=1.1, eta=0.37, variant=variant
        )
        scalar_fd = np.empty(3)
        aether_fd = np.empty(3)
        for index in range(3):
            delta = np.zeros(3)
            delta[index] = step
            scalar_fd[index] = (
                alignment_interaction_density(
                    scalar + delta,
                    aether,
                    a_sigma=1.1,
                    eta=0.37,
                    variant=variant,
                )
                - alignment_interaction_density(
                    scalar - delta,
                    aether,
                    a_sigma=1.1,
                    eta=0.37,
                    variant=variant,
                )
            ) / (2.0 * step)
            aether_fd[index] = (
                alignment_interaction_density(
                    scalar,
                    aether + delta,
                    a_sigma=1.1,
                    eta=0.37,
                    variant=variant,
                )
                - alignment_interaction_density(
                    scalar,
                    aether - delta,
                    a_sigma=1.1,
                    eta=0.37,
                    variant=variant,
                )
            ) / (2.0 * step)
        assert np.allclose(exact.scalar_gradient_flux, scalar_fd, rtol=2.0e-9)
        assert np.allclose(exact.aether_acceleration_flux, aether_fd, rtol=2.0e-9)


def test_small_j_vector_kinetic_bound_is_exact() -> None:
    values = aether_rest_vector_kinetic_eigenvalues(
        1.0, k_b=1.0, eta=2.0 / 3.0
    )
    assert values["activation"] == pytest.approx(1.0)
    assert values["parallel"] == pytest.approx(1.0)
    assert values["perpendicular"] == pytest.approx(1.0 / 3.0)
    assert maximum_perpendicular_amplification(
        k_b=1.0, eta=2.0 / 3.0
    ) == pytest.approx(3.0)


def test_one_sided_term_has_finite_principal_rank_surface() -> None:
    surface = find_one_sided_rank_surface(eta=2.0 / 3.0, k_b=1.0)
    assert surface["Z_over_a_squared"] == pytest.approx(3.4984663731, rel=2.0e-9)
    assert surface["J_over_a"] == pytest.approx(1.87041877, rel=2.0e-7)
    assert surface["inertia_below"] != surface["inertia_above"]
    assert surface["minimum_absolute_eigenvalue"] < 1.0e-8
    assert float(surface["null_mode_aether_power"]) > 0.01
    assert float(surface["null_mode_scalar_power"]) > 0.01


def test_saturated_autodiff_hessian_matches_centered_finite_difference() -> None:
    state = np.array([0.2, -0.3, 0.1, 0.8, 0.25, -0.15])
    exact = static_principal_hessian(
        state, k_b=1.0, eta=2.0 / 3.0, variant="saturated"
    )
    finite = centered_finite_difference_static_hessian(
        state,
        k_b=1.0,
        eta=2.0 / 3.0,
        variant="saturated",
        step=2.0e-4,
    )
    assert np.linalg.norm(exact - finite) / np.linalg.norm(exact) < 2.0e-7


def test_saturated_repair_preserves_static_inertia_on_focused_scan() -> None:
    report = scan_saturated_static_principal_symbol(
        eta=2.0 / 3.0,
        k_b=1.0,
        y_values=np.geomspace(1.0e-3, 1.0e3, 7),
        z_values=np.concatenate(([0.0], np.geomspace(1.0e-3, 1.0e3, 7))),
        cosine_values=np.linspace(-1.0, 1.0, 5),
        random_samples=64,
        random_seed=41,
    )
    assert report["necessary_static_gate_passed"]
    assert report["inertia_changes"] == 0
    assert float(report["minimum_singular_value"]) > 0.1


def test_amplitude_bound_exposes_angular_requirement_and_spherical_null() -> None:
    full = angle_required_for_amplification(
        1.0 / 0.318, k_b=1.0, eta=2.0 / 3.0
    )
    partial = angle_required_for_amplification(
        1.0 + 0.75 * (1.0 / 0.318 - 1.0),
        k_b=1.0,
        eta=2.0 / 3.0,
    )
    assert not full["reachable"]
    assert partial["reachable"]
    assert float(partial["minimum_angle_degrees"]) > 70.0
    assert float(partial["minimum_angle_degrees"]) < 80.0


def test_synthetic_geometry_has_exact_single_source_control_and_local_two_source_activation() -> None:
    result = synthetic_multisource_misalignment(grid_size=81)
    assert result["single_source_maximum_sine_squared"] < 1.0e-12
    assert result["two_source_maximum_sine_squared"] > 0.8
    assert result["two_source_field_weighted_mean_sine_squared"] < 0.1
    assert result["two_source_field_weighted_fraction_above_half"] < 0.05


@pytest.mark.parametrize(
    ("function", "args", "kwargs"),
    [
        (alignment_invariants, (np.zeros(2), np.zeros(3)), {}),
        (
            alignment_interaction_density,
            (np.zeros(3), np.zeros(3)),
            {"a_sigma": 0.0, "eta": 1.0},
        ),
        (
            alignment_interaction_density,
            (np.zeros(3), np.zeros(3)),
            {"a_sigma": 1.0, "eta": 1.0, "variant": "bad"},
        ),
        (
            angle_required_for_amplification,
            (0.5,),
            {"k_b": 1.0, "eta": 0.5},
        ),
    ],
)
def test_invalid_inputs_are_rejected(function, args, kwargs) -> None:
    with pytest.raises(ValueError):
        function(*args, **kwargs)
