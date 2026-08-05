from __future__ import annotations

import math

import numpy as np
import pytest

from voidscreen.sigma_source_invariants import (
    analytic_press_unexplained_fraction,
    anisotropic_stress,
    axial_angle_deg,
    axial_difference_deg,
    axial_interval_summary_deg,
    axial_orientation_deg,
    central_gradient,
    central_hessian,
    component_overlap,
    gradient_detection_sigma,
    projected_baroclinicity,
    projected_source_maps,
    quadratic_design,
    region_means,
    relative_current,
    robust_detection_sigma,
    symmetric_fractional_change,
    thermodynamic_gradient_stress,
)


def test_component_overlap_has_expected_limits_and_species_symmetry() -> None:
    gas = np.array([0.0, 1.0, 1.0, 3.0])
    stars = np.array([0.0, 0.0, 1.0, 1.0])
    result = component_overlap(gas, stars)
    assert result == pytest.approx([0.0, 0.0, 1.0, 0.75])
    assert component_overlap(stars, gas) == pytest.approx(result)


def test_relative_current_is_common_boost_invariant_and_rotation_covariant() -> None:
    gas = np.array([[3.0, 4.0], [2.0, -1.0]])
    stars = np.array([[0.0, 0.0], [-1.0, -1.0]])
    boost = np.array([17.0, -9.0])
    vector, norm = relative_current(gas, stars, 4.0, 3.0)
    boosted_vector, boosted_norm = relative_current(
        gas + boost, stars + boost, 4.0, 3.0
    )
    assert boosted_vector == pytest.approx(vector)
    assert boosted_norm == pytest.approx(norm)
    assert norm == pytest.approx([1.0, 0.36])

    rotation = np.array([[0.0, -1.0], [1.0, 0.0]])
    rotated_vector, rotated_norm = relative_current(
        gas @ rotation.T, stars @ rotation.T, 4.0, 3.0
    )
    assert rotated_vector == pytest.approx(vector @ rotation.T)
    assert rotated_norm == pytest.approx(norm)


def test_anisotropic_stress_is_symmetric_trace_free_and_rotation_invariant() -> None:
    stress = np.array([[4.0, 2.0], [0.0, 2.0]])
    trace_free, norm = anisotropic_stress(stress, 2.0)
    np.testing.assert_allclose(trace_free, [[1.0, 1.0], [1.0, -1.0]])
    assert np.trace(trace_free) == pytest.approx(0.0, abs=1e-14)
    assert norm == pytest.approx(1.0)

    angle = np.radians(31.0)
    rotation = np.array(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    )
    rotated, rotated_norm = anisotropic_stress(rotation @ stress @ rotation.T, 2.0)
    assert rotated == pytest.approx(rotation @ trace_free @ rotation.T)
    assert rotated_norm == pytest.approx(norm)


def test_gradient_stress_is_trace_free_and_recovers_a_manufactured_axis() -> None:
    _y, x = np.mgrid[-2:3, -2:3]
    density = np.exp(0.4 * x)
    entropy = np.exp(0.7 * x)
    tensor = thermodynamic_gradient_stress(density, entropy)
    assert np.max(np.abs(np.trace(tensor, axis1=-2, axis2=-1))) < 1e-14
    assert axial_orientation_deg(tensor) == pytest.approx(90.0)


def test_projected_baroclinicity_separates_parallel_and_orthogonal_gradients() -> None:
    y, x = np.mgrid[-2:3, -2:3]
    density = np.exp(0.3 * x)
    parallel_pressure = np.exp(0.8 * x)
    orthogonal_pressure = np.exp(0.8 * y)
    signed_parallel, squared_parallel = projected_baroclinicity(
        density, parallel_pressure
    )
    signed_orthogonal, squared_orthogonal = projected_baroclinicity(
        density, orthogonal_pressure
    )
    assert np.max(np.abs(signed_parallel)) < 1e-14
    assert np.max(np.abs(squared_parallel)) < 1e-14
    assert np.abs(signed_orthogonal) == pytest.approx(np.ones_like(signed_orthogonal))
    assert squared_orthogonal == pytest.approx(np.ones_like(squared_orthogonal))


def test_invalid_or_unidentifiable_inputs_fail_closed() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        component_overlap([-1.0], [2.0])
    with pytest.raises(ValueError, match="positive"):
        relative_current([[1.0, 0.0]], [[0.0, 0.0]], 0.0, 0.0)
    with pytest.raises(ValueError, match="isotropic"):
        axial_orientation_deg(np.eye(2))


def test_central_derivatives_recover_quadratic_polynomial() -> None:
    axis = np.arange(-4.0, 5.0)
    east, north = np.meshgrid(axis, axis)
    field = 3.0 * east**2 + 2.0 * east * north + 5.0 * north**2
    d_e, d_n = central_gradient(field, 1.0)
    d_ee, d_nn, d_en = central_hessian(field, 1.0)
    interior = np.s_[1:-1, 1:-1]
    assert np.allclose(d_e[interior], (6.0 * east + 2.0 * north)[interior])
    assert np.allclose(d_n[interior], (2.0 * east + 10.0 * north)[interior])
    assert np.allclose(d_ee[interior], 6.0)
    assert np.allclose(d_nn[interior], 10.0)
    assert np.allclose(d_en[interior], 2.0)


def test_i4_tensor_and_i5_baroclinicity_have_expected_limits() -> None:
    axis = np.arange(-5.0, 6.0)
    east, north = np.meshgrid(axis, axis)
    density = np.exp(0.03 * east)
    entropy = np.exp(0.04 * east)
    pressure_parallel = np.exp(0.05 * east)
    pressure_perpendicular = np.exp(0.05 * north)
    surface = np.exp(0.02 * east + 0.01 * north)
    parallel = projected_source_maps(
        density,
        entropy,
        pressure_parallel,
        surface,
        spacing_kpc=1.0,
        resolution_fwhm_kpc=10.0,
    )
    perpendicular = projected_source_maps(
        density,
        entropy,
        pressure_perpendicular,
        surface,
        spacing_kpc=1.0,
        resolution_fwhm_kpc=10.0,
    )
    center = (5, 5)
    assert math.isclose(parallel["i4_q_plus"][center], 0.06, rel_tol=1e-12)
    assert abs(parallel["i4_q_cross"][center]) < 1e-13
    assert parallel["i5_baroclinicity"][center] < 1e-24
    assert math.isclose(
        perpendicular["i5_baroclinicity"][center], 1.0, rel_tol=1e-12
    )


def test_i4_tensor_angle_rotates_as_a_spin_two_axis() -> None:
    angles = axial_angle_deg(np.asarray([1.0, 0.0, -1.0]), np.asarray([0.0, 1.0, 0.0]))
    assert np.allclose(angles, [0.0, 45.0, 90.0])
    assert np.allclose(axial_difference_deg([179.0, 5.0], [1.0, 175.0]), [2.0, 10.0])


def test_axial_interval_handles_wrap_at_180_degrees() -> None:
    summary = axial_interval_summary_deg(np.asarray([178.0, 179.0, 0.0, 1.0, 2.0]))
    assert summary["width_95_deg"] < 5.0
    assert min(summary["median_axis_deg"], 180.0 - summary["median_axis_deg"]) < 1e-8


def test_region_means_support_draw_stacks_and_radial_masks() -> None:
    labels = np.asarray([[0, 0, 1], [0, 1, 1]])
    draws = np.asarray(
        [
            [[1.0, 3.0, 10.0], [5.0, 20.0, 30.0]],
            [[2.0, 4.0, 12.0], [6.0, 22.0, 32.0]],
        ]
    )
    admitted = np.asarray([[True, True, True], [False, True, True]])
    result = region_means(draws, labels, [0, 1], radial_mask=admitted)
    assert np.allclose(result, [[2.0, 20.0], [3.0, 22.0]])


def test_quadratic_press_recovers_control_and_rejects_independent_structure() -> None:
    rng = np.random.default_rng(9)
    predictors = rng.normal(size=(180, 5))
    design = quadratic_design(predictors)
    controlled = design @ np.linspace(-0.5, 0.5, design.shape[1])
    controlled_score = analytic_press_unexplained_fraction(predictors, controlled)
    assert controlled_score["joint_unexplained_fraction"] < 1e-20
    independent = rng.normal(size=180)
    independent_score = analytic_press_unexplained_fraction(predictors, independent)
    assert independent_score["joint_unexplained_fraction"] > 0.8
    assert independent_score["coefficient_count"] == 21


def test_joint_tensor_press_fraction_is_rotation_invariant() -> None:
    rng = np.random.default_rng(42)
    predictors = rng.normal(size=(150, 5))
    response = rng.normal(size=(150, 2))
    angle = math.radians(37.0)
    rotation = np.asarray(
        [[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]]
    )
    first = analytic_press_unexplained_fraction(predictors, response)
    second = analytic_press_unexplained_fraction(predictors, response @ rotation.T)
    assert math.isclose(
        first["joint_unexplained_fraction"],
        second["joint_unexplained_fraction"],
        rel_tol=1e-12,
    )


def test_detection_and_fraction_helpers() -> None:
    assert robust_detection_sigma([9.0, 10.0, 11.0, 10.5, 9.5]) > 3.0
    assert np.allclose(symmetric_fractional_change([1.0, 0.0], [1.1, 0.0]), [2.0 / 21.0, 0.0])
    rng = np.random.default_rng(8)
    east = rng.normal(2.0, 0.1, 1000)
    north = rng.normal(0.0, 0.1, 1000)
    assert gradient_detection_sigma(east, north) > 10.0
    assert math.isinf(gradient_detection_sigma(np.ones(5), np.zeros(5)))
    assert gradient_detection_sigma(np.zeros(5), np.zeros(5)) == 0.0
