import numpy as np
import pytest

from voidscreen.route_template import (
    baryonic_route_directions,
    conservative_explicit_direction_route_template,
    conservative_local_attractor_route_template,
    center_return_endpoints,
    conservative_directional_route_template,
    conservative_route_template,
    local_baryonic_attractor_endpoints,
    weighted_radius,
)


def test_weighted_radius_and_endpoints_are_deterministic():
    radius = np.array([1.0, 2.0, 4.0])
    weight = np.array([0.2, 0.3, 0.5])
    assert 1.0 <= weighted_radius(radius, weight, 0.5) <= 4.0
    positions = np.array([[-2.0, 0.0], [2.0, 0.0]])
    endpoint, center = center_return_endpoints(
        positions,
        [0.5, 0.5],
        return_scale=1.0,
        radius_exponent=0.0,
        reference_radius=1.0,
    )
    assert np.allclose(center, 0.0)
    assert np.allclose(endpoint, [[-1.0, 0.0], [1.0, 0.0]])


def test_route_template_preserves_total_weight():
    axis = np.linspace(-10.0, 10.0, 81)
    image, audit = conservative_route_template(
        axis,
        [[-2.0, 0.0], [2.0, 0.0]],
        [0.5, 0.5],
        routing_fraction=0.45,
        return_scale=1.0,
        radius_exponent=0.0,
        reference_radius=1.0,
        smoothing=0.5,
    )
    assert np.isclose(image.sum(), 1.0)
    assert audit["normalization_error"] < 1.0e-14


def test_single_centered_source_has_identical_local_and_routed_maps():
    axis = np.linspace(-10.0, 10.0, 81)
    _, audit = conservative_route_template(
        axis,
        [[0.0, 0.0]],
        [1.0],
        routing_fraction=0.8,
        return_scale=5.0,
        radius_exponent=-0.5,
        reference_radius=1.0,
        smoothing=0.5,
    )
    assert np.allclose(audit["local_map"], audit["routed_map"])


def test_local_direction_responds_to_neighbor_geometry_without_amplitude_leak():
    positions = np.array([[-3.0, 0.0], [0.0, 1.0], [2.0, 0.0]])
    weights = np.array([0.2, 0.3, 0.5])
    global_direction, _, _ = baryonic_route_directions(
        positions, weights, local_mix=0.0, softening=1.0
    )
    local_direction, _, _ = baryonic_route_directions(
        positions, weights, local_mix=1.0, softening=1.0
    )
    assert np.allclose(np.linalg.norm(global_direction, axis=1), 1.0)
    assert np.allclose(np.linalg.norm(local_direction, axis=1), 1.0)
    assert not np.allclose(global_direction, local_direction)


def test_symmetric_bend_preserves_weight_and_has_no_handedness():
    axis = np.linspace(-10.0, 10.0, 81)
    image, audit = conservative_directional_route_template(
        axis,
        [[-2.0, 0.0], [2.0, 0.0]],
        [0.5, 0.5],
        routing_fraction=0.45,
        return_scale=1.0,
        radius_exponent=0.0,
        reference_radius=1.0,
        smoothing=0.5,
        local_mix=0.0,
        softening=1.0,
        symmetric_bend_degrees=30.0,
    )
    assert np.isclose(image.sum(), 1.0)
    assert audit["normalization_error"] < 1.0e-14
    assert len(audit["endpoints"]) == 4
    assert np.isclose(np.sum(audit["endpoint_weights"]), 1.0)
    assert np.allclose(audit["endpoints"][:2, 1], -audit["endpoints"][2:, 1])


def test_distance_and_neighbor_weight_parameters_change_only_direction():
    positions = np.array([[-4.0, 0.0], [0.0, 2.0], [3.0, 0.0]])
    weights = np.array([0.2, 0.3, 0.5])
    shallow, _, _ = baryonic_route_directions(
        positions,
        weights,
        local_mix=1.0,
        softening=1.0,
        distance_power=1.0,
        neighbor_weights=[0.6, 0.3, 0.1],
    )
    steep, _, _ = baryonic_route_directions(
        positions,
        weights,
        local_mix=1.0,
        softening=1.0,
        distance_power=3.0,
        neighbor_weights=[0.1, 0.3, 0.6],
    )
    assert np.allclose(np.linalg.norm(shallow, axis=1), 1.0)
    assert np.allclose(np.linalg.norm(steep, axis=1), 1.0)
    assert not np.allclose(shallow, steep)


def test_explicit_direction_template_moves_weight_without_adding_it():
    axis = np.linspace(-10.0, 10.0, 81)
    image, audit = conservative_explicit_direction_route_template(
        axis,
        [[-2.0, 0.0], [2.0, 0.0]],
        [0.5, 0.5],
        [[1.0, 0.0], [-1.0, 0.0]],
        routing_fraction=0.4,
        return_scale=1.0,
        radius_exponent=0.0,
        reference_radius=1.0,
        smoothing=0.5,
    )
    assert np.isclose(image.sum(), 1.0)
    assert audit["normalization_error"] < 1e-14
    assert np.allclose(audit["endpoints"], [[-1.0, 0.0], [1.0, 0.0]])


def test_no_cross_travel_laws_never_overshoot_the_center():
    positions = np.array([[0.5, 0.0], [2.0, 0.0]])
    constant, _ = center_return_endpoints(
        positions,
        [0.5, 0.5],
        return_scale=1.0,
        radius_exponent=0.0,
        reference_radius=1.0,
        travel_mode="constant",
        center=[0.0, 0.0],
    )
    assert constant[0, 0] < 0.0
    for mode in ("hard_no_cross", "tanh_no_cross", "rational_no_cross"):
        endpoint, _ = center_return_endpoints(
            positions,
            [0.5, 0.5],
            return_scale=1.0,
            radius_exponent=0.0,
            reference_radius=1.0,
            travel_mode=mode,
            center=[0.0, 0.0],
        )
        assert np.all(endpoint[:, 0] >= 0.0)
        assert np.all(endpoint[:, 0] <= positions[:, 0])


def test_route_audit_records_center_crossing_weight():
    axis = np.linspace(-5.0, 5.0, 101)
    _, constant = conservative_route_template(
        axis,
        [[0.5, 0.0], [2.0, 0.0]],
        [0.75, 0.25],
        routing_fraction=0.5,
        return_scale=1.0,
        radius_exponent=0.0,
        reference_radius=1.0,
        smoothing=0.2,
        travel_mode="constant",
        center=[0.0, 0.0],
    )
    _, bounded = conservative_route_template(
        axis,
        [[0.5, 0.0], [2.0, 0.0]],
        [0.75, 0.25],
        routing_fraction=0.5,
        return_scale=1.0,
        radius_exponent=0.0,
        reference_radius=1.0,
        smoothing=0.2,
        travel_mode="tanh_no_cross",
        center=[0.0, 0.0],
    )
    assert constant["sources_crossing_center"] == 1
    assert constant["source_weight_crossing_center"] == pytest.approx(0.75)
    assert bounded["sources_crossing_center"] == 0
    assert bounded["source_weight_crossing_center"] == 0.0


def test_invalid_travel_mode_is_rejected():
    with pytest.raises(ValueError, match="travel_mode"):
        center_return_endpoints(
            [[1.0, 0.0]],
            [1.0],
            return_scale=1.0,
            radius_exponent=0.0,
            reference_radius=1.0,
            travel_mode="teleport",
            center=[0.0, 0.0],
        )


def test_local_attractors_are_baryon_derived_and_tanh_bounded():
    positions = np.array([[-4.0, 0.0], [0.0, 1.0], [5.0, 0.0]])
    weights = np.array([0.2, 0.5, 0.3])
    endpoints, targets, _ = local_baryonic_attractor_endpoints(
        positions,
        weights,
        return_scale=3.0,
        softening=2.0,
        distance_power=2.0,
        local_mix=1.0,
        travel_mode="tanh_no_cross",
    )
    target_distance = np.linalg.norm(targets - positions, axis=1)
    travel = np.linalg.norm(endpoints - positions, axis=1)
    assert np.all(travel <= target_distance)
    assert np.all(travel <= 3.0)
    assert not np.allclose(targets, np.average(positions, axis=0, weights=weights))


def test_local_attractor_template_conserves_weight_and_single_source_collapses():
    axis = np.linspace(-10.0, 10.0, 101)
    image, audit = conservative_local_attractor_route_template(
        axis,
        [[-3.0, 0.0], [0.0, 1.0], [4.0, 0.0]],
        [0.2, 0.5, 0.3],
        routing_fraction=0.4,
        return_scale=2.0,
        smoothing=0.3,
        softening=1.0,
        distance_power=2.0,
        local_mix=1.0,
    )
    assert np.isclose(image.sum(), 1.0)
    assert audit["normalization_error"] < 1e-14
    assert audit["source_weight_crossing_target"] == 0.0
    _, point = conservative_local_attractor_route_template(
        axis,
        [[0.0, 0.0]],
        [1.0],
        routing_fraction=0.9,
        return_scale=5.0,
        smoothing=0.3,
        softening=1.0,
        distance_power=2.0,
        local_mix=1.0,
    )
    assert np.allclose(point["local_map"], point["routed_map"])
