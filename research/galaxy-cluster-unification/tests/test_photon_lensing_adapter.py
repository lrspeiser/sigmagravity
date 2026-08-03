from __future__ import annotations

import numpy as np
import pytest

from voidscreen.observation_adapters import evaluate_observation_targets
from voidscreen.sky_lensing import C_M_S


def photon_model(target: str = "photons") -> dict:
    return {
        "observables": [
            {
                "id": "photon_acceleration",
                "target": target,
                "rank": "vector",
                "unit": "m/s^2",
            }
        ]
    }


def target(**changes) -> dict:
    return {
        "schemaVersion": "sigma-observation-target/1",
        "id": "photon-map",
        "kind": "photon_lensing_map",
        "observable": "photon_acceleration",
        "northAxis": 0,
        "eastAxis": 1,
        "lineOfSightAxis": 2,
        "distanceRatio": 0.7,
        "lensAngularDiameterDistanceM": 1.0e20,
        "minimumValidPixels": 25,
        "provenance": {"kind": "analytic photon fixture"},
        "license": {"id": "CC0-1.0", "redistributionAllowed": True},
        **changes,
    }


def geometry(shape=(9, 11, 13), spacing=(2.0, 3.0, 5.0)) -> dict:
    return {
        "coordinateSystem": "cartesian_3d",
        "dimensions": 3,
        "spacing": list(spacing),
        "origin": [
            -0.5 * (cells - 1) * step for cells, step in zip(shape, spacing, strict=True)
        ],
    }


def axisymmetric_geometry(
    shape: tuple[int, int] = (33, 65),
    spacing: tuple[float, float] = (1.0, 1.0),
) -> dict:
    return {
        "coordinateSystem": "axisymmetric_cylindrical",
        "dimensions": 2,
        "spacing": list(spacing),
        "origin": [0.0, -0.5 * (shape[1] - 1) * spacing[1]],
        "axisOrder": ["r", "z"],
    }


def axisymmetric_target(**changes) -> dict:
    specification = {
        "axisymmetricInclinationDeg": 0.0,
        "skyShape": [33, 33],
        "lineOfSightSamples": 65,
    }
    specification.update(changes)
    result = target(**specification)
    for key in ("northAxis", "eastAxis", "lineOfSightAxis"):
        if key not in changes:
            result.pop(key, None)
    return result


def axisymmetric_observables(
    radial: np.ndarray, vertical: np.ndarray
) -> dict[str, np.ndarray]:
    return {
        "photon_acceleration__axis0": radial,
        "photon_acceleration__axis1": vertical,
    }


def uniform_observables(shape, north=0.0, east=0.0, los=0.0) -> dict:
    return {
        "photon_acceleration__axis0": np.full(shape, north),
        "photon_acceleration__axis1": np.full(shape, east),
        "photon_acceleration__axis2": np.full(shape, los),
    }


def test_uniform_field_freezes_sign_normalization_and_linear_scaling() -> None:
    shape = (9, 11, 13)
    spacing = (2.0, 3.0, 5.0)
    observables = uniform_observables(shape, north=4.0, east=-3.0)
    maps: dict[str, np.ndarray] = {}
    evaluation, rows = evaluate_observation_targets(
        photon_model(),
        observables,
        geometry(shape, spacing),
        [target()],
        map_outputs=maps,
    )
    path_length = (shape[2] - 1) * spacing[2]
    multiplier = -2.0 * 0.7 * path_length / C_M_S**2
    np.testing.assert_allclose(
        maps["target_000__alpha_north_radian"], multiplier * 4.0, rtol=1e-15
    )
    np.testing.assert_allclose(
        maps["target_000__alpha_east_radian"], multiplier * -3.0, rtol=1e-15
    )
    assert rows == []
    assert evaluation["targetKinds"] == ["photon_lensing_map"]
    assert evaluation["targets"][0]["observableTarget"] == "photons"
    assert evaluation["rmseMPerS"] is None

    doubled: dict[str, np.ndarray] = {}
    evaluate_observation_targets(
        photon_model(),
        observables,
        geometry(shape, spacing),
        [target(distanceRatio=1.4)],
        map_outputs=doubled,
    )
    np.testing.assert_allclose(
        doubled["target_000__alpha_east_radian"],
        2.0 * maps["target_000__alpha_east_radian"],
        rtol=1e-15,
    )


def test_affine_deflection_recovers_exact_lensing_invariants() -> None:
    shape = (17, 19, 21)
    spacing = (2.0e17, 3.0e17, 4.0e17)
    lens_distance = 2.0e22
    distance_ratio = 0.65
    axes = [
        origin + np.arange(cells) * step
        for origin, cells, step in zip(
            geometry(shape, spacing)["origin"], shape, spacing, strict=True
        )
    ]
    north_m, east_m, _los_m = np.meshgrid(*axes, indexing="ij")
    north_angle = north_m / lens_distance
    east_angle = east_m / lens_distance
    d_east_d_east = 0.04
    d_east_d_north = 0.01
    d_north_d_east = 0.01
    d_north_d_north = 0.02
    alpha_east = d_east_d_east * east_angle + d_east_d_north * north_angle
    alpha_north = d_north_d_east * east_angle + d_north_d_north * north_angle
    path_length = (shape[2] - 1) * spacing[2]
    field_scale = -(C_M_S**2) / (2.0 * distance_ratio * path_length)
    observables = {
        "photon_acceleration__axis0": field_scale * alpha_north,
        "photon_acceleration__axis1": field_scale * alpha_east,
        "photon_acceleration__axis2": np.zeros(shape),
    }
    maps: dict[str, np.ndarray] = {}
    evaluate_observation_targets(
        photon_model(),
        observables,
        geometry(shape, spacing),
        [
            target(
                distanceRatio=distance_ratio,
                lensAngularDiameterDistanceM=lens_distance,
            )
        ],
        map_outputs=maps,
    )
    np.testing.assert_allclose(maps["target_000__convergence"], 0.03, atol=2e-14)
    np.testing.assert_allclose(maps["target_000__shear_1"], 0.01, atol=2e-14)
    np.testing.assert_allclose(maps["target_000__shear_2"], 0.01, atol=2e-14)
    np.testing.assert_allclose(maps["target_000__rotation"], 0.0, atol=2e-14)
    expected_determinant = 0.96 * 0.98 - 0.01**2
    np.testing.assert_allclose(
        maps["target_000__jacobian_determinant"], expected_determinant, atol=3e-14
    )


def test_axis_permutation_preserves_named_north_and_east_maps() -> None:
    physical_shape = (7, 9, 11)
    physical_spacing = (2.0, 3.0, 5.0)
    rng = np.random.default_rng(734)
    physical = tuple(rng.normal(size=physical_shape) for _ in range(3))
    canonical = {
        f"photon_acceleration__axis{axis}": value
        for axis, value in enumerate(physical)
    }
    canonical_maps: dict[str, np.ndarray] = {}
    evaluate_observation_targets(
        photon_model(),
        canonical,
        geometry(physical_shape, physical_spacing),
        [target(lensAngularDiameterDistanceM=1.0e19)],
        map_outputs=canonical_maps,
    )

    declared_axes = (2, 0, 1)
    inverse = tuple(np.argsort(declared_axes))
    stored = [None, None, None]
    for physical_axis, storage_axis in enumerate(declared_axes):
        stored[storage_axis] = np.transpose(physical[physical_axis], axes=inverse)
    stored_shape = stored[0].shape
    stored_spacing = [0.0, 0.0, 0.0]
    for physical_axis, storage_axis in enumerate(declared_axes):
        stored_spacing[storage_axis] = physical_spacing[physical_axis]
    permuted_maps: dict[str, np.ndarray] = {}
    evaluate_observation_targets(
        photon_model(),
        {
            f"photon_acceleration__axis{axis}": value
            for axis, value in enumerate(stored)
        },
        geometry(stored_shape, tuple(stored_spacing)),
        [
            target(
                northAxis=declared_axes[0],
                eastAxis=declared_axes[1],
                lineOfSightAxis=declared_axes[2],
                lensAngularDiameterDistanceM=1.0e19,
            )
        ],
        map_outputs=permuted_maps,
    )
    for name, expected in canonical_maps.items():
        np.testing.assert_allclose(permuted_maps[name], expected, rtol=1e-13)


def test_exact_observations_are_scored_in_separate_channels() -> None:
    shape = (9, 11, 13)
    observables = uniform_observables(shape, north=4.0, east=-3.0)
    predicted: dict[str, np.ndarray] = {}
    evaluate_observation_targets(
        photon_model(), observables, geometry(shape), [target()], map_outputs=predicted
    )
    arrays = {
        "alpha_e": predicted["target_000__alpha_east_arcsec"],
        "alpha_n": predicted["target_000__alpha_north_arcsec"],
        "alpha_sigma": np.full(shape[:2], 0.05),
        "g1": predicted["target_000__reduced_shear_1"],
        "g2": predicted["target_000__reduced_shear_2"],
        "g_sigma": np.full(shape[:2], 0.01),
        "mask": np.ones(shape[:2]),
    }
    evaluation, _rows = evaluate_observation_targets(
        photon_model(),
        observables,
        geometry(shape),
        [
            target(
                observedAlphaEastArcsecArrayKey="alpha_e",
                observedAlphaNorthArcsecArrayKey="alpha_n",
                deflectionUncertaintyArcsecArrayKey="alpha_sigma",
                observedReducedShear1ArrayKey="g1",
                observedReducedShear2ArrayKey="g2",
                reducedShearUncertaintyArrayKey="g_sigma",
                scoreMaskArrayKey="mask",
            )
        ],
        arrays=arrays,
    )
    assert evaluation["scoredTargetCount"] == 1
    assert evaluation["rmseMPerS"] is None
    assert set(evaluation["channelAggregates"]) == {
        "deflection_arcsec",
        "reduced_shear_dimensionless",
    }
    assert evaluation["channelAggregates"]["deflection_arcsec"]["rmse"] < 1e-15
    assert (
        evaluation["channelAggregates"]["reduced_shear_dimensionless"]["rmse"]
        < 1e-15
    )


def test_point_mass_projection_recovers_gr_deflection() -> None:
    cells = (65, 65, 257)
    spacing = (1.0, 1.0, 1.0)
    axes = [
        -0.5 * (count - 1) * step + np.arange(count) * step
        for count, step in zip(cells, spacing, strict=True)
    ]
    north, east, line_of_sight = np.meshgrid(*axes, indexing="ij")
    radius_squared = north**2 + east**2 + line_of_sight**2
    safe_radius = np.sqrt(np.maximum(radius_squared, 0.25))
    gravity_mass = 2.0e24
    factor = -gravity_mass / safe_radius**3
    observables = {
        "photon_acceleration__axis0": factor * north,
        "photon_acceleration__axis1": factor * east,
        "photon_acceleration__axis2": factor * line_of_sight,
    }
    maps: dict[str, np.ndarray] = {}
    evaluate_observation_targets(
        photon_model(),
        observables,
        geometry(cells, spacing),
        [target(distanceRatio=1.0, lensAngularDiameterDistanceM=1.0e9)],
        map_outputs=maps,
    )
    offsets = np.arange(4, 33)
    center = cells[0] // 2
    predicted = np.abs(maps["target_000__alpha_east_radian"][center, center + offsets])
    expected = 4.0 * gravity_mass / (C_M_S**2 * offsets)
    relative = np.abs(predicted / expected - 1.0)
    assert float(np.median(relative)) < 0.02
    assert float(np.quantile(relative, 0.95)) < 0.04


def test_axisymmetric_face_on_harmonic_field_recovers_affine_convergence() -> None:
    shape = (33, 65)
    geometry_spec = axisymmetric_geometry(shape)
    radial_axis = np.arange(shape[0], dtype=float)
    path_length = float(shape[1] - 1)
    distance_ratio = 0.7
    lens_distance = 2.0e20
    expected_convergence = 0.03
    omega_squared = (
        expected_convergence
        * C_M_S**2
        / (2.0 * distance_ratio * path_length * lens_distance)
    )
    radial = np.broadcast_to(
        (-omega_squared * radial_axis)[:, None], shape
    ).copy()
    vertical = np.zeros(shape)
    maps: dict[str, np.ndarray] = {}
    evaluation, _rows = evaluate_observation_targets(
        photon_model(),
        axisymmetric_observables(radial, vertical),
        geometry_spec,
        [
            axisymmetric_target(
                distanceRatio=distance_ratio,
                lensAngularDiameterDistanceM=lens_distance,
            )
        ],
        map_outputs=maps,
    )
    sky_axis = np.linspace(-32.0, 32.0, 33)
    north, east = np.meshgrid(sky_axis, sky_axis, indexing="ij")
    interior = np.hypot(north, east) <= 24.0
    scale = expected_convergence / lens_distance
    np.testing.assert_allclose(
        maps["target_000__alpha_east_radian"][interior],
        scale * east[interior],
        rtol=2.0e-13,
        atol=1.0e-30,
    )
    np.testing.assert_allclose(
        maps["target_000__alpha_north_radian"][interior],
        scale * north[interior],
        rtol=2.0e-13,
        atol=1.0e-30,
    )
    invariant_interior = np.hypot(north, east) <= 20.0
    np.testing.assert_allclose(
        maps["target_000__convergence"][invariant_interior],
        expected_convergence,
        rtol=2.0e-13,
    )
    np.testing.assert_allclose(
        maps["target_000__shear_magnitude"][invariant_interior],
        0.0,
        atol=2.0e-15,
    )
    metadata = evaluation["targets"][0]
    assert metadata["coordinateSystem"] == "axisymmetric_cylindrical"
    assert metadata["samplingMode"] == "axisymmetric_cylindrical_ray_integral"
    assert metadata["projection"]["diagnostics"]["supportedPixels"] > 700


def test_axisymmetric_edge_on_uniform_vertical_field_has_exact_chord_length() -> None:
    shape = (33, 33)
    acceleration = 4.0
    observables = axisymmetric_observables(
        np.zeros(shape), np.full(shape, acceleration)
    )
    maps: dict[str, np.ndarray] = {}
    evaluate_observation_targets(
        photon_model(),
        observables,
        axisymmetric_geometry(shape),
        [
            axisymmetric_target(
                axisymmetricInclinationDeg=90.0,
                skyShape=[33, 33],
                lineOfSightSamples=65,
                distanceRatio=1.0,
            )
        ],
        map_outputs=maps,
    )
    east_axis = np.linspace(-32.0, 32.0, 33)
    expected_path = 2.0 * np.sqrt(np.maximum(32.0**2 - east_axis**2, 0.0))
    expected_north = -2.0 * acceleration * expected_path / C_M_S**2
    center_row = shape[0] // 2
    np.testing.assert_allclose(
        maps["target_000__alpha_north_radian"][center_row, 1:-1],
        expected_north[1:-1],
        rtol=2.0e-14,
        atol=1.0e-30,
    )
    np.testing.assert_allclose(
        maps["target_000__alpha_east_radian"][center_row, 1:-1],
        0.0,
        atol=1.0e-30,
    )


def test_axisymmetric_point_mass_recovers_gr_deflection() -> None:
    shape = (65, 257)
    geometry_spec = axisymmetric_geometry(shape)
    radial_axis = np.arange(shape[0], dtype=float)
    vertical_axis = geometry_spec["origin"][1] + np.arange(shape[1], dtype=float)
    radial_grid, vertical_grid = np.meshgrid(
        radial_axis, vertical_axis, indexing="ij"
    )
    radius_squared = radial_grid**2 + vertical_grid**2
    safe_radius = np.sqrt(np.maximum(radius_squared, 0.25))
    gravity_mass = 2.0e24
    radial = -gravity_mass * radial_grid / safe_radius**3
    vertical = -gravity_mass * vertical_grid / safe_radius**3
    center = 32
    offsets = np.arange(2, 17)
    physical_radius = 2.0 * offsets
    expected = 4.0 * gravity_mass / (C_M_S**2 * physical_radius)
    errors: dict[int, tuple[float, float]] = {}
    for line_samples in (17, 33, 65, 129, 257):
        maps: dict[str, np.ndarray] = {}
        evaluate_observation_targets(
            photon_model(),
            axisymmetric_observables(radial, vertical),
            geometry_spec,
            [
                axisymmetric_target(
                    skyShape=[65, 65],
                    lineOfSightSamples=line_samples,
                    distanceRatio=1.0,
                    lensAngularDiameterDistanceM=1.0e9,
                )
            ],
            map_outputs=maps,
        )
        predicted = np.abs(
            maps["target_000__alpha_east_radian"][center, center + offsets]
        )
        relative = np.abs(predicted / expected - 1.0)
        errors[line_samples] = (
            float(np.median(relative)),
            float(np.quantile(relative, 0.95)),
        )
    assert errors[17][0] > errors[33][0] > errors[65][0] > errors[129][0]
    assert abs(errors[129][0] - errors[257][0]) < 1.0e-5
    relative = errors[257]
    assert relative[0] < 0.02
    assert relative[1] < 0.04


def test_axisymmetric_exact_maps_score_in_separate_channels() -> None:
    shape = (33, 65)
    radial_axis = np.arange(shape[0], dtype=float)
    radial = np.broadcast_to((-1.0e-12 * radial_axis)[:, None], shape).copy()
    observables = axisymmetric_observables(radial, np.zeros(shape))
    predicted: dict[str, np.ndarray] = {}
    projection_target = axisymmetric_target()
    evaluate_observation_targets(
        photon_model(),
        observables,
        axisymmetric_geometry(shape),
        [projection_target],
        map_outputs=predicted,
    )
    finite = np.isfinite(predicted["target_000__reduced_shear_1"])
    arrays = {
        "alpha_e": predicted["target_000__alpha_east_arcsec"],
        "alpha_n": predicted["target_000__alpha_north_arcsec"],
        "alpha_sigma": np.full((33, 33), 0.05),
        "g1": predicted["target_000__reduced_shear_1"],
        "g2": predicted["target_000__reduced_shear_2"],
        "g_sigma": np.full((33, 33), 0.01),
        "mask": finite.astype(float),
    }
    evaluation, _rows = evaluate_observation_targets(
        photon_model(),
        observables,
        axisymmetric_geometry(shape),
        [
            {
                **projection_target,
                "observedAlphaEastArcsecArrayKey": "alpha_e",
                "observedAlphaNorthArcsecArrayKey": "alpha_n",
                "deflectionUncertaintyArcsecArrayKey": "alpha_sigma",
                "observedReducedShear1ArrayKey": "g1",
                "observedReducedShear2ArrayKey": "g2",
                "reducedShearUncertaintyArrayKey": "g_sigma",
                "scoreMaskArrayKey": "mask",
            }
        ],
        arrays=arrays,
    )
    assert evaluation["scoredTargetCount"] == 1
    for channel in evaluation["channelAggregates"].values():
        assert channel["rmse"] < 1.0e-15


def test_axisymmetric_photon_projection_rejects_ambiguous_geometry() -> None:
    shape = (9, 9)
    observables = axisymmetric_observables(np.zeros(shape), np.zeros(shape))
    bad_axis_order = axisymmetric_geometry(shape)
    bad_axis_order["axisOrder"] = ["z", "r"]
    with pytest.raises(ValueError, match="axisOrder"):
        evaluate_observation_targets(
            photon_model(), observables, bad_axis_order, [axisymmetric_target()]
        )
    bad_origin = axisymmetric_geometry(shape)
    bad_origin["origin"][0] = 1.0
    with pytest.raises(ValueError, match="radial origin"):
        evaluate_observation_targets(
            photon_model(), observables, bad_origin, [axisymmetric_target()]
        )
    with pytest.raises(ValueError, match="Cartesian sky-axis"):
        evaluate_observation_targets(
            photon_model(),
            observables,
            axisymmetric_geometry(shape),
            [axisymmetric_target(northAxis=0)],
        )
    with pytest.raises(ValueError, match="origin must match"):
        evaluate_observation_targets(
            photon_model(),
            observables,
            axisymmetric_geometry(shape),
            [axisymmetric_target(gridOriginM=[0.0, -3.0])],
        )


def test_photon_target_rejects_massive_only_observable_and_bad_axes() -> None:
    shape = (9, 9, 9)
    observables = uniform_observables(shape)
    with pytest.raises(ValueError, match="photons or both"):
        evaluate_observation_targets(
            photon_model("massive_tracers"),
            observables,
            geometry(shape),
            [target()],
        )
    with pytest.raises(ValueError, match="permutation"):
        evaluate_observation_targets(
            photon_model(),
            observables,
            geometry(shape),
            [target(eastAxis=0)],
        )
