from __future__ import annotations

import numpy as np
import pytest

from voidscreen.observation_adapters import evaluate_observation_targets


def model(target: str = "massive_tracers") -> dict:
    return {
        "observables": [
            {
                "id": "acceleration",
                "target": target,
                "rank": "vector",
                "unit": "m/s^2",
            }
        ]
    }


@pytest.mark.parametrize("dimensions,cells", [(2, 65), (3, 33)])
def test_solid_body_acceleration_recovers_exact_circular_speed(
    dimensions: int, cells: int
) -> None:
    spacing = 0.25
    origin = -0.5 * (cells - 1) * spacing
    axes = [origin + np.arange(cells) * spacing for _ in range(dimensions)]
    mesh = np.meshgrid(*axes, indexing="ij")
    omega = 3.0
    observables = {
        "acceleration__axis0": -(omega**2) * mesh[0],
        "acceleration__axis1": -(omega**2) * mesh[1],
    }
    if dimensions == 3:
        observables["acceleration__axis2"] = np.zeros_like(mesh[2])
    radii = np.asarray([0.5, 1.0, 2.0, 3.0])
    expected = omega * radii
    evaluation, rows = evaluate_observation_targets(
        model(),
        observables,
        {
            "coordinateSystem": f"cartesian_{dimensions}d",
            "dimensions": dimensions,
            "spacing": [spacing] * dimensions,
            "origin": [origin] * dimensions,
        },
        [
            {
                "schemaVersion": "sigma-observation-target/1",
                "id": "solid-body",
                "kind": "circular_speed_curve",
                "observable": "acceleration",
                "centerM": [0.0] * dimensions,
                "planeAxes": [0, 1],
                "radiiM": radii.tolist(),
                "observedSpeedsMPerS": expected.tolist(),
                "uncertaintiesMPerS": [0.2] * len(radii),
                "azimuthalSamples": 128,
                "minimumAzimuthalCoverage": 1.0,
                "provenance": {"kind": "analytic solid-body fixture"},
                "license": {"id": "CC0-1.0", "redistributionAllowed": True},
            }
        ],
    )
    predicted = np.asarray([row["predicted_speed_m_s"] for row in rows])
    np.testing.assert_allclose(predicted, expected, rtol=2e-14, atol=2e-14)
    assert evaluation["scoredTargetCount"] == 1
    assert evaluation["rmseMPerS"] < 1e-12
    assert evaluation["chiSquare"] < 1e-20


def _axisymmetric_solid_body_fixture(
    *, cells_r: int = 65, cells_z: int = 33, spacing: float = 0.25
) -> tuple[dict, dict]:
    radius = np.arange(cells_r, dtype=float) * spacing
    vertical = -0.5 * (cells_z - 1) * spacing + np.arange(cells_z) * spacing
    radial_grid, vertical_grid = np.meshgrid(radius, vertical, indexing="ij")
    omega = 3.0
    observables = {
        "acceleration__axis0": -(omega**2) * radial_grid,
        "acceleration__axis1": np.zeros_like(vertical_grid),
    }
    geometry = {
        "coordinateSystem": "axisymmetric_cylindrical",
        "dimensions": 2,
        "spacing": [spacing, spacing],
        "origin": [0.0, float(vertical[0])],
        "axisOrder": ["r", "z"],
    }
    return observables, geometry


def test_axisymmetric_solid_body_field_recovers_exact_circular_speed() -> None:
    observables, geometry = _axisymmetric_solid_body_fixture()
    radii = np.asarray([0.375, 0.875, 1.625, 2.375])
    expected = 3.0 * radii
    evaluation, rows = evaluate_observation_targets(
        model(),
        observables,
        geometry,
        [
            {
                "schemaVersion": "sigma-observation-target/1",
                "id": "axisymmetric-solid-body",
                "kind": "circular_speed_curve",
                "observable": "acceleration",
                "centerM": [0.0, 0.125],
                "radiiM": radii.tolist(),
                "observedSpeedsMPerS": expected.tolist(),
                "uncertaintiesMPerS": [0.2] * len(radii),
                "minimumAzimuthalCoverage": 1.0,
                "provenance": {"kind": "analytic axisymmetric fixture"},
                "license": {"id": "CC0-1.0", "redistributionAllowed": True},
            }
        ],
    )
    predicted = np.asarray([row["predicted_speed_m_s"] for row in rows])
    np.testing.assert_allclose(predicted, expected, rtol=1e-14, atol=1e-14)
    target = evaluation["targets"][0]
    assert target["samplingMode"] == "axisymmetric_midplane_direct"
    assert target["coordinateSystem"] == "axisymmetric_cylindrical"
    assert target["axisOrder"] == ["r", "z"]
    assert target["samplingPlaneZM"] == 0.125
    assert target["planeAxes"] is None
    assert target["azimuthalSamples"] is None
    assert target["score"]["rmseMPerS"] < 1e-12


def test_full_covariance_is_used_for_scoring() -> None:
    cells = 33
    axis = np.linspace(-4.0, 4.0, cells)
    x, y = np.meshgrid(axis, axis, indexing="ij")
    observables = {
        "acceleration__axis0": -x,
        "acceleration__axis1": -y,
    }
    evaluation, _rows = evaluate_observation_targets(
        model(),
        observables,
        {
            "coordinateSystem": "cartesian_2d",
            "dimensions": 2,
            "spacing": [0.25, 0.25],
            "origin": [-4.0, -4.0],
        },
        [
            {
                "schemaVersion": "sigma-observation-target/1",
                "id": "covariance",
                "kind": "circular_speed_curve",
                "observable": "acceleration",
                "centerM": [0.0, 0.0],
                "radiiM": [1.0, 2.0],
                "observedSpeedsMPerS": [1.1, 1.3],
                "covarianceM2PerS2": [[0.04, 0.01], [0.01, 0.09]],
                "provenance": {"kind": "covariance fixture"},
                "license": {"id": "CC0-1.0", "redistributionAllowed": True},
            }
        ],
    )
    residual = np.asarray([1.0 - 1.1, 2.0 - 1.3])
    covariance = np.asarray([[0.04, 0.01], [0.01, 0.09]])
    expected = float(residual @ np.linalg.solve(covariance, residual))
    assert evaluation["chiSquare"] == pytest.approx(expected)


@pytest.mark.parametrize("dimensions,cells", [(2, 65), (3, 33)])
def test_resolved_velocity_field_recovers_projected_solid_body_map(
    dimensions: int, cells: int
) -> None:
    spacing = 0.25
    origin = -0.5 * (cells - 1) * spacing
    axes = [origin + np.arange(cells) * spacing for _ in range(dimensions)]
    mesh = np.meshgrid(*axes, indexing="ij")
    omega = 3.0
    observables = {
        "acceleration__axis0": -(omega**2) * mesh[0],
        "acceleration__axis1": -(omega**2) * mesh[1],
    }
    if dimensions == 3:
        observables["acceleration__axis2"] = np.zeros_like(mesh[2])
    map_axis = np.linspace(-2.0, 2.0, 17)
    major, minor = np.meshgrid(map_axis, map_axis, indexing="ij")
    inclination = 60.0
    systemic = 100.0
    expected = omega * major * np.sin(np.radians(inclination))
    arrays = {
        "major": major,
        "minor": minor,
        "observed": expected + systemic,
        "uncertainty": np.full_like(major, 0.2),
        "mask": np.ones_like(major),
    }
    evaluation, rows = evaluate_observation_targets(
        model(),
        observables,
        {
            "coordinateSystem": f"cartesian_{dimensions}d",
            "dimensions": dimensions,
            "spacing": [spacing] * dimensions,
            "origin": [origin] * dimensions,
        },
        [
            {
                "schemaVersion": "sigma-observation-target/1",
                "id": "solid-body-map",
                "kind": "line_of_sight_velocity_field",
                "observable": "acceleration",
                "centerM": [0.0] * dimensions,
                "planeAxes": [0, 1],
                "inclinationDeg": inclination,
                "handedness": 1,
                "majorCoordinateArrayKey": "major",
                "minorCoordinateArrayKey": "minor",
                "observedVelocityArrayKey": "observed",
                "uncertaintyArrayKey": "uncertainty",
                "observedVelocityZeroPointMPerS": systemic,
                "maskArrayKey": "mask",
                "minimumValidPixels": 200,
                "provenance": {"kind": "analytic projected solid-body fixture"},
                "license": {"id": "CC0-1.0", "redistributionAllowed": True},
            }
        ],
        arrays=arrays,
    )
    target = evaluation["targets"][0]
    assert evaluation["targetKinds"] == ["line_of_sight_velocity_field"]
    assert target["mapShape"] == [17, 17]
    assert target["score"]["validPoints"] == 17 * 17 - 1
    assert target["score"]["rmseMPerS"] < 1e-12
    assert len(rows) == 17 * 17 - 1
    assert {"row_index", "column_index", "predicted_velocity_m_s"}.issubset(rows[0])


def test_axisymmetric_field_recovers_resolved_projected_velocity_map() -> None:
    observables, geometry = _axisymmetric_solid_body_fixture()
    map_axis = np.linspace(-2.0, 2.0, 17)
    major, minor = np.meshgrid(map_axis, map_axis, indexing="ij")
    inclination = 60.0
    systemic = 100.0
    expected = 3.0 * major * np.sin(np.radians(inclination))
    arrays = {
        "major": major,
        "minor": minor,
        "observed": expected + systemic,
        "uncertainty": np.full_like(major, 0.2),
        "mask": np.ones_like(major),
    }
    evaluation, rows = evaluate_observation_targets(
        model(),
        observables,
        geometry,
        [
            {
                "schemaVersion": "sigma-observation-target/1",
                "id": "axisymmetric-solid-body-map",
                "kind": "line_of_sight_velocity_field",
                "observable": "acceleration",
                "centerM": [0.0, 0.125],
                "inclinationDeg": inclination,
                "handedness": 1,
                "majorCoordinateArrayKey": "major",
                "minorCoordinateArrayKey": "minor",
                "observedVelocityArrayKey": "observed",
                "uncertaintyArrayKey": "uncertainty",
                "observedVelocityZeroPointMPerS": systemic,
                "maskArrayKey": "mask",
                "minimumValidPixels": 200,
                "provenance": {"kind": "axisymmetric projected solid-body fixture"},
                "license": {"id": "CC0-1.0", "redistributionAllowed": True},
            }
        ],
        arrays=arrays,
    )
    target = evaluation["targets"][0]
    assert target["samplingMode"] == "axisymmetric_midplane_direct"
    assert target["axisOrder"] == ["r", "z"]
    assert target["planeAxes"] is None
    assert target["score"]["validPoints"] == 17 * 17 - 1
    assert target["score"]["rmseMPerS"] < 1e-12
    assert len(rows) == 17 * 17 - 1


@pytest.mark.parametrize(
    ("geometry_change", "target_change", "message"),
    [
        ({"axisOrder": ["z", "r"]}, {}, "axisOrder"),
        ({"origin": [1.0, -4.0]}, {}, "radial origin"),
        ({}, {"centerM": [1.0, 0.0]}, "centerM"),
        ({}, {"planeAxes": [0, 1]}, "do not accept Cartesian planeAxes"),
        ({}, {"azimuthalSamples": 128}, "does not accept azimuthalSamples"),
    ],
)
def test_axisymmetric_curve_rejects_ambiguous_coordinate_semantics(
    geometry_change: dict, target_change: dict, message: str
) -> None:
    observables, geometry = _axisymmetric_solid_body_fixture()
    geometry.update(geometry_change)
    target = {
        "schemaVersion": "sigma-observation-target/1",
        "id": "axisymmetric-coordinate-gate",
        "kind": "circular_speed_curve",
        "observable": "acceleration",
        "centerM": [0.0, 0.0],
        "radiiM": [1.0],
        "provenance": {"kind": "negative axisymmetric fixture"},
        "license": {"id": "CC0-1.0", "redistributionAllowed": True},
        **target_change,
    }
    with pytest.raises(ValueError, match=message):
        evaluate_observation_targets(model(), observables, geometry, [target])


def test_velocity_field_applies_declared_beam_and_intensity_weighting() -> None:
    cells = 65
    axis = np.linspace(-8.0, 8.0, cells)
    x, y = np.meshgrid(axis, axis, indexing="ij")
    observables = {
        "acceleration__axis0": -x,
        "acceleration__axis1": -y,
    }
    map_axis = np.linspace(-2.0, 2.0, 17)
    major, minor = np.meshgrid(map_axis, map_axis, indexing="ij")
    kernel_axis = np.arange(-3, 4)
    kernel_x, kernel_y = np.meshgrid(kernel_axis, kernel_axis, indexing="ij")
    kernel = np.exp(-0.5 * (kernel_x**2 + kernel_y**2))
    arrays = {
        "major": major,
        "minor": minor,
        "observed": major * np.sin(np.radians(45.0)),
        "uncertainty": np.ones_like(major),
        "intensity": np.ones_like(major),
        "beam": kernel,
    }
    evaluation, _rows = evaluate_observation_targets(
        model(),
        observables,
        {
            "coordinateSystem": "cartesian_2d",
            "dimensions": 2,
            "spacing": [0.25, 0.25],
            "origin": [-8.0, -8.0],
        },
        [
            {
                "schemaVersion": "sigma-observation-target/1",
                "id": "beam-map",
                "kind": "line_of_sight_velocity_field",
                "observable": "acceleration",
                "centerM": [0.0, 0.0],
                "inclinationDeg": 45.0,
                "handedness": 1,
                "majorCoordinateArrayKey": "major",
                "minorCoordinateArrayKey": "minor",
                "observedVelocityArrayKey": "observed",
                "uncertaintyArrayKey": "uncertainty",
                "intensityWeightArrayKey": "intensity",
                "beamKernelArrayKey": "beam",
                "weighting": "intensity_inverse_variance",
                "minimumValidPixels": 100,
                "provenance": {"kind": "beam fixture"},
                "license": {"id": "CC0-1.0", "redistributionAllowed": True},
            }
        ],
        arrays=arrays,
    )
    target = evaluation["targets"][0]
    assert target["beamConvolution"]["kernelShape"] == [7, 7]
    assert target["beamConvolution"]["normalizedKernelSum"] == pytest.approx(1.0)
    assert target["score"]["weighting"] == "intensity_inverse_variance"


def test_velocity_score_mask_is_applied_after_beam_convolution() -> None:
    cells = 65
    axis = np.linspace(-8.0, 8.0, cells)
    x, y = np.meshgrid(axis, axis, indexing="ij")
    observables = {
        "acceleration__axis0": -x,
        "acceleration__axis1": -y,
    }
    map_axis = np.linspace(-2.0, 2.0, 17)
    major, minor = np.meshgrid(map_axis, map_axis, indexing="ij")
    kernel_axis = np.arange(-3, 4)
    kernel_x, kernel_y = np.meshgrid(kernel_axis, kernel_axis, indexing="ij")
    score_mask = np.zeros_like(major)
    score_mask[4:13, 4:13] = 1.0
    arrays = {
        "major": major,
        "minor": minor,
        "intensity": np.where(major > 0.5, 8.0, 1.0),
        "beam": np.exp(-0.5 * (kernel_x**2 + kernel_y**2)),
        "score_mask": score_mask,
    }
    geometry = {
        "coordinateSystem": "cartesian_2d",
        "dimensions": 2,
        "spacing": [0.25, 0.25],
        "origin": [-8.0, -8.0],
    }
    common = {
        "schemaVersion": "sigma-observation-target/1",
        "kind": "line_of_sight_velocity_field",
        "observable": "acceleration",
        "centerM": [0.0, 0.0],
        "inclinationDeg": 45.0,
        "handedness": 1,
        "majorCoordinateArrayKey": "major",
        "minorCoordinateArrayKey": "minor",
        "intensityWeightArrayKey": "intensity",
        "beamKernelArrayKey": "beam",
        "minimumValidPixels": 50,
        "provenance": {"kind": "post-convolution score-mask fixture"},
        "license": {"id": "CC0-1.0", "redistributionAllowed": True},
    }
    _full, full_rows = evaluate_observation_targets(
        model(), observables, geometry, [{**common, "id": "full"}], arrays=arrays
    )
    _masked, masked_rows = evaluate_observation_targets(
        model(),
        observables,
        geometry,
        [{**common, "id": "masked", "scoreMaskArrayKey": "score_mask"}],
        arrays=arrays,
    )
    full_predictions = {
        (row["row_index"], row["column_index"]): row["predicted_velocity_m_s"]
        for row in full_rows
    }
    assert len(masked_rows) == 9 * 9
    for row in masked_rows:
        key = (row["row_index"], row["column_index"])
        assert row["predicted_velocity_m_s"] == pytest.approx(
            full_predictions[key], abs=1e-12
        )


def test_velocity_nonpositive_inward_policy_is_explicit() -> None:
    axis = np.linspace(-8.0, 8.0, 65)
    x, y = np.meshgrid(axis, axis, indexing="ij")
    observables = {
        "acceleration__axis0": x,
        "acceleration__axis1": y,
    }
    map_axis = np.linspace(-2.0, 2.0, 17)
    major, minor = np.meshgrid(map_axis, map_axis, indexing="ij")
    arrays = {"major": major, "minor": minor}
    target = {
        "schemaVersion": "sigma-observation-target/1",
        "id": "nonpositive-inward",
        "kind": "line_of_sight_velocity_field",
        "observable": "acceleration",
        "centerM": [0.0, 0.0],
        "inclinationDeg": 45.0,
        "handedness": 1,
        "majorCoordinateArrayKey": "major",
        "minorCoordinateArrayKey": "minor",
        "minimumValidPixels": 200,
        "provenance": {"kind": "nonpositive-inward fixture"},
        "license": {"id": "CC0-1.0", "redistributionAllowed": True},
    }
    geometry = {
        "coordinateSystem": "cartesian_2d",
        "dimensions": 2,
        "spacing": [0.25, 0.25],
        "origin": [-8.0, -8.0],
    }
    with pytest.raises(ValueError, match="too few valid"):
        evaluate_observation_targets(
            model(), observables, geometry, [target], arrays=arrays
        )
    evaluation, rows = evaluate_observation_targets(
        model(),
        observables,
        geometry,
        [{**target, "nonPositiveInwardPolicy": "zero_speed"}],
        arrays=arrays,
    )
    assert evaluation["targets"][0]["nonPositiveInwardPolicy"] == "zero_speed"
    assert len(rows) == 17 * 17
    assert all(row["predicted_velocity_m_s"] == 0.0 for row in rows)


def test_circular_speed_rejects_a_photon_observable() -> None:
    zeros = np.zeros((17, 17))
    target = {
        "schemaVersion": "sigma-observation-target/1",
        "id": "wrong-channel",
        "kind": "circular_speed_curve",
        "observable": "acceleration",
        "centerM": [0.0, 0.0],
        "radiiM": [1.0],
        "provenance": {"kind": "negative fixture"},
        "license": {"id": "CC0-1.0", "redistributionAllowed": True},
    }
    with pytest.raises(ValueError, match="massive_tracers"):
        evaluate_observation_targets(
            model("photons"),
            {"acceleration__axis0": zeros, "acceleration__axis1": zeros},
            {
                "coordinateSystem": "cartesian_2d",
                "dimensions": 2,
                "spacing": [1.0, 1.0],
                "origin": [-8.0, -8.0],
            },
            [target],
        )


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"provenance": None}, "requires provenance"),
        ({"license": None}, "explicit license"),
        ({"fittedNuisanceParameters": 1}, "smaller than the scored point count"),
    ],
)
def test_python_execution_repeats_metadata_and_nuisance_gates(
    change: dict, message: str
) -> None:
    zeros = np.zeros((17, 17))
    target = {
        "schemaVersion": "sigma-observation-target/1",
        "id": "metadata-gate",
        "kind": "circular_speed_curve",
        "observable": "acceleration",
        "centerM": [0.0, 0.0],
        "radiiM": [1.0],
        "observedSpeedsMPerS": [1.0],
        "uncertaintiesMPerS": [0.1],
        "provenance": {"kind": "negative fixture"},
        "license": {"id": "CC0-1.0", "redistributionAllowed": True},
        **change,
    }
    with pytest.raises(ValueError, match=message):
        evaluate_observation_targets(
            model(),
            {"acceleration__axis0": zeros, "acceleration__axis1": zeros},
            {
                "coordinateSystem": "cartesian_2d",
                "dimensions": 2,
                "spacing": [1.0, 1.0],
                "origin": [-8.0, -8.0],
            },
            [target],
        )


def test_python_execution_caps_target_count() -> None:
    with pytest.raises(ValueError, match="at most 32"):
        evaluate_observation_targets(model(), {}, {}, [{}] * 33)
