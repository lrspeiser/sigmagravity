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
