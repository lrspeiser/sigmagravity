from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from voidscreen.field_job import execute_field_job, write_array_bundle
from voidscreen.observation_evaluation_job import (
    execute_observation_evaluation_job,
)


def field_model(dimensions: int, observable_target: str = "massive_tracers") -> dict:
    coordinate_system = f"cartesian_{dimensions}d"
    return {
        "schemaVersion": "sigma-field-model/1",
        "name": f"P0732 {dimensions}D solid-body fixture",
        "modelClass": "stationary_elliptic",
        "source": {
            "format": "plain_text",
            "text": "laplacian(u) = forcing; acceleration = -gradient(u)",
            "confirmedCanonical": True,
        },
        "geometry": {
            "coordinateSystem": coordinate_system,
            "dimensions": dimensions,
            "domain": {"lengthUnit": "m", "boundaryExtent": "fixture box"},
        },
        "fields": {
            "forcing": {
                "rank": "scalar",
                "role": "source",
                "unit": "1/s^2",
                "datasetKey": "forcing",
            },
            "u": {
                "rank": "scalar",
                "role": "solved",
                "unit": "m^2/s^2",
                "boundary": {"type": "dirichlet", "value": 0.0},
            },
        },
        "parameters": {},
        "equations": [
            {
                "id": "poisson",
                "kind": "equality",
                "lhs": {"op": "laplacian", "args": [{"field": "u"}]},
                "rhs": {"field": "forcing"},
            }
        ],
        "observables": [
            {
                "id": "acceleration",
                "target": observable_target,
                "rank": "vector",
                "unit": "m/s^2",
                "expression": {
                    "op": "negate",
                    "args": [{"op": "gradient", "args": [{"field": "u"}]}],
                },
            }
        ],
        "dataRequirements": [
            {"key": "forcing", "rank": "scalar", "unit": "1/s^2"}
        ],
        "solver": {
            "family": "finite_volume_elliptic",
            "relativeTolerance": 1e-10,
            "maxIterations": 8,
            "damping": 1.0,
        },
        "parameterPolicy": {"mode": "universal_fixed", "perObjectParameters": []},
    }


def bundle_metadata(dimensions: int, spacing: float, arrays: dict) -> dict:
    extent = 0.5 * (17 - 1) * spacing
    return {
        "schemaVersion": "sigma-array-bundle-request/1",
        "geometry": {
            "coordinateSystem": f"cartesian_{dimensions}d",
            "dimensions": dimensions,
            "spacing": [spacing] * dimensions,
            "origin": [-extent] * dimensions,
            "lengthUnit": "m",
        },
        "arrays": arrays,
        "provenance": {"kind": "P0732 manufactured fixture"},
        "license": {"id": "CC0-1.0", "redistributionAllowed": True},
    }


def array_spec(npz_key: str, unit: str, role: str = "observation") -> dict:
    return {"npzKey": npz_key, "unit": unit, "rank": "scalar", "role": role}


def solve_fixture(
    tmp_path: Path,
    dimensions: int,
    targets: list[dict],
    observation_arrays: dict,
    *,
    observable_target: str = "massive_tracers",
) -> tuple[Path, Path, Path]:
    cells = 17
    spacing = 0.5
    omega = 1.25
    axis = np.linspace(-4.0, 4.0, cells)
    mesh = np.meshgrid(*([axis] * dimensions), indexing="ij")
    potential = 0.5 * omega**2 * (mesh[0] ** 2 + mesh[1] ** 2)
    forcing = np.full_like(potential, 2.0 * omega**2)
    combined_arrays = {
        "forcing": forcing,
        "boundary": potential,
        **observation_arrays,
    }
    combined_metadata = {
        "forcing": array_spec("forcing", "1/s^2", "source"),
        "u_boundary": array_spec("boundary", "m^2/s^2", "boundary"),
        **{
            key: array_spec(key, "m" if key in {"major", "minor"} else "1")
            for key in observation_arrays
        },
    }
    for key in ("observed", "uncertainty"):
        if key in combined_metadata:
            combined_metadata[key]["unit"] = "m/s"
    field_bundle = tmp_path / f"field_bundle_{dimensions}d"
    write_array_bundle(
        field_bundle,
        combined_arrays,
        bundle_metadata(dimensions, spacing, combined_metadata),
    )
    observation_bundle = tmp_path / f"observation_bundle_{dimensions}d"
    observation_values = observation_arrays or {"placeholder": np.zeros((5, 5))}
    observation_metadata = {
        key: array_spec(key, "m" if key in {"major", "minor"} else "1")
        for key in observation_values
    }
    for key in ("observed", "uncertainty"):
        if key in observation_metadata:
            observation_metadata[key]["unit"] = "m/s"
    write_array_bundle(
        observation_bundle,
        observation_values,
        bundle_metadata(dimensions, spacing, observation_metadata),
    )
    request = {
        "schemaVersion": "sigma-field-job-request/1",
        "boundaryFields": {"u": {"arrayKey": "u_boundary"}},
        "requestedObservables": ["acceleration"],
    }
    source_run = tmp_path / f"source_{dimensions}d"
    integrated_run = tmp_path / f"integrated_{dimensions}d"
    execute_field_job(
        field_model(dimensions, observable_target), field_bundle, request, source_run
    )
    execute_field_job(
        field_model(dimensions, observable_target),
        field_bundle,
        {**request, "observationTargets": targets},
        integrated_run,
    )
    return source_run, observation_bundle, integrated_run


def test_decoupled_2d_curve_byte_matches_integrated_field_job(tmp_path: Path) -> None:
    radii = [0.5, 1.0, 2.0, 3.0]
    target = {
        "schemaVersion": "sigma-observation-target/1",
        "id": "p0732-2d-curve",
        "kind": "circular_speed_curve",
        "observable": "acceleration",
        "centerM": [0.0, 0.0],
        "planeAxes": [0, 1],
        "radiiM": radii,
        "observedSpeedsMPerS": [1.25 * value for value in radii],
        "uncertaintiesMPerS": [0.1] * len(radii),
        "minimumAzimuthalCoverage": 1.0,
        "provenance": {"kind": "P0732 2D curve fixture"},
        "license": {"id": "CC0-1.0", "redistributionAllowed": True},
    }
    source, bundle, integrated = solve_fixture(tmp_path, 2, [target], {})
    standalone = tmp_path / "standalone_2d"
    execute_observation_evaluation_job(
        source,
        bundle,
        {
            "schemaVersion": "sigma-observation-evaluation-job-request/1",
            "observationTargets": [target],
        },
        standalone,
    )
    assert (standalone / "observation_scores.json").read_bytes() == (
        integrated / "observation_scores.json"
    ).read_bytes()
    assert (standalone / "observation_predictions.csv").read_bytes() == (
        integrated / "observation_predictions.csv"
    ).read_bytes()
    resource = json.loads((standalone / "resource_log.json").read_text())
    assert resource["fieldSolverInvocations"] == 0


def test_decoupled_3d_velocity_map_byte_matches_integrated_field_job(
    tmp_path: Path,
) -> None:
    map_axis = np.linspace(-2.0, 2.0, 9)
    major, minor = np.meshgrid(map_axis, map_axis, indexing="ij")
    inclination = 55.0
    systemic = 120.0
    observed = 1.25 * major * np.sin(np.radians(inclination)) + systemic
    arrays = {
        "major": major,
        "minor": minor,
        "observed": observed,
        "uncertainty": np.full_like(major, 0.25),
        "score_mask": np.ones_like(major),
    }
    target = {
        "schemaVersion": "sigma-observation-target/1",
        "id": "p0732-3d-map",
        "kind": "line_of_sight_velocity_field",
        "observable": "acceleration",
        "centerM": [0.0, 0.0, 0.0],
        "planeAxes": [0, 1],
        "inclinationDeg": inclination,
        "handedness": 1,
        "majorCoordinateArrayKey": "major",
        "minorCoordinateArrayKey": "minor",
        "observedVelocityArrayKey": "observed",
        "uncertaintyArrayKey": "uncertainty",
        "observedVelocityZeroPointMPerS": systemic,
        "scoreMaskArrayKey": "score_mask",
        "minimumValidPixels": 50,
        "provenance": {"kind": "P0732 3D velocity-map fixture"},
        "license": {"id": "CC0-1.0", "redistributionAllowed": True},
    }
    source, bundle, integrated = solve_fixture(tmp_path, 3, [target], arrays)
    standalone = tmp_path / "standalone_3d"
    execute_observation_evaluation_job(
        source,
        bundle,
        {
            "schemaVersion": "sigma-observation-evaluation-job-request/1",
            "observationTargets": [target],
        },
        standalone,
    )
    assert (standalone / "observation_scores.json").read_bytes() == (
        integrated / "observation_scores.json"
    ).read_bytes()
    assert (standalone / "observation_velocity_field_predictions.csv").read_bytes() == (
        integrated / "observation_velocity_field_predictions.csv"
    ).read_bytes()
    result = json.loads((standalone / "scientific_result.json").read_text())
    assert result["evaluationAddedGravityParameters"] == 0
    assert result["parameterAccounting"]["perObjectCount"] == 0


def test_decoupled_photon_maps_byte_match_integrated_field_job(tmp_path: Path) -> None:
    target = {
        "schemaVersion": "sigma-observation-target/1",
        "id": "p0734-photon-map",
        "kind": "photon_lensing_map",
        "observable": "acceleration",
        "northAxis": 0,
        "eastAxis": 1,
        "lineOfSightAxis": 2,
        "distanceRatio": 0.7,
        "lensAngularDiameterDistanceM": 1.0e20,
        "minimumValidPixels": 25,
        "provenance": {"kind": "P0734 photon-map parity fixture"},
        "license": {"id": "CC0-1.0", "redistributionAllowed": True},
    }
    source, bundle, integrated = solve_fixture(
        tmp_path,
        3,
        [target],
        {},
        observable_target="photons",
    )
    standalone = tmp_path / "standalone_photon"
    execute_observation_evaluation_job(
        source,
        bundle,
        {
            "schemaVersion": "sigma-observation-evaluation-job-request/1",
            "observationTargets": [target],
        },
        standalone,
    )
    assert (standalone / "observation_scores.json").read_bytes() == (
        integrated / "observation_scores.json"
    ).read_bytes()
    assert (standalone / "observation_photon_lensing_maps.npz").read_bytes() == (
        integrated / "observation_photon_lensing_maps.npz"
    ).read_bytes()
    scores = json.loads((standalone / "observation_scores.json").read_text())
    assert scores["targets"][0]["observableTarget"] == "photons"
    assert scores["targets"][0]["state"] == "predicted_not_scored"
    assert scores["mapArchive"]["path"] == "observation_photon_lensing_maps.npz"


def test_decoupled_raw_multiple_images_byte_match_integrated_field_job(
    tmp_path: Path,
) -> None:
    target = {
        "schemaVersion": "sigma-observation-target/1",
        "id": "p0735-raw-images",
        "kind": "multiple_image_systems",
        "observable": "acceleration",
        "northAxis": 0,
        "eastAxis": 1,
        "lineOfSightAxis": 2,
        "lensAngularDiameterDistanceM": 1.0e6,
        "skyCenterM": [0.0, 0.0, 0.0],
        "rootSearchBoundArcsec": 0.5,
        "rootGridPoints": 21,
        "supplementalGridPoints": [],
        "closureToleranceArcsec": 1.0e-4,
        "deduplicationToleranceArcsec": 0.01,
        "jacobianStepArcsec": 0.01,
        "families": [
            {
                "id": "source-a",
                "distanceRatio": 0.7,
                "observedImagesArcsec": [[-0.1, 0.0], [0.1, 0.0]],
                "positionUncertaintiesArcsec": [0.01, 0.01],
            }
        ],
        "provenance": {"kind": "P0735 raw-image parity fixture"},
        "license": {"id": "CC0-1.0", "redistributionAllowed": True},
    }
    source, bundle, integrated = solve_fixture(
        tmp_path,
        3,
        [target],
        {},
        observable_target="photons",
    )
    standalone = tmp_path / "standalone_raw_images"
    execute_observation_evaluation_job(
        source,
        bundle,
        {
            "schemaVersion": "sigma-observation-evaluation-job-request/1",
            "observationTargets": [target],
        },
        standalone,
    )
    for name in (
        "observation_scores.json",
        "observation_multiple_image_predictions.csv",
        "observation_multiple_image_families.csv",
        "observation_multiple_image_roots.npz",
    ):
        assert (standalone / name).read_bytes() == (integrated / name).read_bytes()
    scores = json.loads((standalone / "observation_scores.json").read_text())
    assert scores["targets"][0]["state"] == "incomplete_topology"
    assert scores["targets"][0]["score"]["channels"]["image_position_arcsec"]["rmse"] is None
    assert scores["rootArchive"]["path"] == "observation_multiple_image_roots.npz"
