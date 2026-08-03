from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from voidscreen.field_job import (
    execute_field_job,
    execute_request_file,
    load_array_bundle,
    model_sha256,
    require_model_confirmation,
    write_array_bundle,
)

ROOT = Path(__file__).resolve().parents[1]


def confirm_manifest(manifest: dict) -> dict:
    manifest["source"]["confirmedCanonical"] = True
    manifest["source"]["confirmedModelSha256"] = model_sha256(manifest)
    return manifest


def manufactured_manifest() -> dict:
    manifest = {
        "schemaVersion": "sigma-field-model/1",
        "name": "Content-addressed manufactured field",
        "modelClass": "stationary_elliptic",
        "source": {
            "format": "plain_text",
            "text": "laplacian(u) = forcing",
            "confirmedCanonical": False,
        },
        "geometry": {
            "coordinateSystem": "cartesian_2d",
            "dimensions": 2,
            "domain": {"lengthUnit": "m", "boundaryExtent": "unit square"},
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
                "id": "manufactured",
                "kind": "equality",
                "lhs": {"op": "laplacian", "args": [{"field": "u"}]},
                "rhs": {"field": "forcing"},
            }
        ],
        "observables": [
            {
                "id": "gradient",
                "target": "diagnostic",
                "rank": "vector",
                "unit": "m/s^2",
                "expression": {"op": "gradient", "args": [{"field": "u"}]},
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
    return confirm_manifest(manifest)


def test_model_confirmation_is_bound_to_exact_computational_hash():
    manifest = manufactured_manifest()
    assert require_model_confirmation(manifest) == model_sha256(manifest)
    manifest["solver"]["maxIterations"] += 1
    with pytest.raises(ValueError, match="exact computational model hash"):
        require_model_confirmation(manifest)


def bundle_metadata(spacing: float) -> dict:
    return {
        "schemaVersion": "sigma-array-bundle-request/1",
        "geometry": {
            "coordinateSystem": "cartesian_2d",
            "dimensions": 2,
            "spacing": [spacing, spacing],
            "lengthUnit": "m",
            "axisOrder": ["x", "y"],
            "referenceFrame": "manufactured_unit_square",
        },
        "arrays": {
            "forcing": {
                "npzKey": "raw_forcing",
                "unit": "1/s^2",
                "rank": "scalar",
                "role": "source",
            }
        },
        "provenance": {
            "kind": "analytic_manufactured_solution",
            "citation": "repository test fixture",
        },
        "license": {"id": "CC0-1.0", "redistributionAllowed": True},
    }


def manufactured_values(cells: int = 25) -> tuple[np.ndarray, np.ndarray, float]:
    axis = np.linspace(0.0, 1.0, cells)
    x, y = np.meshgrid(axis, axis, indexing="ij")
    expected = np.sin(np.pi * x) * np.sin(np.pi * y)
    forcing = -2.0 * np.pi**2 * expected
    return expected, forcing, 1.0 / (cells - 1)


def test_python_model_hash_matches_hosted_javascript_validator():
    root = ROOT / "hosted-simulator" / "examples" / "models"
    newtonian = json.loads((root / "newtonian-poisson.json").read_text(encoding="utf-8"))
    refracted = json.loads((root / "refracted-gravity.json").read_text(encoding="utf-8"))
    assert model_sha256(newtonian) == "43738f2c7bb3c4e94193763cf46f39b2a47fa852b25d1c063fef00fa7e1aa661"
    assert model_sha256(refracted) == "4a0c9be0ba6f430d4d073f0ee6bf1e98de437908de81d0ad533261fea5d14bef"


def test_cli_returns_structured_input_rejection(tmp_path: Path):
    request_path = tmp_path / "request.json"
    request_path.write_text(
        json.dumps({"schemaVersion": "not-a-field-job"}), encoding="utf-8"
    )
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "run_generic_field_job.py"),
            "run",
            "--request",
            str(request_path),
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 2
    failure = json.loads(completed.stderr.strip().splitlines()[-1])
    assert failure["state"] == "rejected_input"
    assert failure["errorType"] == "ValueError"


def test_array_bundle_detects_data_changed_after_registration(tmp_path: Path):
    _expected, forcing, spacing = manufactured_values(17)
    bundle_path = tmp_path / "bundle"
    bundle = write_array_bundle(
        bundle_path,
        {"raw_forcing": forcing},
        bundle_metadata(spacing),
    )
    loaded, arrays = load_array_bundle(bundle_path)
    assert loaded["bundleSha256"] == bundle["bundleSha256"]
    assert np.array_equal(arrays["forcing"], forcing)

    changed = forcing.copy()
    changed[8, 8] += 1.0
    with (bundle_path / "arrays.npz").open("wb") as handle:
        np.savez_compressed(handle, forcing=changed)
    with pytest.raises(ValueError, match="content hash mismatch"):
        load_array_bundle(bundle_path)


def test_large_physical_spacing_survives_javascript_json_roundtrip(tmp_path: Path):
    spacing = float(4_571_058_612_641_913_300)
    bundle_path = tmp_path / "large_spacing_bundle"
    bundle = write_array_bundle(
        bundle_path,
        {"raw_forcing": np.ones((9, 9))},
        bundle_metadata(spacing),
    )
    serialized = json.loads((bundle_path / "bundle.json").read_text(encoding="utf-8"))
    serialized["geometry"]["spacing"] = [
        4_571_058_612_641_913_300,
        4_571_058_612_641_913_300,
    ]
    (bundle_path / "bundle.json").write_text(
        json.dumps(serialized), encoding="utf-8"
    )
    loaded, _ = load_array_bundle(bundle_path)
    assert loaded["bundleSha256"] == bundle["bundleSha256"]
    assert loaded["geometry"]["spacing"] == [spacing, spacing]


def test_large_physical_origin_survives_javascript_json_roundtrip(tmp_path: Path):
    spacing = float(4_571_058_612_641_913_300)
    origin = float(-146_273_875_604_541_230_000)
    metadata = bundle_metadata(spacing)
    metadata["geometry"]["origin"] = [origin, origin]
    bundle_path = tmp_path / "large_origin_bundle"
    bundle = write_array_bundle(
        bundle_path,
        {"raw_forcing": np.ones((9, 9))},
        metadata,
    )
    serialized = json.loads((bundle_path / "bundle.json").read_text(encoding="utf-8"))
    serialized["geometry"]["origin"] = [
        -146_273_875_604_541_230_000,
        -146_273_875_604_541_230_000,
    ]
    (bundle_path / "bundle.json").write_text(json.dumps(serialized), encoding="utf-8")
    loaded, _ = load_array_bundle(bundle_path)
    assert loaded["bundleSha256"] == bundle["bundleSha256"]
    assert loaded["geometry"]["origin"] == [origin, origin]


def test_identical_job_replays_have_identical_scientific_hashes(tmp_path: Path):
    expected, forcing, spacing = manufactured_values()
    bundle_path = tmp_path / "bundle"
    write_array_bundle(
        bundle_path,
        {"raw_forcing": forcing},
        bundle_metadata(spacing),
    )
    request = {
        "schemaVersion": "sigma-field-job-request/1",
        "spacing": [spacing, spacing],
        "boundaryFields": {"u": {"value": 0.0}},
        "requestedObservables": ["gradient"],
        "seed": 1729,
    }
    first = execute_field_job(
        manufactured_manifest(), bundle_path, request, tmp_path / "run_one"
    )
    second = execute_field_job(
        manufactured_manifest(), bundle_path, request, tmp_path / "run_two"
    )
    assert first["jobId"] == second["jobId"]
    assert first["jobSha256"] == second["jobSha256"]
    assert first["scientificResultSha256"] == second["scientificResultSha256"]

    first_result = json.loads(
        (tmp_path / "run_one" / "scientific_result.json").read_text(encoding="utf-8")
    )
    assert first_result["state"] == "succeeded"
    assert first_result["parameterAccounting"]["perObjectCount"] == 0
    with np.load(tmp_path / "run_one" / "fields.npz", allow_pickle=False) as archive:
        relative_error = np.linalg.norm(archive["u"] - expected) / np.linalg.norm(expected)
    assert relative_error < 0.006

    history = (tmp_path / "run_one" / "residual_history.csv").read_text(
        encoding="utf-8"
    )
    assert "maximum_relative_update" in history
    assert len(history.splitlines()) >= 3
    artifact_index = json.loads(
        (tmp_path / "run_one" / "artifact_index.json").read_text(encoding="utf-8")
    )
    artifact_names = {record["path"] for record in artifact_index["artifacts"]}
    assert {
        "job.json",
        "scientific_result.json",
        "fields.npz",
        "observables.npz",
        "residual_history.csv",
        "resource_log.json",
    }.issubset(artifact_names)


def test_axisymmetric_job_binds_coordinates_and_executes_end_to_end(tmp_path: Path):
    cells = 33
    axis = np.linspace(0.0, 1.0, cells)
    spacing = float(axis[1] - axis[0])
    radius, vertical = np.meshgrid(axis, axis, indexing="ij")
    expected = (1.0 - radius**2) * np.sin(np.pi * vertical)
    forcing = -4.0 * np.sin(np.pi * vertical) - np.pi**2 * expected
    model = manufactured_manifest()
    model["geometry"]["coordinateSystem"] = "axisymmetric_cylindrical"
    confirm_manifest(model)
    metadata = bundle_metadata(spacing)
    metadata["geometry"].update(
        {
            "coordinateSystem": "axisymmetric_cylindrical",
            "origin": [0.0, 0.0],
            "axisOrder": ["r", "z"],
            "referenceFrame": "manufactured_axisymmetric_cylinder",
        }
    )
    bundle_path = tmp_path / "axisymmetric_bundle"
    write_array_bundle(bundle_path, {"raw_forcing": forcing}, metadata)
    output_path = tmp_path / "axisymmetric_run"
    run = execute_field_job(
        model,
        bundle_path,
        {
            "schemaVersion": "sigma-field-job-request/1",
            "requestedObservables": ["gradient"],
            "seed": 0,
        },
        output_path,
    )

    job = json.loads((output_path / "job.json").read_text(encoding="utf-8"))
    result = json.loads(
        (output_path / "scientific_result.json").read_text(encoding="utf-8")
    )
    with np.load(output_path / "fields.npz", allow_pickle=False) as archive:
        relative_error = np.linalg.norm(archive["u"] - expected) / np.linalg.norm(
            expected
        )
    with np.load(output_path / "observables.npz", allow_pickle=False) as archive:
        assert np.array_equal(archive["gradient__axis0"][0, :], np.zeros(cells))
    assert run["state"] == "succeeded"
    assert relative_error < 0.003
    assert job["geometry"]["axisOrder"] == ["r", "z"]
    assert job["geometry"]["origin"] == [0.0, 0.0]
    assert job["worker"]["version"] == "1.3.0-preview"
    assert result["numericalMetadata"]["coordinate_system"] == (
        "axisymmetric_cylindrical"
    )
    assert result["numericalMetadata"]["axisymmetric_cylindrical"][
        "axis_boundary"
    ] == "zero_radial_flux_regularity"


def test_axisymmetric_job_scores_a_rotation_curve_without_cartesian_proxy(
    tmp_path: Path,
) -> None:
    cells = 33
    spacing = 0.25
    omega = 3.0
    radius = np.arange(cells, dtype=float) * spacing
    vertical = -0.5 * (cells - 1) * spacing + np.arange(cells) * spacing
    radial_grid, _vertical_grid = np.meshgrid(radius, vertical, indexing="ij")
    expected_potential = 0.5 * omega**2 * radial_grid**2
    forcing = np.full_like(expected_potential, 2.0 * omega**2)
    model = manufactured_manifest()
    model["geometry"]["coordinateSystem"] = "axisymmetric_cylindrical"
    model["observables"][0]["target"] = "massive_tracers"
    model["observables"][0]["expression"] = {
        "op": "negate",
        "args": [{"op": "gradient", "args": [{"field": "u"}]}],
    }
    confirm_manifest(model)
    metadata = bundle_metadata(spacing)
    metadata["geometry"].update(
        {
            "coordinateSystem": "axisymmetric_cylindrical",
            "origin": [0.0, float(vertical[0])],
            "axisOrder": ["r", "z"],
            "referenceFrame": "analytic_axisymmetric_rotation_fixture",
        }
    )
    metadata["arrays"]["potential_boundary"] = {
        "npzKey": "raw_boundary",
        "unit": "m^2/s^2",
        "rank": "scalar",
        "role": "boundary",
    }
    bundle_path = tmp_path / "axisymmetric_rotation_bundle"
    write_array_bundle(
        bundle_path,
        {"raw_forcing": forcing, "raw_boundary": expected_potential},
        metadata,
    )
    radii = [0.375, 0.875, 1.625, 2.375]
    expected_speeds = [omega * value for value in radii]
    output_path = tmp_path / "axisymmetric_rotation_run"
    run = execute_field_job(
        model,
        bundle_path,
        {
            "schemaVersion": "sigma-field-job-request/1",
            "boundaryFields": {"u": {"arrayKey": "potential_boundary"}},
            "requestedObservables": ["gradient"],
            "observationTargets": [
                {
                    "schemaVersion": "sigma-observation-target/1",
                    "id": "axisymmetric-rotation",
                    "kind": "circular_speed_curve",
                    "observable": "gradient",
                    "centerM": [0.0, 0.125],
                    "radiiM": radii,
                    "observedSpeedsMPerS": expected_speeds,
                    "uncertaintiesMPerS": [0.2] * len(radii),
                    "minimumAzimuthalCoverage": 1.0,
                    "provenance": {"kind": "analytic axisymmetric fixture"},
                    "license": {
                        "id": "CC0-1.0",
                        "redistributionAllowed": True,
                    },
                }
            ],
            "seed": 0,
        },
        output_path,
    )

    scores = json.loads(
        (output_path / "observation_scores.json").read_text(encoding="utf-8")
    )
    predictions = (output_path / "observation_predictions.csv").read_text(
        encoding="utf-8"
    )
    target = scores["targets"][0]
    assert run["state"] == "succeeded"
    assert target["coordinateSystem"] == "axisymmetric_cylindrical"
    assert target["samplingMode"] == "axisymmetric_midplane_direct"
    assert target["score"]["rmseMPerS"] < 1e-11
    assert target["score"]["fittedNuisanceParameters"] == 0
    assert "axisymmetric-rotation" in predictions
    assert len(predictions.splitlines()) == len(radii) + 1


def test_published_two_potential_model_survives_the_immutable_job_path(tmp_path: Path):
    model_path = (
        ROOT / "hosted-simulator" / "examples" / "models" / "two-potential.json"
    )
    model = json.loads(model_path.read_text(encoding="utf-8"))
    assert model_sha256(model) == (
        "bcc7c218ec4d11ee77c85837530daa342e98748c3eb04e460b35f93a7e17accc"
    )
    cells = 9
    spacing = 0.5 * 3.085677581491367e19
    coordinates = (np.arange(cells) - cells // 2) * spacing
    x, y, z = np.meshgrid(coordinates, coordinates, coordinates, indexing="ij")
    density = 2.0e-21 * np.exp(
        -(x**2 + y**2 + z**2) / (2.0 * spacing**2)
    )
    bundle_path = tmp_path / "two_potential_bundle"
    write_array_bundle(
        bundle_path,
        {"raw_baryon_density": density},
        {
            "schemaVersion": "sigma-array-bundle-request/1",
            "geometry": {
                "coordinateSystem": "cartesian_3d",
                "dimensions": 3,
                "spacing": [spacing, spacing, spacing],
                "lengthUnit": "m",
                "axisOrder": ["x", "y", "z"],
                "referenceFrame": "synthetic_baryon_test",
            },
            "arrays": {
                "baryon_density": {
                    "npzKey": "raw_baryon_density",
                    "unit": "kg/m^3",
                    "rank": "scalar",
                    "role": "source",
                }
            },
            "provenance": {
                "kind": "analytic_manufactured_source",
                "citation": "repository test fixture",
            },
            "license": {"id": "CC0-1.0", "redistributionAllowed": True},
        },
    )
    output_path = tmp_path / "two_potential_run"
    manifest = execute_field_job(
        model,
        bundle_path,
        {
            "schemaVersion": "sigma-field-job-request/1",
            "spacing": [spacing, spacing, spacing],
            "requestedObservables": [
                "massive_tracer_acceleration",
                "photon_lensing_acceleration",
            ],
            "seed": 20260802,
        },
        output_path,
    )

    assert manifest["state"] == "succeeded"
    with np.load(output_path / "fields.npz", allow_pickle=False) as fields:
        assert np.linalg.norm(fields["Phi"] - 1.5 * fields["Psi"]) / np.linalg.norm(
            fields["Phi"]
        ) < 1e-12
    with np.load(output_path / "observables.npz", allow_pickle=False) as observables:
        for axis_index in range(3):
            matter = observables[f"massive_tracer_acceleration__axis{axis_index}"]
            photons = observables[f"photon_lensing_acceleration__axis{axis_index}"]
            denominator = max(float(np.linalg.norm(photons)), np.finfo(float).tiny)
            assert np.linalg.norm(photons - 1.25 * matter) / denominator < 1e-12
    result = json.loads(
        (output_path / "scientific_result.json").read_text(encoding="utf-8")
    )
    assert result["parameterAccounting"]["perObjectCount"] == 0
    assert result["numericalMetadata"]["solver_family"] == "coupled_elliptic"
    assert result["numericalMetadata"]["equation_count"] == 2
    assert result["numericalMetadata"]["solved_field_count"] == 2


def test_field_job_scores_massive_tracer_curve_after_solving(tmp_path: Path):
    cells = 33
    spacing = 0.25
    axis = np.linspace(-4.0, 4.0, cells)
    x, y = np.meshgrid(axis, axis, indexing="ij")
    omega = 3.0
    expected_potential = 0.5 * omega**2 * (x**2 + y**2)
    forcing = np.full((cells, cells), 2.0 * omega**2)
    bundle_path = tmp_path / "observation_bundle"
    metadata = bundle_metadata(spacing)
    metadata["geometry"]["origin"] = [-4.0, -4.0]
    metadata["arrays"]["u_boundary"] = {
        "npzKey": "raw_boundary",
        "unit": "m^2/s^2",
        "rank": "scalar",
        "role": "boundary",
    }
    write_array_bundle(
        bundle_path,
        {"raw_forcing": forcing, "raw_boundary": expected_potential},
        metadata,
    )
    model = manufactured_manifest()
    model["observables"][0].update(
        {"id": "acceleration", "target": "massive_tracers"}
    )
    model["observables"][0]["expression"] = {
        "op": "negate",
        "args": [{"op": "gradient", "args": [{"field": "u"}]}],
    }
    confirm_manifest(model)
    radii = [0.5, 1.0, 2.0, 3.0]
    manifest = execute_field_job(
        model,
        bundle_path,
        {
            "schemaVersion": "sigma-field-job-request/1",
            "boundaryFields": {"u": {"arrayKey": "u_boundary"}},
            "requestedObservables": ["acceleration"],
            "observationTargets": [
                {
                    "schemaVersion": "sigma-observation-target/1",
                    "id": "solid-body-curve",
                    "kind": "circular_speed_curve",
                    "observable": "acceleration",
                    "centerM": [0.0, 0.0],
                    "planeAxes": [0, 1],
                    "radiiM": radii,
                    "observedSpeedsMPerS": [omega * radius for radius in radii],
                    "uncertaintiesMPerS": [0.1] * len(radii),
                    "minimumAzimuthalCoverage": 1.0,
                    "provenance": {"kind": "analytic solid-body fixture"},
                    "license": {"id": "CC0-1.0", "redistributionAllowed": True},
                }
            ],
        },
        tmp_path / "observation_run",
    )
    assert manifest["state"] == "succeeded"
    result = json.loads(
        (tmp_path / "observation_run" / "scientific_result.json").read_text(
            encoding="utf-8"
        )
    )
    assert result["observationEvaluation"]["scoredTargetCount"] == 1
    assert result["observationEvaluation"]["rmseMPerS"] < 1e-11
    assert (tmp_path / "observation_run" / "observation_predictions.csv").is_file()
    assert (tmp_path / "observation_run" / "observation_scores.json").is_file()


def test_field_job_scores_resolved_velocity_map_after_solving(tmp_path: Path):
    cells = 33
    spacing = 0.25
    axis = np.linspace(-4.0, 4.0, cells)
    x, y = np.meshgrid(axis, axis, indexing="ij")
    omega = 3.0
    expected_potential = 0.5 * omega**2 * (x**2 + y**2)
    forcing = np.full((cells, cells), 2.0 * omega**2)
    map_axis = np.linspace(-2.0, 2.0, 17)
    major, minor = np.meshgrid(map_axis, map_axis, indexing="ij")
    inclination = 60.0
    systemic = 100.0
    observed = omega * major * np.sin(np.radians(inclination)) + systemic
    bundle_path = tmp_path / "velocity_observation_bundle"
    metadata = bundle_metadata(spacing)
    metadata["geometry"]["origin"] = [-4.0, -4.0]
    for key, npz_key, unit, role in [
        ("u_boundary", "raw_boundary", "m^2/s^2", "boundary"),
        ("major", "raw_major", "m", "auxiliary"),
        ("minor", "raw_minor", "m", "auxiliary"),
        ("observed_velocity", "raw_observed", "m/s", "auxiliary"),
        ("velocity_uncertainty", "raw_uncertainty", "m/s", "uncertainty"),
        ("valid_mask", "raw_mask", "1", "mask"),
    ]:
        metadata["arrays"][key] = {
            "npzKey": npz_key,
            "unit": unit,
            "rank": "scalar",
            "role": role,
        }
    write_array_bundle(
        bundle_path,
        {
            "raw_forcing": forcing,
            "raw_boundary": expected_potential,
            "raw_major": major,
            "raw_minor": minor,
            "raw_observed": observed,
            "raw_uncertainty": np.full_like(major, 0.2),
            "raw_mask": np.ones_like(major),
        },
        metadata,
    )
    model = manufactured_manifest()
    model["observables"][0].update(
        {"id": "acceleration", "target": "massive_tracers"}
    )
    model["observables"][0]["expression"] = {
        "op": "negate",
        "args": [{"op": "gradient", "args": [{"field": "u"}]}],
    }
    confirm_manifest(model)
    manifest = execute_field_job(
        model,
        bundle_path,
        {
            "schemaVersion": "sigma-field-job-request/1",
            "boundaryFields": {"u": {"arrayKey": "u_boundary"}},
            "requestedObservables": ["acceleration"],
            "observationTargets": [
                {
                    "schemaVersion": "sigma-observation-target/1",
                    "id": "solid-body-velocity-map",
                    "kind": "line_of_sight_velocity_field",
                    "observable": "acceleration",
                    "centerM": [0.0, 0.0],
                    "inclinationDeg": inclination,
                    "handedness": 1,
                    "majorCoordinateArrayKey": "major",
                    "minorCoordinateArrayKey": "minor",
                    "observedVelocityArrayKey": "observed_velocity",
                    "uncertaintyArrayKey": "velocity_uncertainty",
                    "observedVelocityZeroPointMPerS": systemic,
                    "maskArrayKey": "valid_mask",
                    "minimumValidPixels": 200,
                    "provenance": {"kind": "analytic resolved velocity fixture"},
                    "license": {"id": "CC0-1.0", "redistributionAllowed": True},
                }
            ],
        },
        tmp_path / "velocity_observation_run",
    )
    assert manifest["state"] == "succeeded"
    result = json.loads(
        (
            tmp_path
            / "velocity_observation_run"
            / "scientific_result.json"
        ).read_text(encoding="utf-8")
    )
    assert result["observationEvaluation"]["targetKinds"] == [
        "line_of_sight_velocity_field"
    ]
    assert result["observationEvaluation"]["validScoredPoints"] == 17 * 17 - 1
    assert result["observationEvaluation"]["rmseMPerS"] < 1e-11
    assert (
        tmp_path
        / "velocity_observation_run"
        / "observation_velocity_field_predictions.csv"
    ).is_file()
    assert not (
        tmp_path / "velocity_observation_run" / "observation_predictions.csv"
    ).exists()


def test_cli_envelope_resolves_relative_paths_and_output_is_immutable(tmp_path: Path):
    _expected, forcing, spacing = manufactured_values(17)
    (tmp_path / "model.json").write_text(
        json.dumps(manufactured_manifest()), encoding="utf-8"
    )
    write_array_bundle(
        tmp_path / "bundle",
        {"raw_forcing": forcing},
        bundle_metadata(spacing),
    )
    envelope = {
        "schemaVersion": "sigma-field-job-cli/1",
        "modelPath": "model.json",
        "inputBundlePath": "bundle",
        "outputDirectory": "run",
        "request": {
            "schemaVersion": "sigma-field-job-request/1",
            "requestedObservables": ["gradient"],
            "seed": 11,
        },
    }
    request_path = tmp_path / "request.json"
    request_path.write_text(json.dumps(envelope), encoding="utf-8")
    manifest = execute_request_file(request_path)
    assert manifest["jobId"].startswith("fieldjob_")
    assert (tmp_path / "run" / "manifest.json").is_file()
    with pytest.raises(FileExistsError, match="immutable output directory"):
        execute_request_file(request_path)


def test_nonconvergence_is_retained_as_a_failed_scientific_result(tmp_path: Path):
    _expected, forcing, spacing = manufactured_values(17)
    bundle_path = tmp_path / "bundle"
    write_array_bundle(
        bundle_path,
        {"raw_forcing": forcing},
        bundle_metadata(spacing),
    )
    model = manufactured_manifest()
    model["solver"]["maxIterations"] = 1
    confirm_manifest(model)
    execute_field_job(
        model,
        bundle_path,
        {
            "schemaVersion": "sigma-field-job-request/1",
            "requestedObservables": ["gradient"],
        },
        tmp_path / "run",
    )
    result = json.loads(
        (tmp_path / "run" / "scientific_result.json").read_text(encoding="utf-8")
    )
    assert result["converged"] is False
    assert result["state"] == "failed_nonconvergence"
    assert result["iterations"] == 1
    assert (tmp_path / "run" / "residual_history.csv").is_file()


def test_runtime_exception_keeps_a_hashed_failure_artifact(tmp_path: Path):
    _expected, forcing, spacing = manufactured_values(17)
    bundle_path = tmp_path / "bundle"
    write_array_bundle(
        bundle_path,
        {"raw_forcing": forcing},
        bundle_metadata(spacing),
    )
    model = manufactured_manifest()
    model["equations"][0]["rhs"] = {
        "op": "line_of_sight_integral",
        "args": [{"field": "forcing"}],
    }
    confirm_manifest(model)
    manifest = execute_field_job(
        model,
        bundle_path,
        {
            "schemaVersion": "sigma-field-job-request/1",
            "requestedObservables": ["gradient"],
        },
        tmp_path / "run",
    )
    failure = json.loads(
        (tmp_path / "run" / "failure.json").read_text(encoding="utf-8")
    )
    assert manifest["state"] == "failed"
    assert manifest["failureSha256"] == failure["failureSha256"]
    assert failure["errorType"] == "ValueError"
    assert "not executable" in failure["message"]
    assert not (tmp_path / "run" / "scientific_result.json").exists()


def test_cli_envelope_cannot_escape_its_request_directory(tmp_path: Path):
    request_path = tmp_path / "request.json"
    request_path.write_text(
        json.dumps(
            {
                "schemaVersion": "sigma-field-job-cli/1",
                "modelPath": "../outside-model.json",
                "inputBundlePath": "bundle",
                "outputDirectory": "run",
                "request": {"schemaVersion": "sigma-field-job-request/1"},
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="must remain inside"):
        execute_request_file(request_path)
