from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from voidscreen.field_job import (
    execute_field_job,
    execute_request_file,
    load_array_bundle,
    model_sha256,
    write_array_bundle,
)

ROOT = Path(__file__).resolve().parents[1]


def manufactured_manifest() -> dict:
    return {
        "schemaVersion": "sigma-field-model/1",
        "name": "Content-addressed manufactured field",
        "modelClass": "stationary_elliptic",
        "source": {
            "format": "plain_text",
            "text": "laplacian(u) = forcing",
            "confirmedCanonical": True,
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
        "op": "convolution",
        "args": [{"field": "forcing"}, {"field": "forcing"}],
    }
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
