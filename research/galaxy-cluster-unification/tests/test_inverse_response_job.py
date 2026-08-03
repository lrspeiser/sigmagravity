from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from voidscreen.field_job import canonical_sha256, file_sha256, write_array_bundle
from voidscreen.inverse_response import convolve_stationary_response
from voidscreen.inverse_response_job import execute_inverse_response_request_file


def build_bundle(root: Path, *, target_role: str = "model_derived_discovery_target") -> Path:
    cells = 15
    axis = np.linspace(-1.0, 1.0, cells)
    x, y = np.meshgrid(axis, axis, indexing="ij")
    kernel_axis = np.arange(-2.0, 3.0)
    kx, ky = np.meshgrid(kernel_axis, kernel_axis, indexing="ij")
    kernel = np.exp(-((kx - 0.4) ** 2 / 2.0 + (ky + 0.3) ** 2 / 3.0))
    kernel /= np.sum(kernel)
    arrays: dict[str, np.ndarray] = {}
    descriptions: dict[str, dict] = {}
    for index, phase in enumerate((0.2, 1.3), start=1):
        source = (
            np.exp(-((x - 0.25 * index) ** 2 / 0.11 + (y + 0.2) ** 2 / 0.18))
            + 0.65
            * np.exp(-((x + 0.3) ** 2 / 0.2 + (y - 0.2 * index) ** 2 / 0.07))
            + 0.1 * np.cos(3.0 * np.arctan2(y, x) + phase) ** 2
        )
        target = 1.6 * convolve_stationary_response(source, kernel, 1.0)
        uncertainty = np.full_like(target, 0.01)
        for label, values, role in (
            ("baryons", source, "baryonic_input"),
            ("response", target, target_role),
            ("uncertainty", uncertainty, "nuisance_or_calibration"),
        ):
            key = f"{label}_{index}"
            arrays[key] = values
            descriptions[key] = {
                "unit": "kg/m^2",
                "rank": "scalar",
                "role": (
                    "source"
                    if label == "baryons"
                    else "uncertainty"
                    if label == "uncertainty"
                    else "auxiliary"
                ),
                "scientificRole": role,
            }
    destination = root / "bundle"
    write_array_bundle(
        destination,
        arrays,
        {
            "schemaVersion": "sigma-array-bundle-request/1",
            "geometry": {
                "coordinateSystem": "cartesian_2d",
                "dimensions": 2,
                "spacing": [1.0, 1.0],
                "lengthUnit": "kpc",
                "axisOrder": ["x", "y"],
                "referenceFrame": "synthetic_inverse_fixture",
            },
            "arrays": descriptions,
            "provenance": {"kind": "synthetic_injected_kernel_fixture"},
            "license": {"id": "CC0-1.0", "redistributionAllowed": True},
        },
    )
    return destination


def write_request(root: Path) -> Path:
    request = {
        "schemaVersion": "sigma-inverse-response-job-cli/1",
        "inputBundlePath": "bundle",
        "outputDirectory": "artifacts",
        "systems": [
            {
                "id": f"SYNTH-{index}",
                "sourceKey": f"baryons_{index}",
                "targetKey": f"response_{index}",
                "uncertaintyKey": f"uncertainty_{index}",
            }
            for index in (1, 2)
        ],
        "kernel": {
            "shape": [5, 5],
            "ridge": 1.0e-10,
            "smoothness": 1.0e-8,
            "nonnegative": True,
            "regularizationMultipliers": [0.1, 1.0, 10.0],
        },
        "uncertainty": {"ensembleSize": 20, "seed": 17},
        "nullControls": {
            "kind": "source_radial_angle_shuffle",
            "count": 19,
            "seed": 23,
        },
        "outputLicense": {"id": "CC-BY-4.0", "redistributionAllowed": True},
    }
    path = root / "request.json"
    path.write_text(json.dumps(request), encoding="utf-8")
    return path


def test_inverse_job_emits_hashed_discovery_artifacts_and_role_audit(tmp_path: Path) -> None:
    build_bundle(tmp_path)
    request_path = write_request(tmp_path)
    manifest = execute_inverse_response_request_file(request_path)
    output = tmp_path / "artifacts"
    assert manifest["state"] == "succeeded"
    result = json.loads((output / "scientific_result.json").read_text(encoding="utf-8"))
    result_core = {key: value for key, value in result.items() if key != "resultSha256"}
    assert canonical_sha256(result_core) == result["resultSha256"]
    assert result["parameterAccounting"]["fittedPerSystemGravityParameters"] == 0
    assert result["parameterAccounting"]["classification"] == (
        "hypothesis_generator_not_forward_theory_fit"
    )
    assert all(
        row["targetRole"] == "model_derived_discovery_target"
        and row["heldOutRawObservationsUsed"] is False
        for row in result["dataRoleAudit"]
    )
    assert result["aggregateMetrics"]["r_squared"] > 0.999
    assert result["nullSummary"]["signal_against_null"] is True
    with np.load(output / "kernels.npz") as archive:
        assert set(archive.files) == {
            "raw_response",
            "normalized",
            "lower_2_5",
            "median",
            "upper_97_5",
        }
        assert np.isclose(np.sum(archive["normalized"]), 1.0, rtol=1e-10)
    index = json.loads((output / "artifact_index.json").read_text(encoding="utf-8"))
    artifact_paths = {record["path"] for record in index["artifacts"]}
    assert {
        "report.html",
        "llm_briefing.md",
        "kernel.csv",
        "null_controls.csv",
        "system_predictions.npz",
        "reproduction.txt",
    } <= artifact_paths
    for record in index["artifacts"]:
        artifact = output / record["path"]
        assert artifact.stat().st_size == record["bytes"]
        assert file_sha256(artifact) == record["sha256"]
    report = (output / "report.html").read_text(encoding="utf-8")
    briefing = (output / "llm_briefing.md").read_text(encoding="utf-8")
    assert "hypothesis generator, not a forward theory test" in report
    assert "must not change scores" in briefing


def test_inverse_job_scientific_result_is_deterministic(tmp_path: Path) -> None:
    build_bundle(tmp_path)
    request_path = write_request(tmp_path)
    first = execute_inverse_response_request_file(request_path)
    second = execute_inverse_response_request_file(
        request_path, tmp_path / "second-artifacts"
    )
    assert first["jobId"] == second["jobId"]
    assert first["scientificResultSha256"] == second["scientificResultSha256"]
    assert (tmp_path / "artifacts" / "scientific_result.json").read_bytes() == (
        tmp_path / "second-artifacts" / "scientific_result.json"
    ).read_bytes()


def test_inverse_job_reports_every_declared_null_family(tmp_path: Path) -> None:
    build_bundle(tmp_path)
    request_path = write_request(tmp_path)
    request = json.loads(request_path.read_text(encoding="utf-8"))
    request["nullControls"] = {
        "combinationRule": "all_declared_families",
        "families": [
            {"kind": "source_phase_scramble", "count": 19, "seed": 41},
            {"kind": "target_system_permutation", "count": 19, "seed": 42},
            {
                "kind": "source_missing_baryon_dropout",
                "count": 19,
                "seed": 43,
                "dropoutFraction": 0.2,
            },
        ],
    }
    request_path.write_text(json.dumps(request), encoding="utf-8")
    execute_inverse_response_request_file(request_path)
    output = tmp_path / "artifacts"
    result = json.loads((output / "scientific_result.json").read_text(encoding="utf-8"))
    assert result["nullSummary"]["combination_rule"] == "all_declared_families"
    assert result["nullSummary"]["family_count"] == 3
    assert result["nullSummary"]["total_count"] == 57
    assert result["nullSummary"]["signal_against_null"] is True
    assert all(
        row["signal_against_null"] for row in result["nullSummary"]["families"]
    )
    csv_text = (output / "null_controls.csv").read_text(encoding="utf-8")
    report = (output / "report.html").read_text(encoding="utf-8")
    briefing = (output / "llm_briefing.md").read_text(encoding="utf-8")
    assert "dropout_fraction" in csv_text
    assert "source_phase_scramble" in report
    assert "target_system_permutation" in briefing


def test_inverse_job_rejects_raw_observation_as_discovery_target(tmp_path: Path) -> None:
    build_bundle(tmp_path, target_role="raw_observation")
    request_path = write_request(tmp_path)
    with pytest.raises(
        ValueError,
        match="must declare scientificRole=model_derived_discovery_target",
    ):
        execute_inverse_response_request_file(request_path)
    assert not (tmp_path / "artifacts").exists()


def test_inverse_job_rejects_nonboolean_kernel_constraint(tmp_path: Path) -> None:
    build_bundle(tmp_path)
    request_path = write_request(tmp_path)
    request = json.loads(request_path.read_text(encoding="utf-8"))
    request["kernel"]["nonnegative"] = "false"
    request_path.write_text(json.dumps(request), encoding="utf-8")
    with pytest.raises(TypeError, match="kernel.nonnegative must be a boolean"):
        execute_inverse_response_request_file(request_path)
    assert not (tmp_path / "artifacts").exists()


def test_inverse_job_rejects_request_path_escape(tmp_path: Path) -> None:
    request_path = tmp_path / "request.json"
    request_path.write_text(
        json.dumps(
            {
                "schemaVersion": "sigma-inverse-response-job-cli/1",
                "inputBundlePath": "../outside",
                "outputDirectory": "artifacts",
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="must remain inside"):
        execute_inverse_response_request_file(request_path)
