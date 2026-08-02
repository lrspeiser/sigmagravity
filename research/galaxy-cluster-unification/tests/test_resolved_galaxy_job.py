from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from voidscreen.field_job import file_sha256, load_array_bundle, write_array_bundle
from voidscreen.resolved_galaxy_job import execute_galaxy_request_file


def input_bundle(root: Path) -> Path:
    axis = np.linspace(-4.0, 4.0, 33)
    xx, yy = np.meshgrid(axis, axis, indexing="ij")
    radius = np.hypot(xx - 0.2, yy + 0.1)
    gas = 3.0e6 * np.exp(-radius / 1.8) * (
        1.0 + 0.12 * np.cos(np.arctan2(yy, xx))
    )
    stars = 6.0e6 * np.exp(-radius / 1.1)
    destination = root / "bundle"
    write_array_bundle(
        destination,
        {"raw_gas": gas, "raw_stars": stars},
        {
            "schemaVersion": "sigma-array-bundle-request/1",
            "geometry": {
                "coordinateSystem": "cartesian_2d",
                "dimensions": 2,
                "spacing": [0.25, 0.25],
                "lengthUnit": "kpc",
                "axisOrder": ["x", "y"],
                "referenceFrame": "synthetic_face_on",
            },
            "arrays": {
                "gas_surface_density": {
                    "npzKey": "raw_gas",
                    "unit": "M_sun/kpc^2",
                    "rank": "scalar",
                    "role": "source",
                },
                "stellar_surface_density": {
                    "npzKey": "raw_stars",
                    "unit": "M_sun/kpc^2",
                    "rank": "scalar",
                    "role": "source",
                },
            },
            "provenance": {"kind": "synthetic_test_fixture"},
            "license": {"id": "CC0-1.0", "redistributionAllowed": True},
        },
    )
    return destination


def extraction_request(root: Path) -> Path:
    input_bundle(root)
    request = {
        "schemaVersion": "sigma-galaxy-job-cli/1",
        "operation": "extract_roundtrip",
        "inputBundlePath": "bundle",
        "outputDirectory": "artifacts",
        "galaxy": "SYNTHETIC-33",
        "sourceObservables": {"kind": "known-answer fixture"},
        "extractionControls": {
            "radialBins": 12,
            "maximumFourierMode": 3,
            "residualFeatureCountPerComponent": 8,
        },
        "vertical": {"enabled": True, "realizations": 2, "zCells": 17, "seed": 42},
        "outputLicense": {"id": "CC0-1.0", "redistributionAllowed": True},
    }
    path = root / "request.json"
    path.write_text(json.dumps(request), encoding="utf-8")
    return path


def test_extract_job_emits_verified_2d_and_3d_bundles(tmp_path: Path) -> None:
    manifest = execute_galaxy_request_file(extraction_request(tmp_path))
    output = tmp_path / "artifacts"
    assert manifest["state"] == "succeeded"
    assert manifest["formulaIndependence"]["gravityParameters"] == 0
    assert manifest["formulaIndependence"]["velocityTargetsUsedForExtraction"] is False
    index = json.loads((output / "artifact_index.json").read_text(encoding="utf-8"))
    for record in index["artifacts"]:
        artifact = output / record["path"]
        assert artifact.stat().st_size == record["bytes"]
        assert file_sha256(artifact) == record["sha256"]
    surface_bundle, surface_arrays = load_array_bundle_from_artifacts(output, "surface_density")
    volume_bundle, volume_arrays = load_array_bundle_from_artifacts(output, "volume_density")
    assert surface_bundle["geometry"]["dimensions"] == 2
    assert volume_bundle["geometry"]["dimensions"] == 3
    dz = volume_bundle["geometry"]["spacing"][2]
    assert np.allclose(
        volume_arrays["total_baryonic_volume_density"].sum(axis=2) * dz,
        surface_arrays["total_baryonic_surface_density"],
        rtol=1e-12,
        atol=1e-7,
    )
    metrics = json.loads((output / "roundtrip_metrics.json").read_text(encoding="utf-8"))
    assert metrics["total"]["mass_relative_error"] < 1e-12
    assert metrics["total"]["pixel_correlation"] > 0.95


def load_array_bundle_from_artifacts(
    output: Path, stem: str
) -> tuple[dict, dict[str, np.ndarray]]:
    directory = output / f"_{stem}_bundle"
    directory.mkdir()
    (directory / "bundle.json").write_bytes((output / f"{stem}_bundle.json").read_bytes())
    (directory / "arrays.npz").write_bytes((output / f"{stem}.npz").read_bytes())
    return load_array_bundle(directory)


def test_generate_job_replays_a_parameter_package_without_source_data(tmp_path: Path) -> None:
    execute_galaxy_request_file(extraction_request(tmp_path))
    package = json.loads((tmp_path / "artifacts" / "parameters.json").read_text(encoding="utf-8"))
    generation_root = tmp_path / "generation"
    generation_root.mkdir()
    request = {
        "schemaVersion": "sigma-galaxy-job-cli/1",
        "operation": "generate",
        "outputDirectory": "artifacts",
        "parameterPackage": package,
        "generationControls": {
            "gas": {"mass_scale": 1.5, "radial_scale": 0.8},
            "stars": {"fourier_scale": 0.0, "residual_scale": 0.0},
        },
        "vertical": {"enabled": False, "realizations": 1, "zCells": 17, "seed": 7},
        "outputLicense": {"id": "CC0-1.0", "redistributionAllowed": True},
    }
    request_path = generation_root / "request.json"
    request_path.write_text(json.dumps(request), encoding="utf-8")
    manifest = execute_galaxy_request_file(request_path)
    output = generation_root / "artifacts"
    result = json.loads((output / "scientific_result.json").read_text(encoding="utf-8"))
    assert manifest["state"] == "succeeded"
    assert result["sourceBundleSha256"] is None
    assert result["roundtripMetrics"] is None
    assert result["parameterAccounting"]["gravityPerObject"] == 0
    assert not (output / "volume_density.npz").exists()
    assert (output / "surface_density.npz").exists()


def test_job_rejects_non_kpc_input_geometry(tmp_path: Path) -> None:
    request_path = extraction_request(tmp_path)
    bundle_path = tmp_path / "bundle" / "bundle.json"
    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    bundle["geometry"]["lengthUnit"] = "m"
    core = {key: value for key, value in bundle.items() if key != "bundleSha256"}
    from voidscreen.field_job import canonical_sha256

    bundle["bundleSha256"] = canonical_sha256(core)
    bundle_path.write_text(json.dumps(bundle), encoding="utf-8")
    try:
        execute_galaxy_request_file(request_path)
    except ValueError as error:
        assert "lengthUnit kpc" in str(error)
    else:
        raise AssertionError("non-kpc geometry was accepted")
