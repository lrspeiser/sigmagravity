from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from voidscreen.field_job import file_sha256, load_array_bundle, write_array_bundle
from voidscreen.resolved_galaxy_job import (
    _output_axis,
    _vertical_products,
    execute_galaxy_request_file,
)


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
    field_volume_bundle, field_volume_arrays = load_array_bundle_from_artifacts(
        output, "field_volume_density"
    )
    assert surface_bundle["geometry"]["dimensions"] == 2
    assert volume_bundle["geometry"]["dimensions"] == 3
    assert field_volume_bundle["geometry"]["lengthUnit"] == "m"
    assert len(field_volume_bundle["geometry"]["origin"]) == 3
    assert {
        record["key"]: record["unit"] for record in field_volume_bundle["arrays"]
    }["baryon_density"] == "kg/m^3"
    dz = volume_bundle["geometry"]["spacing"][2]
    assert np.allclose(
        volume_arrays["total_baryonic_volume_density"].sum(axis=2) * dz,
        surface_arrays["total_baryonic_surface_density"],
        rtol=1e-12,
        atol=1e-7,
    )
    assert np.allclose(
        field_volume_arrays["baryon_density"],
        volume_arrays["total_baryonic_volume_density"]
        * 1.98847e30
        / 3.085677581491367e19**3,
        rtol=1e-13,
    )
    metrics = json.loads((output / "roundtrip_metrics.json").read_text(encoding="utf-8"))
    assert metrics["total"]["mass_relative_error"] < 1e-12
    assert metrics["total"]["pixel_correlation"] > 0.95
    with np.load(output / "surface_density_ensemble.npz") as ensemble:
        assert ensemble["total_baryonic_surface_density"].shape == (1, 33, 33)
        np.testing.assert_array_equal(
            ensemble["total_baryonic_surface_density"][0],
            surface_arrays["total_baryonic_surface_density"],
        )
    with np.load(output / "volume_density_ensemble.npz") as ensemble:
        assert ensemble["total_baryonic_volume_density"].shape == (1, 2, 33, 33, 17)
        np.testing.assert_array_equal(
            ensemble["total_baryonic_volume_density"][0, 0],
            volume_arrays["total_baryonic_volume_density"],
        )


def test_vertical_prior_uses_disclosed_resolution_floor_for_central_pixel() -> None:
    axis = np.linspace(-16.0, 16.0, 33)
    gas = np.zeros((33, 33), dtype=float)
    stars = np.zeros_like(gas)
    gas[16, 16] = 2.0
    stars[16, 16] = 1.0
    volume, metadata, z_axis = _vertical_products(
        {"gas": gas, "stars": stars, "total": gas + stars},
        axis,
        {"enabled": True, "realizations": 1, "zCells": 9, "seed": 7},
    )
    assert volume is not None
    assert z_axis is not None
    assert all(item["measuredR80Kpc"] == 0.0 for item in metadata)
    assert all(item["r80ResolutionFloorApplied"] is True for item in metadata)
    assert all(item["r80ResolutionFloorKpc"] == 1.0 for item in metadata)
    dz = float(z_axis[1] - z_axis[0])
    np.testing.assert_allclose(np.sum(volume["gas"], axis=2) * dz, gas)
    np.testing.assert_allclose(np.sum(volume["stars"], axis=2) * dz, stars)


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
        "outputGrid": {"cellsPerAxis": 25},
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
    assert (output / "field_surface_density.npz").exists()
    with np.load(output / "surface_density.npz") as generated:
        assert generated["total_baryonic_surface_density"].shape == (25, 25)


def test_output_grid_can_expand_box_without_changing_parameter_package(tmp_path: Path) -> None:
    execute_galaxy_request_file(extraction_request(tmp_path))
    package = json.loads((tmp_path / "artifacts" / "parameters.json").read_text(encoding="utf-8"))
    source = _output_axis(package, None)
    expanded = _output_axis(package, {"cellsPerAxis": 49, "extentScale": 1.5})
    assert expanded.shape == (49,)
    assert np.isclose(0.5 * (expanded[0] + expanded[-1]), 0.5 * (source[0] + source[-1]))
    assert np.isclose(expanded[-1] - expanded[0], 1.5 * (source[-1] - source[0]))


def test_generate_job_rejects_fractional_output_grid(tmp_path: Path) -> None:
    execute_galaxy_request_file(extraction_request(tmp_path))
    package = json.loads((tmp_path / "artifacts" / "parameters.json").read_text(encoding="utf-8"))
    generation_root = tmp_path / "generation"
    generation_root.mkdir()
    request = {
        "schemaVersion": "sigma-galaxy-job-cli/1",
        "operation": "generate",
        "outputDirectory": "artifacts",
        "parameterPackage": package,
        "generationControls": {},
        "outputGrid": {"cellsPerAxis": 25.5},
        "vertical": {"enabled": False, "realizations": 1, "zCells": 17, "seed": 7},
        "outputLicense": {"id": "CC0-1.0", "redistributionAllowed": True},
    }
    request_path = generation_root / "request.json"
    request_path.write_text(json.dumps(request), encoding="utf-8")
    try:
        execute_galaxy_request_file(request_path)
    except TypeError as error:
        assert "must be an integer" in str(error)
    else:
        raise AssertionError("fractional output grid was accepted")


def test_job_rejects_fractional_uncertainty_realization_count(tmp_path: Path) -> None:
    request_path = extraction_request(tmp_path)
    request = json.loads(request_path.read_text(encoding="utf-8"))
    request["uncertaintyEnsemble"] = {"enabled": True, "realizations": 2.5}
    request_path.write_text(json.dumps(request), encoding="utf-8")
    try:
        execute_galaxy_request_file(request_path)
    except TypeError as error:
        assert "must be an integer" in str(error)
    else:
        raise AssertionError("fractional uncertainty ensemble size was accepted")


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


def test_uncertainty_ensemble_is_seeded_bounded_and_projects_exactly(tmp_path: Path) -> None:
    request_path = extraction_request(tmp_path)
    request = json.loads(request_path.read_text(encoding="utf-8"))
    request["sourceObservables"] = {"inclinationDeg": 48.0}
    request["uncertaintyEnsemble"] = {
        "enabled": True,
        "realizations": 4,
        "seed": 91,
        "priors": {
            "gas_mass_ln_sigma": 0.12,
            "stellar_mass_ln_sigma": 0.18,
            "gas_radial_scale_ln_sigma": 0.08,
            "stellar_radial_scale_ln_sigma": 0.1,
            "angular_structure_ln_sigma": 0.15,
            "local_structure_ln_sigma": 0.2,
            "center_sigma_kpc": 0.08,
            "rotation_sigma_deg": 4.0,
            "distance_scale_ln_sigma": 0.04,
            "inclination_sigma_deg": 3.0,
            "warp_sigma_deg": 2.0,
            "co_spatial_unseen_baryon_fraction_max": 0.1,
        },
    }
    request["vertical"] = {"enabled": True, "realizations": 2, "zCells": 17, "seed": 42}
    request_path.write_text(json.dumps(request), encoding="utf-8")
    execute_galaxy_request_file(request_path)
    output = tmp_path / "artifacts"
    with np.load(output / "surface_density_ensemble.npz") as saved:
        surfaces = {key: saved[key].copy() for key in saved.files}
    with np.load(output / "volume_density_ensemble.npz") as saved:
        volumes = {key: saved[key].copy() for key in saved.files}
    assert surfaces["total_baryonic_surface_density"].shape == (4, 33, 33)
    assert volumes["total_baryonic_volume_density"].shape == (4, 2, 33, 33, 17)
    assert not np.array_equal(
        surfaces["total_baryonic_surface_density"][0],
        surfaces["total_baryonic_surface_density"][1],
    )
    dz = 2.0 * max(8.0 * 0.25, 0.8 * max(
        resolved["morphology"]["total"]["r80_kpc"]
        for resolved in json.loads(
            (output / "baryonic_uncertainty_ensemble.json").read_text(encoding="utf-8")
        )["draws"]
    )) / 16.0
    np.testing.assert_allclose(
        volumes["total_baryonic_volume_density"].sum(axis=4) * dz,
        np.broadcast_to(
            surfaces["total_baryonic_surface_density"][:, None, :, :],
            volumes["total_baryonic_volume_density"].shape[:4],
        ),
        rtol=1e-12,
        atol=1e-7,
    )
    with np.load(output / "surface_density_quantiles.npz") as quantiles:
        for prefix in (
            "gas_surface_density",
            "stellar_surface_density",
            "total_baryonic_surface_density",
        ):
            assert np.all(quantiles[f"{prefix}_p16"] <= quantiles[f"{prefix}_p50"])
            assert np.all(quantiles[f"{prefix}_p50"] <= quantiles[f"{prefix}_p84"])
    scientific = json.loads((output / "scientific_result.json").read_text(encoding="utf-8"))
    assert scientific["uncertaintyEnsemble"]["surfaceRealizations"] == 4
    assert scientific["uncertaintyEnsemble"]["verticalRealizationsPerSurface"] == 2
    assert scientific["parameterAccounting"]["gravityUniversal"] == 0


def test_uncertainty_ensemble_seed_replays_without_velocity_targets(tmp_path: Path) -> None:
    arrays = []
    scientific_ids = []
    for index in range(2):
        root = tmp_path / f"run-{index}"
        root.mkdir()
        request_path = extraction_request(root)
        request = json.loads(request_path.read_text(encoding="utf-8"))
        request["sourceObservables"] = {
            "kind": "known-answer fixture",
            "withheldVelocityTargetChecksum": f"not-consumed-{index}",
        }
        request["uncertaintyEnsemble"] = {
            "enabled": True,
            "realizations": 3,
            "seed": 712,
            "priors": {"gas_mass_ln_sigma": 0.2, "stellar_mass_ln_sigma": 0.2},
        }
        request_path.write_text(json.dumps(request), encoding="utf-8")
        manifest = execute_galaxy_request_file(request_path)
        scientific_ids.append(manifest["jobId"])
        with np.load(root / "artifacts" / "surface_density_ensemble.npz") as saved:
            arrays.append(saved["total_baryonic_surface_density"].copy())
        package = json.loads((root / "artifacts" / "parameters.json").read_text(encoding="utf-8"))
        assert package["velocityTargetsUsed"] is False
        assert package["gravityParameters"] == {}
    assert scientific_ids[0] != scientific_ids[1]
    np.testing.assert_array_equal(arrays[0], arrays[1])
