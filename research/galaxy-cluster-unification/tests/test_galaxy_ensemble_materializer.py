from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from voidscreen.field_job import array_content_sha256, canonical_sha256, load_array_bundle
from voidscreen.galaxy_ensemble_materializer import (
    KPC_M,
    MSUN_KG,
    materialize_galaxy_ensemble_realization,
)


def _write_ensemble(root: Path, *, volume: bool) -> tuple[Path, Path, str]:
    artifact = "volume_density_ensemble" if volume else "surface_density_ensemble"
    leading = (2, 2) if volume else (2,)
    spatial = (5, 5, 5) if volume else (5, 5)
    names = (
        ["gas_volume_density", "stellar_volume_density", "total_baryonic_volume_density"]
        if volume
        else ["gas_surface_density", "stellar_surface_density", "total_baryonic_surface_density"]
    )
    arrays = {
        name: np.ascontiguousarray(
            np.arange(np.prod(leading + spatial), dtype="<f8").reshape(leading + spatial)
            + index,
            dtype="<f8",
        )
        for index, name in enumerate(names, start=1)
    }
    axes = [{"name": "surfaceRealization", "count": 2, "anchorIndex": 0}]
    if volume:
        axes.append({"name": "verticalRealization", "count": 2, "anchorIndex": 0})
    unit = "M_sun/kpc^3" if volume else "M_sun/kpc^2"
    geometry = {
        "coordinateSystem": "cartesian_3d" if volume else "cartesian_2d",
        "dimensions": 3 if volume else 2,
        "spacing": [0.5] * (3 if volume else 2),
        "origin": [-1.0] * (3 if volume else 2),
        "lengthUnit": "kpc",
        "axisOrder": ["x", "y", "z"] if volume else ["x", "y"],
        "referenceFrame": "test",
    }
    records = [
        {
            "key": name,
            "npzKey": name,
            "unit": unit,
            "rank": "scalar_ensemble",
            "shape": list(values.shape),
            "elementCount": int(values.size),
            "contentSha256": array_content_sha256(values),
        }
        for name, values in sorted(arrays.items())
    ]
    weights_core = {
        "schemaVersion": "sigma-baryonic-surface-weights/1",
        "status": "baryonic_surface_likelihood_conditioned_partial_posterior",
        "weights": [0.8, 0.2],
    }
    core = {
        "schemaVersion": "sigma-galaxy-density-ensemble/1",
        "spatialGeometry": geometry,
        "ensembleAxes": axes,
        "arrays": records,
        "provenance": {
            "kind": "unit_test",
            "uncertaintyStatus": (
                "baryonic_surface_likelihood_conditioned_partial_posterior"
            ),
            "conditioning": {
                "status": weights_core["status"],
                "weightsSha256": canonical_sha256(weights_core),
                "surfaceWeights": weights_core["weights"],
                "surfaceLikelihoodConditioned": True,
                "verticalStructureConditioned": False,
                "effectiveSampleSize": 1.0 / 0.68,
                "normalizedEffectiveSampleSize": 1.0 / 1.36,
                "weightQualityStatus": "adequate_for_commissioning_only",
                "credibleIntervalReady": False,
            },
        },
        "license": {"id": "CC0-1.0", "redistributionAllowed": True},
    }
    bundle = {**core, "bundleSha256": canonical_sha256(core)}
    bundle_path = root / f"{artifact}_bundle.json"
    archive_path = root / f"{artifact}.npz"
    bundle_path.write_text(json.dumps(bundle), encoding="utf-8")
    with archive_path.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    return bundle_path, archive_path, artifact


@pytest.mark.parametrize("volume", [False, True])
def test_materializes_verified_si_realization_deterministically(tmp_path: Path, volume: bool) -> None:
    bundle_path, archive_path, artifact = _write_ensemble(tmp_path, volume=volume)
    selection = {"surfaceRealization": 1}
    if volume:
        selection["verticalRealization"] = 0
    first = tmp_path / "first"
    second = tmp_path / "second"
    first_bundle = materialize_galaxy_ensemble_realization(
        bundle_path=bundle_path,
        archive_path=archive_path,
        artifact=artifact,
        selection=selection,
        output_directory=first,
    )
    second_bundle = materialize_galaxy_ensemble_realization(
        bundle_path=bundle_path,
        archive_path=archive_path,
        artifact=artifact,
        selection=selection,
        output_directory=second,
    )
    assert first_bundle == second_bundle
    assert (first / "arrays.npz").read_bytes() == (second / "arrays.npz").read_bytes()
    loaded_bundle, loaded = load_array_bundle(first)
    assert loaded_bundle["bundleSha256"] == first_bundle["bundleSha256"]
    assert loaded_bundle["provenance"]["realizationSelection"] == selection
    conditioning = loaded_bundle["provenance"]["baryonicConditioning"]
    assert conditioning["surfaceWeight"] == 0.2
    assert conditioning["verticalConditionalWeight"] == (0.5 if volume else 1.0)
    assert conditioning["jointRealizationWeight"] == (0.1 if volume else 0.2)
    assert conditioning["surfaceLikelihoodConditioned"] is True
    assert conditioning["verticalStructureConditioned"] is False
    assert loaded_bundle["geometry"]["lengthUnit"] == "m"
    assert loaded_bundle["geometry"]["spacing"] == [0.5 * KPC_M] * (3 if volume else 2)
    factor = MSUN_KG / KPC_M ** (3 if volume else 2)
    total_key = "baryon_density" if volume else "baryon_surface_density"
    with np.load(archive_path, allow_pickle=False) as source:
        source_total = source[
            "total_baryonic_volume_density" if volume else "total_baryonic_surface_density"
        ][tuple(selection.values())]
    np.testing.assert_allclose(loaded[total_key], source_total * factor, rtol=0, atol=0)


def test_rejects_out_of_range_and_tampered_ensemble(tmp_path: Path) -> None:
    bundle_path, archive_path, artifact = _write_ensemble(tmp_path, volume=False)
    with pytest.raises(ValueError, match="outside"):
        materialize_galaxy_ensemble_realization(
            bundle_path=bundle_path,
            archive_path=archive_path,
            artifact=artifact,
            selection={"surfaceRealization": 3},
            output_directory=tmp_path / "bad-index",
        )
    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    bundle["arrays"][0]["contentSha256"] = "0" * 64
    core = {key: value for key, value in bundle.items() if key != "bundleSha256"}
    bundle["bundleSha256"] = canonical_sha256(core)
    bundle_path.write_text(json.dumps(bundle), encoding="utf-8")
    with pytest.raises(ValueError, match="content hash mismatch"):
        materialize_galaxy_ensemble_realization(
            bundle_path=bundle_path,
            archive_path=archive_path,
            artifact=artifact,
            selection={"surfaceRealization": 0},
            output_directory=tmp_path / "tampered",
        )


def test_rejects_tampered_conditioning_weights(tmp_path: Path) -> None:
    bundle_path, archive_path, artifact = _write_ensemble(tmp_path, volume=False)
    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    bundle["provenance"]["conditioning"]["surfaceWeights"] = [0.7, 0.3]
    core = {key: value for key, value in bundle.items() if key != "bundleSha256"}
    bundle["bundleSha256"] = canonical_sha256(core)
    bundle_path.write_text(json.dumps(bundle), encoding="utf-8")
    with pytest.raises(ValueError, match="weight hash mismatch"):
        materialize_galaxy_ensemble_realization(
            bundle_path=bundle_path,
            archive_path=archive_path,
            artifact=artifact,
            selection={"surfaceRealization": 0},
            output_directory=tmp_path / "tampered-weights",
        )
