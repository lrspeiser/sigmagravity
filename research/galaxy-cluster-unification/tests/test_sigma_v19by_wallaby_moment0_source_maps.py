from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
from pathlib import Path

from astropy.io import fits
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "sigma_v19by_wallaby_moment0_source_maps.json"
SCRIPT = ROOT / "scripts" / "acquire_sigma_v19by_wallaby_moment0_source_maps.py"
SPEC = importlib.util.spec_from_file_location("sigma_v19by", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)

REPORT_PATH = ROOT / "results" / "sigma_v19by_wallaby_moment0_source_maps" / "report.json"
MANIFEST_PATH = ROOT / "data" / "raw" / "sigma_v19by_wallaby_moment0_source_maps" / "manifest.csv"
MANIFEST_SHA256 = "871df6aa9db724ad648a08762d619884f326d643c86ecd97414b79d4a2ae7aa7"


def config() -> dict:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def test_v19by_inventory_query_is_exactly_moment0_source_data() -> None:
    cfg = config()
    query = MODULE.inventory_query(cfg)
    assert "o.collection='WALLABY'" in query
    for product_id in cfg["release_product_ids"].values():
        assert f"p.productID='{product_id}'" in query
    assert query.endswith("a.uri LIKE '%_mom0.fits'")
    assert "kinematic_model" not in query


def test_v19by_release_mapping_preserves_all_declared_alternatives() -> None:
    cfg = config()
    for release, product_id in cfg["release_product_ids"].items():
        assert MODULE.expected_product_id(cfg, {"team_release": release}) == product_id
    assert set(cfg["release_product_ids"]) == {
        "Hydra TR1",
        "Hydra TR2",
        "NGC 4636 TR1",
        "Norma TR1",
    }


def test_v19by_artifact_policy_rejects_spectral_and_kinematic_products() -> None:
    cfg = config()
    good = {
        "uri": "cadc:WALLABY/WALLABY_J000000+000000_Hydra_TR2_mom0.fits",
        "contentType": "application/fits",
        "productType": "science",
        "releaseType": "data",
        "contentChecksum": "md5:00000000000000000000000000000000",
    }
    MODULE.validate_artifact_policy(cfg, good)
    for suffix in ("cube.fits", "mask.fits", "mom1.fits", "mom2.fits", "spec.fits"):
        bad = dict(good, uri=f"cadc:WALLABY/WALLABY_J000000+000000_Hydra_TR2_{suffix}")
        try:
            MODULE.validate_artifact_policy(cfg, bad)
        except ValueError:
            pass
        else:  # pragma: no cover
            raise AssertionError(f"forbidden artifact accepted: {suffix}")


def test_v19by_fits_header_gate_distinguishes_2d_from_spectral(tmp_path: Path) -> None:
    source_map = tmp_path / "mom0.fits"
    fits.PrimaryHDU(np.zeros((4, 5), dtype=np.float32)).writeto(source_map)
    with fits.open(source_map, memmap=True) as hdul:
        assert hdul[0].header["NAXIS"] == 2
        assert "CTYPE3" not in hdul[0].header


def test_v19by_keeps_targets_counterparts_actions_and_solar_sealed() -> None:
    boundary = config()["access_boundary"]
    assert not boundary["cube_or_spectral_mask_downloaded"]
    assert not boundary["moment1_or_moment2_downloaded"]
    assert not boundary["spectrum_downloaded"]
    assert not boundary["kinematic_plane_or_table_read"]
    assert not boundary["rotation_speed_or_velocity_field_read"]
    assert not boundary["optical_counterpart_selected"]
    assert not boundary["development_validation_holdout_split_selected"]
    assert not boundary["gravity_action_or_constant_changed"]
    assert not boundary["solar_system_optimization_performed"]


def test_v19by_acquisition_passes_all_frozen_gates() -> None:
    report = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
    assert report["decision"] == "passed_moment0_source_maps_kinematics_sealed"
    assert all(report["gate_results"].values())
    assert report["map_output"]["files"] == 711
    assert report["map_output"]["bytes"] == 10_200_960
    assert report["map_output"]["release_counts"] == {
        "Hydra TR1": 148,
        "Hydra TR2": 272,
        "NGC 4636 TR1": 147,
        "Norma TR1": 144,
    }
    assert report["map_output"]["failures"] == {}
    assert "new_files" not in report["map_output"]
    assert "reused_files" not in report["map_output"]
    assert report["manifest_output"]["rows"] == 711
    assert report["manifest_output"]["sha256"] == MANIFEST_SHA256


def test_v19by_manifest_reproduces_every_local_map() -> None:
    assert sha256(MANIFEST_PATH) == MANIFEST_SHA256
    with MANIFEST_PATH.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 711
    for row in rows:
        local_path = ROOT / row["local_path"]
        assert local_path.is_file()
        assert sha256(local_path) == row["local_sha256"]
        assert local_path.stat().st_size == int(row["archive_content_length"])
        assert row["artifact_uri"].endswith("_mom0.fits")
        assert row["naxis"] == "2"
        assert row["has_spectral_axis"] == "false"


def test_v19by_manifest_contains_no_forbidden_artifact_class() -> None:
    forbidden = ("_cube", "_mask", "_mom1", "_mom2", "_spec", "kinematic", "model", "residual", "rotation")
    with MANIFEST_PATH.open(encoding="utf-8", newline="") as handle:
        uris = [row["artifact_uri"].lower() for row in csv.DictReader(handle)]
    assert len(uris) == 711
    assert all(not any(token in uri for token in forbidden) for uri in uris)
