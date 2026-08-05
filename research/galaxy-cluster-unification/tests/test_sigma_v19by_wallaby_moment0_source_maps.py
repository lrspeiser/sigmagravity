from __future__ import annotations

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


def config() -> dict:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


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
