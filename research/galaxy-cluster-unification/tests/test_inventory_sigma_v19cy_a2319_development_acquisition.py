import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = ROOT / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import inventory_sigma_v19cy_a2319_development_acquisition as inventory

REPORT = (
    ROOT
    / "results"
    / "sigma_v19cy_direct_icm_velocity_evidence"
    / "development_acquisition_inventory_report.json"
)


def test_acquisition_config_preserves_validation_and_holdout_seals() -> None:
    config = inventory.load_config(inventory.DEFAULT_CONFIG)
    inventory.validate_config(config)
    authorization = config["authorization"]
    assert authorization["inventory_and_download_all_listed_development_assets"]
    assert not authorization["download_or_open_validation_assets"]
    assert not authorization["download_or_open_holdout_assets"]
    assert not authorization["open_lensing_halo_or_gravity_targets"]


def test_science_selection_uses_only_required_resolve_reprocessing_inputs() -> None:
    config = inventory.load_config(inventory.DEFAULT_CONFIG)
    rows = inventory.selected_xrism_science_rows(config)
    assert len(rows) == 116
    assert {row["obsid"] for row in rows} == {"000101000", "000102000", "000103000"}
    assert all(not row["relative_path"].startswith("xtend/") for row in rows)
    assert all(not row["relative_path"].startswith("resolve/products/") for row in rows)
    assert any("p0px1000_uf.evt.gz" in row["relative_path"] for row in rows)
    assert any("p0px5000_uf.evt.gz" in row["relative_path"] for row in rows)


def test_predecessor_is_calibration_only_and_exactly_enumerated() -> None:
    config = inventory.load_config(inventory.DEFAULT_CONFIG)
    paths = config["xrism"]["calibration_predecessor"]["include_exact_paths"]
    assert len(paths) == len(set(paths)) == 15
    assert any(path.endswith("p0px5000_uf.evt.gz") for path in paths)
    assert any(path.endswith("000_fe55.ghf.gz") for path in paths)
    assert any(path.endswith("000_pxcal.ghf.gz") for path in paths)
    assert not any(path.endswith("p0px1000_uf.evt.gz") for path in paths)
    assert not any(path.endswith(".att.gz") for path in paths)


def test_chandra_role_classification() -> None:
    assert inventory.classify_chandra_role("acisf15187_000N003_evt1.fits.gz") == "evt1"
    assert inventory.classify_chandra_role("pcadf15187_000N001_asol1.fits.gz") == "asol1"
    assert inventory.classify_chandra_role("oif.fits") == "metadata"


def test_write_outputs_records_no_payload_access(tmp_path: Path) -> None:
    config_path = inventory.DEFAULT_CONFIG
    rows = [
        {
            "asset_group": "caldb",
            "role": "test",
            "obsid": "",
            "relative_path": "test.tar.gz",
            "download_path": "caldb/test.tar.gz",
            "url": "https://example.test/test.tar.gz",
            "bytes": 123,
            "last_modified": "",
            "etag": "",
            "content_type": "application/gzip",
        }
    ]
    report = inventory.write_outputs(config_path, tmp_path, rows)
    assert report["remote_totals"]["bytes"] == 123
    assert not report["payload_file_bodies_downloaded"]
    assert report["validation_and_holdout_outcome_seals_preserved"]
    saved = json.loads((tmp_path / "development_acquisition_inventory_report.json").read_text())
    assert saved["manifest"]["rows"] == 1


def test_terminal_development_inventory_is_sealed_and_exact() -> None:
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    assert report["status"] == (
        "a2319_scientifically_complete_development_acquisition_frozen_before_payload_download"
    )
    assert report["manifest"]["rows"] == 197
    assert report["manifest"]["sha256"] == (
        "3ef7816ccfa069a49f34cf18d22d1cd22da1c4fd1dc0ac2ed9151c2e66f16cac"
    )
    assert report["remote_totals"]["bytes"] == 12_742_865_194
    assert report["remote_totals"]["by_asset_group"]["xrism_science"] == {
        "bytes": 9_124_771_397,
        "files": 116,
    }
    assert report["remote_totals"]["by_asset_group"]["xrism_calibration_predecessor"] == {
        "bytes": 1_567_139_641,
        "files": 15,
    }
    assert not report["payload_file_bodies_downloaded"]
    assert not report["validation_or_holdout_asset_accessed"]
    assert not report["lensing_halo_or_gravity_payload_opened"]
    assert report["validation_and_holdout_outcome_seals_preserved"]
