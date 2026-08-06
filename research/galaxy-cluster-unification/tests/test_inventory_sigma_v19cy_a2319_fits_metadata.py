import inspect
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = ROOT / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import inventory_sigma_v19cy_a2319_fits_metadata as metadata


def test_frozen_metadata_inputs_and_selection_are_exact() -> None:
    config, provenance, manifest = metadata.validate_inputs(metadata.DEFAULT_CONFIG)
    rows = metadata.selected_rows(config, manifest)
    assert len(rows) == 87
    assert {row["obsid"] for row in rows} == {
        "000100000",
        "000101000",
        "000102000",
        "000103000",
    }
    assert not provenance["validation_or_holdout_asset_accessed"]


def test_metadata_runner_never_accesses_hdu_data_property() -> None:
    source = inspect.getsource(metadata.inspect_header_only)
    assert ".data" not in source
    assert "_data_loaded" in source


def test_scalar_preserves_json_scalars_and_stringifies_other_values() -> None:
    assert metadata.scalar(3) == 3
    assert metadata.scalar("gain") == "gain"
    assert metadata.scalar(Path("sigma")) == "sigma"


def test_terminal_metadata_inventory_passes_without_loading_data() -> None:
    report_path = (
        ROOT
        / "results"
        / "sigma_v19cy_direct_icm_velocity_evidence"
        / "development_fits_metadata_inventory.json"
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["status"] == (
        "a2319_development_fits_metadata_and_schemas_inventoried_without_loading_data"
    )
    assert report["files"] == 87
    assert report["compressed_bytes"] == 7_661_862_987
    assert report["hdus"] == 452
    assert report["every_hdu_data_object_remained_unloaded"]
    assert not report["table_or_image_value_read"]
    assert not report["scientific_fit_performed"]
    assert not report["validation_or_holdout_accessed"]
    assert report["authorization"]["freeze_gain_reconstruction_protocol"]
