import inspect
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
