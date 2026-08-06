import inspect
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = ROOT / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import close_sigma_v19cy_a2319_official_gain_counts as closure


def test_frozen_official_count_scope_and_seals_are_exact() -> None:
    config, provenance = closure.validate_inputs(closure.DEFAULT_CONFIG)
    assert config["source"]["obsid"] == "000100000"
    assert config["source"]["allowed_scalar_columns"] == ["TIME", "PIXEL"]
    assert len(config["relative_exclusion_intervals_seconds"]["saa"]) == 45
    assert len(config["relative_exclusion_intervals_seconds"]["adr"]) == 4
    assert not provenance["validation_or_holdout_asset_accessed"]


def test_reader_is_limited_to_time_and_pixel() -> None:
    source = inspect.getsource(closure.read_time_and_pixel)
    assert 'hdu.data["TIME"]' in source
    assert 'hdu.data["PIXEL"]' in source
    assert "BINMESH" not in source
    assert "SPECTRUM" not in source
    assert "PHA" not in source


def test_count_and_interval_helpers_are_boundary_inclusive() -> None:
    pixels = np.asarray([12, 1, 12, 0, 1])
    assert closure.count_pixels(pixels) == {"0": 1, "1": 2, "12": 2}

    overlaps = closure.interval_overlaps(
        np.asarray([99.0, 100.0, 101.0, 102.0]),
        100.0,
        {"saa": [[0.0, 1.0]], "adr": [[2.0, 2.0]]},
    )
    assert overlaps["saa"]["unique_rows_inside"] == 2
    assert overlaps["adr"]["unique_rows_inside"] == 1


def test_terminal_official_gain_count_closure_passes_without_opening_seals() -> None:
    report_path = (
        ROOT
        / "results"
        / "sigma_v19cy_direct_icm_velocity_evidence"
        / "development_official_gain_count_closure.json"
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))

    assert report["status"] == "a2319_official_gain_count_closure_audited"
    assert report["raw_rows"] == 2765
    assert report["actual_counts"] == {
        "fe55_non_pixel_12": 2294,
        "intermittent_calibration_pixel_12": 471,
    }
    assert report["actual_per_pixel"] == report["expected_per_pixel"]
    assert all(report["checks"].values())
    assert report["interval_overlap_audit"]["saa"]["intervals"] == 45
    assert report["interval_overlap_audit"]["saa"]["unique_rows_inside"] == 0
    assert report["interval_overlap_audit"]["adr"]["intervals"] == 4
    assert report["interval_overlap_audit"]["adr"]["unique_rows_inside"] == 0
    assert report["official_count_closure_reproduced"]
    assert report["decision"] == (
        "authorize_separate_gain_reconstruction_protocol_freeze"
    )

    assert not report["continuous_calibration_pixel_history_accessed"]
    assert not report["gain_history_array_column_read"]
    assert not report["event_row_or_energy_read"]
    assert not report["gain_applied_or_interpolated"]
    assert not report["spectrum_or_velocity_fit_performed"]
    assert not report["validation_or_holdout_accessed"]
