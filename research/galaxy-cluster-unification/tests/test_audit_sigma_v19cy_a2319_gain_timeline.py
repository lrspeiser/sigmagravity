import inspect
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = ROOT / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import audit_sigma_v19cy_a2319_gain_timeline as timeline


def test_frozen_gain_timeline_inputs_preserve_all_seals() -> None:
    config, provenance = timeline.validate_inputs(timeline.DEFAULT_CONFIG)
    assert len(config["gain_histories"]["obsids"]) == 4
    assert len(config["science_gtis"]["obsids"]) == 3
    assert len(config["gain_histories"]["science_pixels"]) == 34
    assert not provenance["validation_or_holdout_asset_accessed"]


def test_reader_accesses_only_config_supplied_columns() -> None:
    source = inspect.getsource(timeline.read_columns)
    assert "table[name]" in source
    assert "BINMESH" not in source
    assert "SPECTRUM" not in source
    assert "EVENTS" not in source


def test_deduplication_is_exact_in_time_and_pixel() -> None:
    rows = [
        {"TIME": 1.0, "PIXEL": 2, "TEMP_FIT": 50.0},
        {"TIME": 1.0, "PIXEL": 2, "TEMP_FIT": 51.0},
        {"TIME": 1.0, "PIXEL": 3, "TEMP_FIT": 50.0},
    ]
    unique = timeline.deduplicate_time_pixel(rows)
    assert len(unique) == 2
    assert unique[0]["TEMP_FIT"] == 50.0


def test_interval_membership_includes_boundaries() -> None:
    intervals = [{"start": 2.0, "stop": 3.0}]
    assert timeline.inside_any_interval(2.0, intervals)
    assert timeline.inside_any_interval(3.0, intervals)
    assert not timeline.inside_any_interval(3.1, intervals)


def test_terminal_gain_timeline_audit_stops_at_the_frozen_boundary() -> None:
    report_path = (
        ROOT
        / "results"
        / "sigma_v19cy_direct_icm_velocity_evidence"
        / "development_gain_timeline_evidence.json"
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))

    assert report["status"] == "a2319_gain_timeline_scalar_evidence_audited"
    assert report["gain_rows"] == {"calpixel": 4219, "fe55": 11405}
    assert not report["official_count_closure_reproduced"]
    assert report["official_count_matches"] == {"calpixel": [], "fe55": []}
    assert report["decision"] == (
        "stop_before_gain_application_and_require_documented_solution_selection_rule"
    )

    coverage = report["coverage"]
    assert len(coverage) == 6
    assert all(item["pixels_with_preceding_anchor"] == 34 for item in coverage)
    assert all(item["pixels_with_following_anchor"] == 34 for item in coverage[:5])
    assert coverage[-1]["pixels_with_following_anchor"] == 0

    assert not report["gain_history_array_column_read"]
    assert not report["event_row_or_energy_read"]
    assert not report["gain_applied_or_interpolated"]
    assert not report["spectrum_or_velocity_fit_performed"]
    assert not report["validation_or_holdout_accessed"]
    assert not report["authorization"]["freeze_gain_interpolation_protocol"]
