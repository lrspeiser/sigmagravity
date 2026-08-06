import inspect
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = ROOT / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import audit_sigma_v19cy_a2319_region_readiness as readiness


def test_frozen_detector_regions_exactly_partition_science_pixels() -> None:
    config, parent = readiness.validate_inputs()
    readiness.validate_partition(config)
    assert parent["terminal_gate_passed"]
    pointings = config["pixel_partition"]["pointings"]
    assert list(pointings["P1"]["regions"]) == ["a", "b", "d"]
    assert list(pointings["P2"]["regions"]) == [
        "b_prime",
        "c_prime",
        "d_prime",
        "e_prime",
    ]
    assert 12 not in config["pixel_partition"]["science_pixels"]
    assert 27 not in config["pixel_partition"]["science_pixels"]
    assert config["terminal_gate"]["minimum_aggregate_rows_per_detector_region"] == 1000


def test_count_only_expression_matches_frozen_official_screen() -> None:
    config, _ = readiness.validate_inputs()
    expression = readiness.screen_expression(config, [0, 1])
    assert "ITYPE==0" in expression
    assert "RISE_TIME+0.00075*DERIV_MAX" in expression
    assert ">46.0" in expression
    assert "<58.0" in expression
    assert "STATUS[4]==b0" in expression
    assert "PIXEL!=27" in expression
    assert "(PIXEL==0||PIXEL==1)" in expression


def test_readiness_implementation_never_accesses_energy_values() -> None:
    source = inspect.getsource(readiness)
    forbidden_exact_column_accesses = (
        '["PI"]',
        "['PI']",
        '["EPI"]',
        "['EPI']",
        '["EPI2"]',
        "['EPI2']",
        '["PHA"]',
        "['PHA']",
        '["PHA2"]',
        "['PHA2']",
    )
    assert all(token not in source for token in forbidden_exact_column_accesses)
    assert "mean(" not in source
    assert "median(" not in source
    assert "histogram(" not in source
    assert "percentile(" not in source
    assert not readiness.load_json(readiness.DEFAULT_CONFIG)["authorization"][
        "read_or_summarize_any_energy_column"
    ]
