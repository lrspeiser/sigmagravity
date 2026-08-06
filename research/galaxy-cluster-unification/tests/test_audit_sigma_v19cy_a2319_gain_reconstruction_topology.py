import inspect
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = ROOT / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import audit_sigma_v19cy_a2319_gain_reconstruction_topology as topology


def test_frozen_topology_scope_and_seals_are_exact() -> None:
    config, provenance = topology.validate_inputs(topology.DEFAULT_CONFIG)
    assert len(config["branches"]) == 7
    assert config["segment_detection"]["robustness_gap_seconds"] == [
        7200,
        10800,
        14400,
    ]
    assert config["gain_histories"]["allowed_scalar_columns"] == [
        "TIME",
        "PIXEL",
        "TEMP_FIT",
        "NEVENT",
        "CHISQ",
    ]
    assert not provenance["validation_or_holdout_asset_accessed"]


def test_reader_never_references_event_or_array_columns() -> None:
    source = inspect.getsource(topology.read_scalar_rows)
    assert "table[name]" in source
    assert "BINMESH" not in source
    assert "SPECTRUM" not in source
    assert "EVENTS" not in source
    assert "PHA" not in source


def test_segment_detection_respects_frozen_strict_gap_rule() -> None:
    rows = [
        {"TIME": 0.0, "PIXEL": 0, "obsid": "a"},
        {"TIME": 10.0, "PIXEL": 0, "obsid": "a"},
        {"TIME": 21.0, "PIXEL": 0, "obsid": "b"},
    ]
    segments = topology.detect_segments(rows, gap_seconds=10.0)
    assert len(segments) == 2
    assert segments[0]["start"] == 0.0
    assert segments[0]["stop"] == 10.0
    assert segments[1]["start"] == 21.0


def test_anchor_selection_matches_each_paper_branch_type() -> None:
    segments = [
        {"segment": 0, "start": 0.0, "stop": 10.0},
        {"segment": 1, "start": 30.0, "stop": 40.0},
    ]
    cross = topology.choose_anchors(
        {"start": 15.0, "stop": 25.0, "method": "cross_segment_linear_fit"},
        segments,
    )
    forward = topology.choose_anchors(
        {
            "start": 15.0,
            "stop": 25.0,
            "method": "preceding_segment_linear_extrapolation",
        },
        segments,
    )
    backward = topology.choose_anchors(
        {
            "start": 15.0,
            "stop": 25.0,
            "method": "following_segment_linear_extrapolation",
        },
        segments,
    )
    assert [item["segment"] for item in cross] == [0, 1]
    assert [item["segment"] for item in forward] == [0]
    assert [item["segment"] for item in backward] == [1]
