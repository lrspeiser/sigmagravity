from __future__ import annotations

import csv
import importlib.util
import io
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "build_sigma_v19bw_wallaby_source_only_variety_frame.py"
SPEC = importlib.util.spec_from_file_location("sigma_v19bw", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def build() -> tuple[bytes, dict]:
    return MODULE.build_artifacts()


def test_v19bw_passes_every_source_only_variety_gate() -> None:
    _, report = build()
    assert report["decision"] == "passed_source_only_variety_frame_not_final_split"
    assert all(report["gate_results"].values())


def test_v19bw_preserves_the_candidate_universe_and_availability_lane() -> None:
    payload, report = build()
    rows = list(csv.DictReader(io.StringIO(payload.decode("utf-8"))))
    assert len(rows) == 592
    assert len({row["name"] for row in rows}) == 592
    assert sum(row["canonical_kflag"] == "2" for row in rows) == 109
    assert report["variety_frame"]["canonical_kinematic_availability_names"] == 109


def test_v19bw_has_broad_axis_and_release_field_coverage() -> None:
    _, report = build()
    frame = report["variety_frame"]
    for counts in frame["axis_quartile_counts_in_availability_lane"].values():
        assert set(counts) == {"Q1", "Q2", "Q3", "Q4"}
        assert min(counts.values()) >= 20
    assert set(frame["release_field_counts_in_availability_lane"]) == {
        "Hydra",
        "Norma",
        "NGC 4636",
    }
    assert frame["unique_multiaxis_variety_cells_in_availability_lane"] >= 20


def test_v19bw_propagates_release_row_ambiguity() -> None:
    payload, report = build()
    rows = list(csv.DictReader(io.StringIO(payload.decode("utf-8"))))
    assert sum(row["source_row_policy_sensitive"] == "true" for row in rows) == 92
    assert report["variety_frame"]["source_metric_bin_policy_sensitive_names"] > 0
    assert report["access_boundary_audit"]["raw_alternative_rows_retained"]


def test_v19bw_keeps_targets_actions_splits_and_solar_sealed() -> None:
    payload, report = build()
    columns = next(csv.reader(io.StringIO(payload.decode("utf-8"))))
    config = json.loads((ROOT / report["config"]).read_text(encoding="utf-8"))
    assert set(columns).isdisjoint(config["sealed_target_columns"])
    assert not any(
        any(token in column.lower() for token in ("holdout", "validation", "development", "fold"))
        for column in columns
    )
    boundary = report["access_boundary_audit"]
    assert not boundary["kinematic_table_rows_read"]
    assert not boundary["rotation_speed_or_velocity_field_read"]
    assert not boundary["development_validation_holdout_split_selected"]
    assert not boundary["final_galaxy_sample_selected"]
    assert not boundary["gravity_action_or_constant_changed"]
    assert not boundary["solar_system_optimization_performed"]


def test_v19bw_committed_outputs_match_rebuild() -> None:
    payload, expected_report = build()
    csv_path = (
        ROOT
        / "data/derived/sigma_v19bw_wallaby_source_only_variety_frame/wallaby_source_only_variety_frame.csv"
    )
    report_path = ROOT / "results/sigma_v19bw_wallaby_source_only_variety_frame/report.json"
    assert csv_path.read_bytes() == payload
    assert json.loads(report_path.read_text(encoding="utf-8")) == expected_report
