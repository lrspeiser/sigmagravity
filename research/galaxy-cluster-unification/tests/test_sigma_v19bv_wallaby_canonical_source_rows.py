from __future__ import annotations

import csv
import importlib.util
import io
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "build_sigma_v19bv_wallaby_canonical_source_rows.py"
SPEC = importlib.util.spec_from_file_location("sigma_v19bv", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def build() -> tuple[bytes, dict]:
    return MODULE.build_artifacts()


def test_v19bv_passes_every_source_only_gate() -> None:
    _, report = build()
    assert report["decision"] == "passed_canonical_source_rows_not_holdout_selection"
    assert all(report["gate_results"].values())


def test_v19bv_emits_one_default_row_per_source_name() -> None:
    payload, report = build()
    rows = list(csv.DictReader(io.StringIO(payload.decode("utf-8"))))
    assert len(rows) == 592
    assert len({row["name"] for row in rows}) == 592
    assert report["canonical_output"]["canonical_rows"] == 592
    assert all(row["canonical_policy_id"] == MODULE.CANONICAL_POLICY for row in rows)


def test_v19bv_exposes_policy_sensitive_duplicates_instead_of_hiding_them() -> None:
    _, report = build()
    output = report["canonical_output"]
    assert output["duplicate_names_resolved"] == 119
    assert output["duplicate_names_with_all_five_policies_agreeing"] == 27
    assert output["duplicate_names_policy_ambiguous"] == 92
    assert output["all_name_distinct_choice_counts"] == {"1": 500, "2": 92}
    assert len(set(output["policy_choice_sha256"].values())) == 4
    assert report["access_boundary_audit"]["raw_alternative_rows_retained"]


def test_v19bv_uses_archive_primary_id_for_release_row_identity() -> None:
    _, report = build()
    hashes = report["canonical_output"]["policy_choice_sha256"]
    assert hashes[MODULE.CANONICAL_POLICY] == hashes[
        "KFLAG_QFLAG_RELIABILITY_SNR_PIXELS_RELEASE"
    ]
    assert hashes[MODULE.CANONICAL_POLICY] != hashes[
        "QFLAG_RELEASE_KFLAG_RELIABILITY_SNR_PIXELS"
    ]


def test_v19bv_keeps_kinematic_gravity_and_solar_targets_sealed() -> None:
    payload, report = build()
    columns = next(csv.reader(io.StringIO(payload.decode("utf-8"))))
    parent = json.loads(
        (ROOT / "results/sigma_v19bu_wallaby_source_only_metadata/report.json").read_text(
            encoding="utf-8"
        )
    )
    assert set(columns).isdisjoint(
        parent["source_target_boundary"]["sealed_target_columns"]
    )
    boundary = report["access_boundary_audit"]
    assert not boundary["kinematic_table_rows_read"]
    assert not boundary["rotation_speed_or_velocity_field_read"]
    assert not boundary["final_holdout_sample_selected"]
    assert not boundary["gravity_formula_or_constant_changed"]
    assert not boundary["solar_system_optimization_performed"]


def test_v19bv_committed_outputs_match_rebuild() -> None:
    payload, expected_report = build()
    csv_path = (
        ROOT
        / "data/derived/sigma_v19bv_wallaby_canonical_source_rows/wallaby_pilot_dr1_canonical_source_only.csv"
    )
    report_path = ROOT / "results/sigma_v19bv_wallaby_canonical_source_rows/report.json"
    assert csv_path.read_bytes() == payload
    assert json.loads(report_path.read_text(encoding="utf-8")) == expected_report
