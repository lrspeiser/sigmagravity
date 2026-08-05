from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "sigma_v19ca_skymapper_gaia_foreground_diagnostics.json"
SCRIPT = ROOT / "scripts" / "acquire_sigma_v19ca_skymapper_gaia_foreground_diagnostics.py"
SPEC = importlib.util.spec_from_file_location("sigma_v19ca", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)

REPORT_PATH = ROOT / "results" / "sigma_v19ca_skymapper_gaia_foreground_diagnostics" / "report.json"
RAW_PATH = ROOT / "data" / "raw" / "sigma_v19ca_skymapper_gaia_foreground_diagnostics" / "skymapper_gaia_dr3_nearest.csv"
DERIVED_PATH = ROOT / "data" / "derived" / "sigma_v19ca_skymapper_gaia_foreground_diagnostics" / "candidate_foreground_diagnostics.csv"
RAW_SHA256 = "41aaa5f42b796094db769c11473f2a0e6d933253742e27557a67fdd33bc0fffc"
DERIVED_SHA256 = "a7b5b38bd7e1bf5fac88dda097be3fd8757d8428774d59682096595886f43d15"


def config() -> dict:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def test_v19ca_query_is_exact_id_join_and_excludes_radial_velocity() -> None:
    cfg = config()
    query = MODULE.build_query(cfg, [3, 1, 2])
    assert "LEFT OUTER JOIN ext.gaia_dr3 AS g ON m.gaia_dr3_id1=g.source_id" in query
    assert "m.object_id IN (1,2,3)" in query
    assert query.endswith("ORDER BY m.object_id")
    lowered = query.lower()
    for token in ("radial_velocity", "rotation", "halo", "lensing"):
        assert token not in lowered


def test_v19ca_foreground_evidence_requires_exact_five_sigma_astrometry() -> None:
    policy = config()["diagnostic_policy"]
    base = {
        "object_id": "1",
        "matched_gaia_source_id": "10",
        "gaia_dr3_dist1": "0.2",
        "parallax": "1.0",
        "parallax_error": "0.1",
        "pmra": "0.0",
        "pmra_error": "0.1",
        "pmdec": "0.0",
        "pmdec_error": "0.1",
        "ruwe": "1.0",
        "astrometric_params_solved": "95",
    }
    result = MODULE.derive_foreground_diagnostic(base, policy)
    assert result["foreground_astrometric_evidence"] == "true"
    assert result["quality_controlled_foreground_contamination"] == "true"
    assert result["evidence_channels"] == "positive_parallax"
    assert MODULE.derive_foreground_diagnostic(
        dict(base, gaia_dr3_dist1="1.1"), policy
    )["foreground_astrometric_evidence"] == "false"
    assert MODULE.derive_foreground_diagnostic(dict(base, parallax="0.49"), policy)[
        "foreground_astrometric_evidence"
    ] == "false"


def test_v19ca_quality_flag_retains_poor_astrometric_solutions() -> None:
    policy = config()["diagnostic_policy"]
    row = {
        "object_id": "1",
        "matched_gaia_source_id": "10",
        "gaia_dr3_dist1": "0.2",
        "parallax": "",
        "parallax_error": "",
        "pmra": "6.0",
        "pmra_error": "1.0",
        "pmdec": "",
        "pmdec_error": "",
        "ruwe": "2.0",
        "astrometric_params_solved": "95",
    }
    result = MODULE.derive_foreground_diagnostic(row, policy)
    assert result["foreground_astrometric_evidence"] == "true"
    assert result["quality_controlled_foreground_contamination"] == "false"


def test_v19ca_discloses_pilot_and_keeps_all_targets_sealed() -> None:
    cfg = config()
    honesty = cfg["honesty_boundary"]
    assert honesty["three_candidate_rows_piloted_before_freeze"]
    assert not honesty["complete_candidate_population_queried_before_freeze"]
    assert not honesty["gravity_kinematic_or_lensing_target_inspected"]
    boundary = cfg["access_boundary"]
    for key in (
        "gaia_radial_velocity_read",
        "hard_star_mask_applied",
        "candidate_removed_or_weighted",
        "optical_counterpart_selected",
        "wallaby_kinematic_table_row_read",
        "rotation_speed_or_velocity_field_read",
        "gravity_formula_residual_or_halo_result_read",
        "development_validation_holdout_split_selected",
        "gravity_action_or_constant_changed",
        "lensing_payload_opened",
        "solar_system_optimization_performed",
    ):
        assert not boundary[key]


def test_v19ca_complete_acquisition_passes_and_quantifies_foreground() -> None:
    report = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
    assert report["decision"] == "foreground_diagnostics_acquired_without_candidate_assignment"
    assert all(report["gate_results"].values())
    assert report["input_audit"]["candidate_occurrences"] == 17_094
    assert report["input_audit"]["unique_object_ids"] == 17_034
    assert report["input_audit"]["returned_rows"] == 17_034
    assert report["input_audit"]["query_batches"] == 43
    assert report["input_audit"]["missing"] == []
    assert report["input_audit"]["unexpected"] == []
    assert report["input_audit"]["duplicates"] == []
    summary = report["diagnostic_summary"]
    assert summary["exact_gaia_matches"] == 13_958
    assert summary["foreground_astrometric_evidence"] == 12_801
    assert summary["quality_controlled_foreground_contamination"] == 12_347
    assert summary["unflagged_or_unresolved"] == 4_687
    assert summary["field_summary"]["Norma"][
        "quality_controlled_foreground_contamination_objects"
    ] == 10_213


def test_v19ca_outputs_are_exact_unique_and_contain_no_radial_velocity() -> None:
    report = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
    assert sha256(RAW_PATH) == RAW_SHA256
    assert sha256(DERIVED_PATH) == DERIVED_SHA256
    assert report["outputs"]["raw_crossmatch"]["sha256"] == RAW_SHA256
    assert report["outputs"]["derived_diagnostics"]["sha256"] == DERIVED_SHA256
    with RAW_PATH.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        assert reader.fieldnames is not None
        raw_fields = list(reader.fieldnames)
        raw_rows = list(reader)
    assert len(raw_rows) == 17_034
    assert len({row["object_id"] for row in raw_rows}) == 17_034
    assert not any("radial_velocity" in field.lower() for field in raw_fields)
    with DERIVED_PATH.open(encoding="utf-8", newline="") as stream:
        derived_rows = list(csv.DictReader(stream))
    assert len(derived_rows) == 17_034
    assert len({row["object_id"] for row in derived_rows}) == 17_034
