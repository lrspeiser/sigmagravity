from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "sigma_v19cb_foreground_treatment_information_audit.json"
SCRIPT = ROOT / "scripts" / "run_sigma_v19cb_foreground_treatment_information_audit.py"
SPEC = importlib.util.spec_from_file_location("sigma_v19cb", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)

REPORT_PATH = ROOT / "results" / "sigma_v19cb_foreground_treatment_information_audit" / "report.json"
OUTPUT_PATH = ROOT / "data" / "derived" / "sigma_v19cb_foreground_treatment_information_audit" / "release_branch_information.csv"
OUTPUT_SHA256 = "3ebd4088f253d49811b7fb6ab5e4c07aa0a069136e12cc492b6b2f81d7997b29"


def config() -> dict:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def test_v19cb_weight_branches_have_declared_meanings() -> None:
    branches = {row["id"]: row for row in config()["treatment_branches"]}
    clean = {"foreground_astrometric_evidence": "false", "quality_controlled_foreground_contamination": "false"}
    quality = {"foreground_astrometric_evidence": "true", "quality_controlled_foreground_contamination": "true"}
    weak = {"foreground_astrometric_evidence": "true", "quality_controlled_foreground_contamination": "false"}
    assert MODULE.candidate_weight(clean, branches["retain_all"]) == 1.0
    assert MODULE.candidate_weight(quality, branches["soft_quality_0p1"]) == 0.1
    assert MODULE.candidate_weight(quality, branches["mask_quality_diagnostic"]) == 0.0
    assert MODULE.candidate_weight(weak, branches["mask_quality_diagnostic"]) == 1.0
    assert MODULE.candidate_weight(weak, branches["mask_any_astrometry_diagnostic"]) == 0.0


def test_v19cb_discloses_exploration_and_authorizes_no_mask() -> None:
    cfg = config()
    assert cfg["honesty_boundary"]["complete_v19ca_source_result_inspected_before_freeze"]
    assert not cfg["honesty_boundary"]["gravity_kinematic_or_lensing_target_inspected"]
    assert not cfg["honesty_boundary"]["this_is_a_preregistered_theory_or_holdout_gate"]
    boundary = cfg["access_boundary"]
    for key in ("hard_star_mask_authorized", "treatment_branch_selected", "candidate_or_galaxy_removed", "optical_counterpart_selected", "wallaby_kinematic_table_row_read", "rotation_speed_or_velocity_field_read", "gravity_formula_residual_or_halo_result_read", "development_validation_holdout_split_selected", "gravity_action_or_constant_changed", "lensing_payload_opened", "solar_system_optimization_performed"):
        assert not boundary[key]


def test_v19cb_foreground_treatments_do_not_resolve_association() -> None:
    report = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
    assert all(report["gate_results"].values())
    assert report["decision"] == "foreground_astrometry_reduces_crowding_but_does_not_resolve_association"
    assert not report["association_resolved_by_foreground_astrometry"]
    summary = report["branch_summary"]
    assert summary["retain_all"]["robust_margin_ge_3"] == 3
    assert summary["soft_quality_0p1"]["robust_margin_ge_3"] == 34
    assert summary["mask_quality_diagnostic"]["robust_margin_ge_3"] == 35
    assert summary["mask_any_astrometry_diagnostic"]["robust_margin_ge_3"] == 41
    assert summary["mask_any_astrometry_diagnostic"]["field_summary"]["Norma"][
        "robust_margin_ge_3"
    ] == 3
    assert report["best_robust_fraction"] == 41 / 711


def test_v19cb_output_is_exact_and_contains_all_release_branches() -> None:
    report = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
    assert sha256(OUTPUT_PATH) == OUTPUT_SHA256
    assert report["output"]["sha256"] == OUTPUT_SHA256
    with OUTPUT_PATH.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 2_844
    assert len({(row["treatment"], row["source_row_id"]) for row in rows}) == 2_844
    assert {row["treatment"] for row in rows} == {
        "retain_all",
        "soft_quality_0p1",
        "mask_quality_diagnostic",
        "mask_any_astrometry_diagnostic",
    }
