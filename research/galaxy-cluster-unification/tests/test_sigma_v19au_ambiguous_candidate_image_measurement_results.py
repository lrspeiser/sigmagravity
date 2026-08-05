import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "results" / "sigma_v19au_ambiguous_candidate_image_measurement" / "report.json"


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_v19au_failed_only_the_frozen_positive_fraction_gate():
    report = json.loads(REPORT.read_text())
    assert report["decision"] == "failed_closed"
    assert report["counts"] == {
        "complete_grizY_candidates": 454,
        "complete_griz_candidates": 461,
        "image_groups": 123,
        "planned_and_retained_measurements": 40812,
        "unique_candidates": 568,
        "valid_measurements": 24382,
    }
    assert report["gate_results"] == {
        "all_measurement_memberships_retained": True,
        "complete_griz_candidate_fraction": True,
        "no_candidate_association_scored": True,
        "overall_valid_fraction": False,
    }
    assert report["quality"]["measurement_status"] == {
        "nonpositive_retained": 16430,
        "valid": 24382,
    }


def test_v19au_outputs_are_hash_bound_and_have_no_processing_drop():
    report = json.loads(REPORT.read_text())
    for name in ("measurements", "aggregates", "group_audit"):
        path = ROOT / report["outputs"][name]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == report["outputs"][f"{name}_sha256"]
    measurements = rows(ROOT / report["outputs"]["measurements"])
    assert len(measurements) == 40812
    assert {row["measurement_status"] for row in measurements} == {
        "valid",
        "nonpositive_retained",
    }
    assert all(not row["measurement_error"] for row in measurements)


def test_signed_non_detections_retain_flux_and_uncertainty():
    report = json.loads(REPORT.read_text())
    measurements = rows(ROOT / report["outputs"]["measurements"])
    nonpositive = [row for row in measurements if row["measurement_status"] == "nonpositive_retained"]
    assert len(nonpositive) == 16430
    assert all(row["flux"] and row["flux_uncertainty"] for row in nonpositive)
    assert all(not row["magnitude"] or row["magnitude"].lower() == "nan" for row in nonpositive)
