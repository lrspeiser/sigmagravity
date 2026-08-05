import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "results" / "sigma_v19at_decam_forced_photometry_validation" / "report.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_v19at_passed_every_frozen_gate_and_is_hash_bound():
    report = json.loads(REPORT.read_text())
    assert report["decision"] == "passed"
    assert report["counts"] == {
        "complete_griz_validation_objects": 5,
        "validation_anchors": 5,
        "validation_image_groups": 110,
        "validation_measurements": 362,
    }
    assert all(report["gate_results"].values())
    for name in (
        "measurements",
        "aggregates",
        "development_color_fit",
        "validation_predictions",
        "validation_retrieval",
        "group_audit",
    ):
        path = ROOT / report["outputs"][name]
        assert sha256(path) == report["outputs"][f"{name}_sha256"]


def test_v19at_metrics_meet_exact_predeclared_thresholds():
    report = json.loads(REPORT.read_text())
    metrics = report["validation_metrics"]
    assert metrics["top1_retrievals"] >= 3
    assert metrics["mean_reciprocal_rank"] >= 0.65
    assert all(value <= 0.25 for value in metrics["median_absolute_error_mag"].values())
    assert metrics["true_pair_ranks"] == {"21": 3, "26": 1, "57": 1, "66": 3, "71": 1}


def test_crowded_member57_has_every_griz_exposure_and_low_scatter():
    report = json.loads(REPORT.read_text())
    aggregates = rows(ROOT / report["outputs"]["aggregates"])
    member57 = {row["filter"]: row for row in aggregates if row["member_id"] == "57"}
    assert {band: int(member57[band]["valid_exposures"]) for band in "griz"} == {
        "g": 14,
        "r": 25,
        "i": 14,
        "z": 11,
    }
    assert all(float(member57[band]["robust_scatter_mag"]) < 0.05 for band in "griz")
