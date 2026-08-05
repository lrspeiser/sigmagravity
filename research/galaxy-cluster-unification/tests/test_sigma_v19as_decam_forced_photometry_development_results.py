import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "results" / "sigma_v19as_decam_forced_photometry_development" / "report.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_v19as_outputs_are_hash_bound_and_complete():
    report = json.loads(REPORT.read_text())
    assert report["decision"] == "development_completed_validation_still_sealed"
    assert report["counts"] == {
        "development_anchors": 10,
        "development_image_groups": 122,
        "development_measurements": 670,
        "output_measurement_rows": 4020,
        "validation_anchors_measured": 0,
    }
    for name in ("measurements", "aggregates", "ranking", "group_audit"):
        path = ROOT / report["outputs"][name]
        assert sha256(path) == report["outputs"][f"{name}_sha256"]


def test_v19as_measurements_never_contain_validation_members():
    config = json.loads(
        (ROOT / "configs" / "sigma_v19as_decam_forced_photometry_development.json").read_text()
    )
    report = json.loads(REPORT.read_text())
    measurements = rows(ROOT / report["outputs"]["measurements"])
    members = {row["member_id"] for row in measurements}
    assert members == set(config["split"]["development_ids"])
    assert members.isdisjoint(config["split"]["validation_ids"])
    assert len(measurements) == 670 * 2 * 3


def test_v19as_recommendation_is_the_frozen_four_arcsecond_area_rule():
    report = json.loads(REPORT.read_text())
    winner = report["recommendation_for_separate_validation_freeze"]
    assert winner["variant"] == "area_scaled"
    assert winner["aperture_diameter_arcsec"] == 4.0
    assert winner["valid_measurement_fraction"] == 1.0
    assert winner["complete_griz_development_objects"] == 10
    assert winner["median_repeatability_scatter_mag"] < 0.04
    assert winner["leave_one_out_color_mae_mag"] < 0.04
