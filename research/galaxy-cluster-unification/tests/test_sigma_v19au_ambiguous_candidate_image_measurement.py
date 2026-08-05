import csv
import hashlib
import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "sigma_v19au_ambiguous_candidate_image_measurement.json"
METADATA = ROOT / "results" / "sigma_v19au_ambiguous_candidate_image_measurement" / "metadata_plan.json"
SCRIPT = ROOT / "scripts" / "run_sigma_v19au_ambiguous_candidate_image_measurement.py"
SPEC = importlib.util.spec_from_file_location("sigma_v19au", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_metadata_plan_is_complete_and_hash_bound():
    config = json.loads(CONFIG.read_text())
    report = json.loads(METADATA.read_text())
    assert report["members"] == 57
    assert report["member_candidate_hypotheses"] == 640
    assert report["unique_candidates"] == 568
    assert report["image_groups_with_candidates"] == 123
    assert report["candidate_exposure_measurements"] == 40812
    assert report["all_candidates_complete_grizY"]
    assert not report["science_pixels_opened_or_interpreted"]
    for name in ("candidate_measurement_plan", "candidate_hypotheses"):
        path = ROOT / report["outputs"][name]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == report["outputs"][f"{name}_sha256"]
    assert config["gates"]["measurements_by_filter"] == {
        "g": 7553,
        "r": 13757,
        "i": 8133,
        "z": 6175,
        "Y": 5194,
    }


def test_v19au_forbids_candidate_scoring_and_retains_failures():
    config = json.loads(CONFIG.read_text())
    assert config["frozen_measurement"]["variant"] == "area_scaled"
    assert config["frozen_measurement"]["aperture_diameter_arcsec"] == 4.0
    assert not config["authorization"]["fit_or_score_bri_color_likelihood"]
    assert not config["authorization"]["combine_with_positional_posterior"]
    assert not config["authorization"]["select_or_rank_counterparts"]
    assert config["authorization"]["retain_non_detections_and_failures"]


def test_aggregate_keeps_nonpositive_flux_without_calling_it_a_magnitude():
    sample = [
        {"candidate_id": "c1", "filter": "g", "flux": 2.0, "magnitude": 20.0},
        {"candidate_id": "c1", "filter": "g", "flux": -1.0, "magnitude": float("nan")},
    ]
    aggregate = MODULE.aggregate(sample, ["c1"], ["g"])[0]
    assert aggregate["planned_exposures"] == 2
    assert aggregate["finite_flux_exposures"] == 2
    assert aggregate["valid_exposures"] == 1
    assert aggregate["median_flux"] == 0.5
