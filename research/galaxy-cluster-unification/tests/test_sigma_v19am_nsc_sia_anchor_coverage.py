import csv
import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "audit_sigma_v19am_nsc_sia_anchor_coverage.py"
CONFIG = ROOT / "configs" / "sigma_v19am_nsc_sia_anchor_coverage.json"
REPORT = ROOT / "results" / "sigma_v19am_nsc_sia_anchor_coverage" / "report.json"
MANIFEST = (
    ROOT
    / "data"
    / "derived"
    / "sigma_v19am_nsc_sia_anchor_coverage"
    / "exposure_cutout_manifest.csv"
)
SPEC = importlib.util.spec_from_file_location("sigma_v19am", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_frozen_runner_and_parent_hashes_match():
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    hashes = MODULE.validate_config(CONFIG, config)
    assert hashes["runner"] == config["implementation"]["runner_sha256"]
    for artifact in config["parent_artifacts"]:
        assert hashes[artifact["path"]] == artifact["sha256"]


def test_frozen_anchor_sample_and_split_are_exact():
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    anchors = MODULE.load_anchors(config)
    assert len(anchors) == 15
    assert sum(row["split"] == "development" for row in anchors) == 10
    assert sum(row["split"] == "validation" for row in anchors) == 5
    assert len({row["nsc_id"] for row in anchors}) == 15


def test_query_and_access_parsing_are_exact_not_fuzzy():
    url = MODULE.build_query_url("https://example.test/sia", 12.5, -3.25, 0.01)
    assert "POS=12.500000000000%2C-3.250000000000" in url
    assert "SIZE=0.01000000" in url
    ref, extension = MODULE.parse_access_descriptor(
        "https://example.test/cutout?col=nsc_dr2&siaRef=tu123.fits.fz&extn=7&POS=1,2&SIZE=.01,.01"
    )
    assert ref == "tu123"
    assert extension == "7"


def test_config_forbids_pixel_access_selection_and_science_inference():
    authorization = json.loads(CONFIG.read_text(encoding="utf-8"))["authorization"]
    assert authorization["query_sia_metadata_for_exact_fifteen_anchor_coordinates"]
    assert not authorization["download_image_pixels"]
    assert not authorization["rank_or_select_exposures"]
    assert not authorization["inspect_image_pixels"]
    assert not authorization["query_ambiguous_candidates"]
    assert not authorization["infer_photometry_mass_or_current"]
    assert not authorization["read_lensing_or_halo_payload"]
    assert not authorization["change_gravity_physics_or_parameters"]


def test_completed_report_and_manifest_prove_exact_metadata_coverage():
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    with MANIFEST.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert report["gates"]["all_metadata_coverage_gates_pass"]
    assert report["counts"]["anchors"] == 15
    assert report["counts"]["measurement_descriptor_pairs"] == 1032
    assert report["counts"]["unique_exposures"] == 82
    assert report["counts"]["unique_exposure_extensions"] == 139
    assert len(rows) == 1032
    assert len({(row["nsc_id"], row["exposure"]) for row in rows}) == 1032
    assert all(row["filter"] == row["sia_obs_bandpass"] for row in rows)
    assert all(row["sia_instrument_name"] == "DECam" for row in rows)
    assert all(row["sia_proctype"] == "InstCal" for row in rows)
    assert MODULE.sha256(MANIFEST) == report["outputs"]["manifest_sha256"]
    assert not report["image_pixels_downloaded"]
    assert not report["exposures_ranked_or_selected"]
    assert not report["ambiguous_candidates_queried"]
    assert not report["lensing_or_halo_payload_opened"]
    assert not report["gravity_formula_or_parameter_changed"]


def test_every_preserved_raw_metadata_payload_matches_its_report_hash():
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    assert len(report["query_records"]) == 15
    for record in report["query_records"]:
        path = ROOT / record["raw_metadata_path"]
        assert path.is_file()
        assert MODULE.sha256(path) == record["raw_metadata_sha256"]
