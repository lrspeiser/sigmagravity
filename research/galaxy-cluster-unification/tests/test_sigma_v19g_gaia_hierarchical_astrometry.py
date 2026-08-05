from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "sigma_v19g_gaia_hierarchical_astrometry.json"
DOWNLOADER = ROOT / "scripts" / "download_sigma_v19g_gaia_astrometry.py"
ACQUISITION = ROOT / "results" / "sigma_v19g_gaia_acquisition" / "provenance.json"
REPORT = ROOT / "results" / "sigma_v19g_chandra_astrometry" / "report.json"


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_downloader():
    scripts = str(ROOT / "scripts")
    sys.path.insert(0, scripts)
    try:
        spec = importlib.util.spec_from_file_location("sigma_v19g_download_test", DOWNLOADER)
        assert spec and spec.loader
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(scripts)

def test_protocol_is_frozen_target_blind_and_content_addressed() -> None:
    module = load_downloader()
    config = module.validate(CONFIG)
    assert set(config["clusters"]) == {"BULLET", "ABELL2146"}
    assert config["integrity"] == {
        "gaia_rows_or_matches_known_at_freeze": False,
        "relative_match_outcomes_known_at_freeze": False,
        "registered_science_image_inspected": False,
        "shock_front_fitted": False,
        "source_constructed": False,
        "lensing_target_opened": False,
        "gravity_parameter_changed": False,
    }


def test_reference_observations_follow_the_frozen_exposure_rule() -> None:
    config = load(CONFIG)
    cleaning = load(ROOT / config["parents"]["cleaning_report"])
    for cluster in config["clusters"]:
        rows = [row for row in cleaning["observations"] if row["cluster"] == cluster]
        expected = min(
            rows,
            key=lambda row: (-float(row["clean_exposure_seconds"]), int(row["obsid"])),
        )
        declared = config["reference_selection"][
            "resolved_from_frozen_cleaning_report"
        ][cluster]
        assert int(declared["obsid"]) == int(expected["obsid"])
        assert config["clusters"][cluster]["reference_obsid"] == int(
            expected["obsid"]
        )


def test_astrometry_cannot_change_scale_rotation_or_shear() -> None:
    config = load(CONFIG)
    for section in ("matching", "relative_matching"):
        assert config[section]["method"] == "trans"
        assert config[section]["rotation_deg"] == 0.0
        assert config[section]["scale"] == 1.0
        assert config[section]["shear"] == 0.0
        assert config[section]["minimum_final_source_pairs"] == 3
        assert config[section]["maximum_final_radial_rms_arcsec"] == 0.5


def test_gaia_acquisition_is_complete_and_pre_match() -> None:
    config = load(CONFIG)
    report = load(ACQUISITION)
    assert report["config_sha256"] == load_downloader().common.sha256(CONFIG)
    assert {row["cluster"] for row in report["records"]} == {
        "BULLET",
        "ABELL2146",
    }
    assert report["files"] == 2
    assert report["rows"] > 0
    assert report["xray_source_crossmatch_run"] is False
    assert report["astrometric_offset_fit"] is False
    assert report["registered_science_image_inspected"] is False
    assert report["lensing_target_opened"] is False
    for row in report["records"]:
        path = ROOT / row["relative_path"]
        assert path.stat().st_size == row["bytes"]
        assert load_downloader().common.sha256(path) == row["sha256"]
    assert report["protocol_version"] == config["protocol_version"]


def test_hierarchical_registration_passes_without_shape_freedom() -> None:
    report = load(REPORT)
    assert report["observation_count"] == 20
    assert report["all_hierarchical_gates_passed"] is True
    assert report["transforms_applied"] is True
    assert report["failed_observations"] == []
    assert report["registered_science_images_inspected"] is False
    assert report["shock_front_fitted"] is False
    assert report["source_constructed"] is False
    assert report["lensing_target_opened"] is False
    assert report["reference_obsids"] == {"BULLET": 5356, "ABELL2146": 12247}
    for row in report["observations"]:
        assert all(row["gates"].values())
        values = row["transform_values"]
        assert values["a11"] == 1.0
        assert values["a12"] == 0.0
        assert values["a21"] == 0.0
        assert values["a22"] == 1.0
        assert row["match_statistics"]["included_pairs"] >= 3
        assert row["match_statistics"]["included_rms_recomputed_arcsec"] <= 0.5
        assert set(row["application"]["corrected_events"]) == {
            "science",
            "blanksky",
        }


def test_runner_is_importable() -> None:
    scripts = str(ROOT / "scripts")
    sys.path.insert(0, scripts)
    try:
        spec = importlib.util.spec_from_file_location(
            "sigma_v19g_runner_test",
            ROOT / "scripts" / "run_sigma_v19g_chandra_astrometry.py",
        )
        assert spec and spec.loader
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(scripts)
