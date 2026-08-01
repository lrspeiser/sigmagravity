import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_ms2137_feasibility_protocol_is_pre_pixel_and_residual_blind():
    config = json.loads((ROOT / "configs/r1_ms2137_muse_feasibility_protocol.json").read_text())
    assert config["status"] == "frozen_before_any_MS2137_cube_download_or_pixel_inspection"
    assert config["selection_blind"] is True
    assert config["published_bcg_center"]["ra_hms"] == "21:40:15.16"
    assert config["archive_product"]["dp_id"] == "ADP.2019-07-31T03:40:02.259"
    assert config["pre_pixel_overlap_target"]["frozen_outer_radial_edge_arcsec"] == 14.0
    assert len(config["pre_pixel_overlap_target"]["preidentified_images_inside_frozen_support_sorted_by_radius"]) == 4
    assert config["frozen_cutout_request"]["radius_arcsec"] == 18.0
    assert config["authorization"]["inspect_science_pixels"] is False
    assert config["authorization"]["fit_new_force_or_action"] is False


def test_ms2137_metadata_feasibility_report_passes_before_download():
    report = json.loads((ROOT / "results/r1_ms2137_muse_feasibility/report.json").read_text())
    assert report["science_pixels_downloaded_or_inspected"] is False
    assert report["archive_product"]["calib_level"] == 2
    assert report["archive_product"]["dataproduct_type"] == "cube"
    assert report["bcg_to_cube_center_arcsec"] < 1.0
    assert report["matched_lens_image_count"] == 4
    assert report["matched_lens_family_count"] == 3
    assert all(report["gates"].values())
    assert report["decision"] == "authorize_frozen_soda_cutout_acquisition"
    assert report["authorization"]["download_frozen_cutout"] is True
    assert report["authorization"]["inspect_science_pixels"] is False
