import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_a383_feasibility_protocol_is_pre_pixel_and_residual_blind():
    config = json.loads((ROOT / "configs/r1_a383_gemini_feasibility_protocol.json").read_text())
    assert config["status"] == "frozen_before_any_A383_raw_download_or_pixel_inspection"
    assert config["selection_blind"] is True
    assert config["pre_pixel_overlap_target"]["preidentified_images_sorted_by_radius"][2]["image_id"] == "5.1"
    assert config["pre_pixel_overlap_target"]["frozen_outer_signed_bin_edge_arcsec"] == 10.5
    assert config["science_selection"]["archive_qa_required"] == "Pass"
    assert len(config["science_selection"]["science_filenames"]) == 4
    assert config["authorization"]["inspect_science_pixels"] is False
    assert config["authorization"]["fit_new_force_or_action"] is False


def test_a383_metadata_feasibility_report_passes_before_download():
    report = json.loads((ROOT / "results/r1_a383_gemini_feasibility/report.json").read_text())
    assert report["science_pixels_downloaded_or_inspected"] is False
    assert report["selected_science_frames"] == 4
    assert report["selected_science_exposure_seconds"] > 7500
    assert report["flat_frames"] == 5
    assert report["arc_frames"] == 2
    assert report["bias_frames"] == 5
    assert all(report["gates"].values())
    assert report["decision"] == "authorize_exact_raw_acquisition"
    assert report["authorization"]["download_exact_frozen_raw_files"] is True
    assert report["authorization"]["reduce_spectra"] is False
