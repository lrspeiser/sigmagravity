import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_a2537_feasibility_is_frozen_disturbed_control():
    config = json.loads((ROOT / "configs/r1_a2537_gemini_feasibility_protocol.json").read_text())
    assert config["status"] == "frozen_before_any_A2537_raw_download_or_pixel_inspection"
    assert config["disturbance_status"]["disturbed_control"] is True
    assert config["published_bcg_center"]["ra_hms"] == "23:08:22.3"
    assert config["pre_pixel_overlap_target"]["frozen_outer_signed_bin_edge_arcsec"] == 16.0
    assert len(config["science_selection"]["science_filenames"]) == 4
    assert config["authorization"]["count_as_non_disturbed_pilot"] is False
    assert config["authorization"]["inspect_science_pixels"] is False


def test_a2537_metadata_report_passes_before_download():
    report = json.loads((ROOT / "results/r1_a2537_gemini_feasibility/report.json").read_text())
    assert report["science_pixels_downloaded_or_inspected"] is False
    assert report["disturbed_control"] is True
    assert report["selected_science_frames"] == 4
    assert report["selected_science_exposure_seconds"] > 7200
    assert report["flat_frames"] == 4
    assert report["arc_frames"] == 2
    assert report["bias_frames"] == 10
    assert all(report["gates"].values())
    assert report["decision"] == "authorize_exact_raw_acquisition_as_disturbed_control"
    assert report["authorization"]["download_exact_frozen_raw_files"] is True
    assert report["authorization"]["count_as_non_disturbed_pilot"] is False
