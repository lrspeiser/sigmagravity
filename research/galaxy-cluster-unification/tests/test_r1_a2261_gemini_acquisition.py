from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_a2261_target_is_frozen_before_raw_download() -> None:
    config = json.loads((ROOT / "configs/r1_a2261_gemini_acquisition_protocol.json").read_text())
    overlap = config["pre_pixel_overlap_target"]
    assert config["frozen_before_raw_download"] is True
    assert overlap["minimum_one_sided_accepted_dynamics_support_kpc"] == 36.0
    assert [row["image_id"] for row in overlap["required_preidentified_images"]] == ["6b", "11b", "7a"]
    assert len({row["family_id"] for row in overlap["required_preidentified_images"]}) == 3
    assert len(config["exact_target_associated_download"]) == 17
    assert len(config["calibration_gate"]["exact_bias_download"]) == 10
    assert config["authorization"]["reduce_spectra"] is False


def test_a2261_raw_gate_passes_but_reduction_stays_closed() -> None:
    subprocess.run(
        [sys.executable, str(ROOT / "scripts/audit_r1_a2261_gemini_acquisition.py")],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    report = json.loads((ROOT / "results/r1_a2261_gemini_acquisition/report.json").read_text())
    assert report["target_associated_files"] == 17
    assert report["science_frames"] == 4
    assert report["science_position_angles_deg"] == [174.0]
    assert report["target_associated_flats"] == 4
    assert report["target_associated_arcs"] == 2
    assert report["matching_bias_frames"] == 10
    assert report["gates"]["raw_acquisition_gate_passed"] is True
    assert report["authorization"]["freeze_reduction_and_covariance_protocol"] is True
    assert report["gates"]["support_36kpc_demonstrated_by_accepted_kinematics"] is False
    assert report["gates"]["science_reduction_authorized"] is False
    assert report["gates"]["gravity_response_fit_authorized"] is False
