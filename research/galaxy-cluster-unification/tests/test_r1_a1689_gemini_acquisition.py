from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_a1689_gemini_manifest_is_exact_and_auditable() -> None:
    config = json.loads((ROOT / "configs/r1_a1689_gemini_acquisition_protocol.json").read_text())
    assert config["frozen_before_raw_download"] is True
    assert len(config["exact_program_associated_download"]) == 17
    assert len(config["calibration_gate"]["exact_bias_download"]) == 10
    amendment = config["post_download_manifest_amendment"]
    assert amendment["original_value"] == 3
    assert amendment["corrected_value"] == 4
    assert amendment["scientific_values_inspected"] is False
    assert amendment["thresholds_changed"] is False


def test_a1689_gemini_raw_gate_passes_but_reduction_stays_closed() -> None:
    subprocess.run(
        [sys.executable, str(ROOT / "scripts/audit_r1_a1689_gemini_acquisition.py")],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    report = json.loads((ROOT / "results/r1_a1689_gemini_acquisition/report.json").read_text())
    assert report["program_associated_files"] == 17
    assert report["science_frames"] == 4
    assert abs(report["science_exposure_seconds"] - 7200.0) < 1.0
    assert report["science_position_angles_deg"] == [163.0]
    assert report["program_associated_flats"] == 4
    assert report["program_associated_arcs"] == 3
    assert report["matching_bias_frames"] == 10
    assert report["gates"]["raw_acquisition_gate_passed"] is True
    assert report["authorization"]["freeze_reduction_and_covariance_protocol"] is True
    assert report["gates"]["reduction_protocol_frozen"] is False
    assert report["gates"]["science_reduction_authorized"] is False
    assert report["gates"]["gravity_response_fit_authorized"] is False
