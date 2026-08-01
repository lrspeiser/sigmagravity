#!/usr/bin/env python3
"""Verify A2537 raw checksums and FITS headers without reading science arrays."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from astropy.io import fits


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/r1_a2537_gemini_feasibility_protocol.json"
ACQUISITION_PROTOCOL_PATH = ROOT / "configs/r1_a2537_gemini_acquisition_protocol.json"
RAW = ROOT / "data/raw/r1_a2537_gemini"
PROVENANCE_PATH = RAW / "provenance.json"
REPORT_PATH = ROOT / "results/r1_a2537_gemini_acquisition/report.json"


def digest(path: Path, algorithm: str) -> str:
    value = hashlib.new(algorithm)
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest().upper()


def build_report() -> dict:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    acquisition_protocol = json.loads(ACQUISITION_PROTOCOL_PATH.read_text(encoding="utf-8"))
    provenance = json.loads(PROVENANCE_PATH.read_text(encoding="utf-8"))
    records = provenance["records"]
    science_names = config["science_selection"]["science_filenames"]
    flat_names = config["calibration_selection"]["exact_flat_download"]
    arc_names = config["calibration_selection"]["exact_arc_download"]
    bias_names = config["calibration_selection"]["exact_bias_download"]
    bpm_name = config["calibration_selection"]["required_bpm"]
    expected = [*science_names, *flat_names, *arc_names, *bias_names, bpm_name]

    exact_list_pass = [record["archive_name"] for record in records] == expected
    checksum_pass = True
    header_pass = True
    header_audit = []
    position_angles = []
    detectors = set()
    bpm_detector = None
    for record in records:
        path = ROOT / record["local_path"]
        checksum_pass &= bool(
            path.exists()
            and path.stat().st_size == record["local_size_bytes"]
            and digest(path, "md5") == record["local_md5"]
            and digest(path, "sha256") == record["local_sha256"]
        )
        with fits.open(path, memmap=False, lazy_load_hdus=True) as hdul:
            primary = hdul[0].header
            ext = hdul[1].header if len(hdul) > 1 else {}
        category = record["category"]
        metadata = record["metadata"]
        common_gmos = primary.get("INSTRUME") == "GMOS-S"
        ccdsum = ext.get("CCDSUM")
        if category == "science":
            header_pa = float(primary.get("PA"))
            rules = acquisition_protocol["header_rules"]
            passed = bool(
                primary.get("INSTRUME") == rules["instrument"]
                and primary.get("OBJECT") == rules["science_object"]
                and primary.get("GEMPRGID") == rules["program_id"]
                and primary.get("OBSCLASS") == rules["science_observation_class"]
                and primary.get("OBSTYPE") == rules["science_observation_type"]
                and str(primary.get("GRATING", "")).startswith(rules["science_grating_prefix"])
                and primary.get("MASKNAME") == rules["science_mask"]
                and ccdsum == rules["ccdsum"]
                and abs(header_pa - rules["expected_science_slit_pa_deg"]) < rules["science_slit_pa_tolerance_deg"]
            )
            position_angles.append(header_pa)
            detectors.add(str(primary.get("DETECTOR")))
        elif category == "flat":
            passed = bool(common_gmos and primary.get("OBSTYPE") == "FLAT" and ccdsum == "2 2")
        elif category == "arc":
            passed = bool(common_gmos and primary.get("OBSTYPE") == "ARC" and ccdsum == "2 2")
        elif category == "bias":
            passed = bool(common_gmos and primary.get("OBSTYPE") == "BIAS" and ccdsum == "2 2")
        else:
            bpm_detector = str(primary.get("DETECTOR"))
            passed = bool(
                common_gmos
                and primary.get("OBJECT") == "BPM"
                and primary.get("OBSTYPE") == "BPM"
                and "EEV" in record["archive_name"]
                and ccdsum == "2 2"
            )
        header_pass &= passed
        header_audit.append({
            "name": record["archive_name"],
            "category": category,
            "instrument": primary.get("INSTRUME"),
            "object": primary.get("OBJECT"),
            "observation_class": primary.get("OBSCLASS"),
            "observation_type": primary.get("OBSTYPE"),
            "detector": primary.get("DETECTOR"),
            "grating": primary.get("GRATING"),
            "central_wavelength_angstrom": primary.get("CENTWAVE"),
            "mask": primary.get("MASKNAME"),
            "position_angle_deg": primary.get("PA"),
            "ccdsum": ccdsum,
            "passed": passed,
        })

    detector_match_pass = bpm_detector is not None and detectors == {bpm_detector}
    gate = bool(exact_list_pass and checksum_pass and header_pass and detector_match_pass)
    report = {
        "report_version": acquisition_protocol["protocol_version"],
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "selection_blind": True,
        "disturbed_control": True,
        "counts_as_non_disturbed_pilot": False,
        "science_arrays_opened": False,
        "first_header_audit_semantic_correction": acquisition_protocol["first_header_audit"],
        "files": len(records),
        "local_raw_bytes": sum(record["local_size_bytes"] for record in records),
        "science_position_angles_deg": position_angles,
        "science_detectors": sorted(detectors),
        "header_audit": header_audit,
        "gates": {
            "exact_frozen_file_list_passed": exact_list_pass,
            "archive_md5_and_local_sha256_passed": checksum_pass,
            "fits_header_metadata_passed": header_pass,
            "science_and_bpm_detector_label_match_passed": detector_match_pass,
            "raw_acquisition_gate_passed": gate,
        },
        "decision": "authorize_disturbed_control_reduction_protocol_freeze" if gate else "stop_A2537_raw_integrity_failure",
        "next_action": "Freeze numerical detector, flat, arc, continuum-center, sky, signed extraction, pPXF, bootstrap, and systematic-grid gates before reading a science array; preserve the disturbed-control label." if gate else "Retain the data and exact failure; do not reduce or inspect science arrays.",
        "authorization": {
            "freeze_reduction_and_covariance_protocol": gate,
            "count_as_non_disturbed_pilot": False,
            "inspect_science_arrays": False,
            "reduce_spectra": False,
            "fit_stellar_kinematics": False,
            "infer_dynamical_or_weyl_response": False,
            "fit_new_force_or_action": False,
        },
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    print(json.dumps(build_report(), indent=2))
