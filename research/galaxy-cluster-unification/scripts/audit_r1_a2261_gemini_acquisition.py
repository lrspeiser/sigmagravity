#!/usr/bin/env python3
"""Verify checksums, headers, and the frozen pre-pixel A2261 support target."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from astropy.io import fits


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw/r1_a2261_gemini"
CONFIG_PATH = ROOT / "configs/r1_a2261_gemini_acquisition_protocol.json"
REPORT_PATH = ROOT / "results/r1_a2261_gemini_acquisition/report.json"


def digest(path: Path, algorithm: str) -> str:
    value = hashlib.new(algorithm)
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest().upper()


def verify_provenance(path: Path, expected_names: list[str]) -> dict[str, dict]:
    provenance = json.loads(path.read_text(encoding="utf-8"))
    records = {record["archive_name"]: record for record in provenance["records"]}
    if list(records) != expected_names:
        raise RuntimeError(f"Frozen file order or contents mismatch in {path}")
    for record in records.values():
        local = ROOT / record["local_path"]
        if local.stat().st_size != record["local_size_bytes"]:
            raise RuntimeError(f"Size mismatch for {local}")
        if digest(local, "sha256") != record["local_sha256"]:
            raise RuntimeError(f"SHA256 mismatch for {local}")
        if digest(local, "md5") != record["local_md5"]:
            raise RuntimeError(f"MD5 mismatch for {local}")
    return records


def primary(name: str):
    return fits.getheader(RAW / name, 0)


def build_audit() -> dict:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    program_names = config["exact_target_associated_download"]
    bias_names = config["calibration_gate"]["exact_bias_download"]
    program = verify_provenance(RAW / "provenance.json", program_names)
    biases = verify_provenance(RAW / "bias_provenance.json", bias_names)
    science_names = config["science_expectation"]["science_filenames"]
    science_exposure = 0.0
    central_wavelengths = set()
    position_angles = set()
    for name in science_names:
        header = primary(name)
        ext = fits.getheader(RAW / name, 1)
        if not (
            header["OBJECT"] == "Abell 2261"
            and header["GEMPRGID"] == config["program_id"]
            and header["INSTRUME"] == "GMOS-N"
            and header["OBSCLASS"] == "science"
            and header["OBSTYPE"] == "OBJECT"
            and header["GRATING"].startswith("B600")
            and header["MASKNAME"] == "0.75arcsec"
            and ext["CCDSUM"] == "2 2"
        ):
            raise RuntimeError(f"Frozen science metadata mismatch for {name}")
        science_exposure += float(header["EXPTIME"])
        central_wavelengths.add(float(header["CENTWAVE"]) / 1000.0)
        position_angles.add(float(header["PA"]))

    flats = [name for name in program_names if primary(name)["OBSTYPE"] == "FLAT"]
    arcs = [name for name in program_names if primary(name)["OBSTYPE"] == "ARC"]
    acquisitions = [name for name in program_names if primary(name)["OBSCLASS"] == "acq"]
    for name in bias_names:
        header = primary(name)
        if not (
            header["INSTRUME"] == "GMOS-N"
            and header["OBSCLASS"] == "dayCal"
            and header["OBSTYPE"] == "BIAS"
            and fits.getheader(RAW / name, 1)["CCDSUM"] == "2 2"
        ):
            raise RuntimeError(f"Frozen bias metadata mismatch for {name}")

    expectation = config["science_expectation"]
    overlap = config["pre_pixel_overlap_target"]
    metadata_gate = bool(
        len(program) == config["download_checks"]["exact_target_associated_file_count"]
        and len(science_names) == expectation["exact_science_frames"]
        and abs(science_exposure - expectation["archive_total_exposure_seconds"]) <= expectation["archive_exposure_tolerance_seconds"]
        and central_wavelengths == set(expectation["central_wavelengths_um"])
        and position_angles == {expectation["published_slit_position_angle_deg"]}
        and len(flats) == config["calibration_gate"]["target_associated_flats_required"]
        and len(arcs) == config["calibration_gate"]["target_associated_arcs_required"]
        and len(biases) >= config["calibration_gate"]["bias_frame_minimum"]
    )
    report = {
        "report_version": config["protocol_version"],
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "selection_blind": True,
        "program_id": config["program_id"],
        "target_associated_files": len(program),
        "science_frames": len(science_names),
        "published_total_exposure_seconds_nominal": expectation["published_total_exposure_seconds_nominal"],
        "fits_science_exposure_seconds": science_exposure,
        "science_position_angles_deg": sorted(position_angles),
        "central_wavelengths_um": sorted(central_wavelengths),
        "target_associated_flats": len(flats),
        "target_associated_arcs": len(arcs),
        "acquisition_frames": len(acquisitions),
        "matching_bias_frames": len(biases),
        "pre_pixel_overlap_target": overlap,
        "total_local_raw_bytes": sum((ROOT / r["local_path"]).stat().st_size for r in [*program.values(), *biases.values()]),
        "gates": {
            "download_checksums_passed": True,
            "fits_header_metadata_passed": metadata_gate,
            "raw_acquisition_gate_passed": metadata_gate,
            "support_36kpc_demonstrated_by_accepted_kinematics": False,
            "reduction_protocol_frozen": False,
            "science_reduction_authorized": False,
            "gravity_response_fit_authorized": False
        },
        "next_action": "If the raw gate passes, freeze detector processing, signed bins reaching 36.0 kpc, S/N floors, sky subtraction, LSF matching, bootstrap blocks, systematic grid, and all acceptance thresholds before reducing a science frame.",
        "authorization": {
            "freeze_reduction_and_covariance_protocol": metadata_gate,
            "start_science_reduction": False,
            "fit_stellar_kinematics": False,
            "infer_dynamical_response": False,
            "fit_gravity_response": False,
            "fit_new_force_or_action": False
        }
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    print(json.dumps(build_audit(), indent=2))
