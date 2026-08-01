#!/usr/bin/env python3
"""Verify checksums and FITS metadata for the frozen A1689 GMOS-N raw set."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from astropy.io import fits


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw/r1_a1689_gemini"
CONFIG_PATH = ROOT / "configs/r1_a1689_gemini_acquisition_protocol.json"
PROGRAM_PROVENANCE = RAW / "provenance.json"
BIAS_PROVENANCE = RAW / "bias_provenance.json"
REPORT_PATH = ROOT / "results/r1_a1689_gemini_acquisition/report.json"


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
    for name, record in records.items():
        local = ROOT / record["local_path"]
        if local.stat().st_size != record["local_size_bytes"]:
            raise RuntimeError(f"Size mismatch for {local}")
        if digest(local, "sha256") != record["local_sha256"]:
            raise RuntimeError(f"SHA256 mismatch for {local}")
        if digest(local, "md5") != record["local_md5"]:
            raise RuntimeError(f"MD5 mismatch for {local}")
    return records


def primary_header(name: str):
    return fits.getheader(RAW / name, 0)


def build_audit() -> dict:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    program_names = config["exact_program_associated_download"]
    bias_names = config["calibration_gate"]["exact_bias_download"]
    program = verify_provenance(PROGRAM_PROVENANCE, program_names)
    biases = verify_provenance(BIAS_PROVENANCE, bias_names)

    science_names = config["science_expectation"]["science_filenames"]
    science_exposure = 0.0
    central_wavelengths = set()
    position_angles = set()
    for name in science_names:
        header = primary_header(name)
        if not (
            header["OBJECT"] == "Abell 1689"
            and header["GEMPRGID"] == config["program_id"]
            and header["INSTRUME"] == "GMOS-N"
            and header["OBSCLASS"] == "science"
            and header["OBSTYPE"] == "OBJECT"
            and header["GRATING"].startswith("B600")
            and header["MASKNAME"] == "0.75arcsec"
            and fits.getheader(RAW / name, 1)["CCDSUM"] == "2 2"
        ):
            raise RuntimeError(f"Frozen science metadata mismatch for {name}")
        science_exposure += float(header["EXPTIME"])
        central_wavelengths.add(float(header["CENTWAVE"]) / 1000.0)
        position_angles.add(float(header["PA"]))

    flats = [name for name in program_names if primary_header(name)["OBSTYPE"] == "FLAT"]
    arcs = [name for name in program_names if primary_header(name)["OBSTYPE"] == "ARC"]
    acquisitions = [name for name in program_names if primary_header(name)["OBSCLASS"] == "acq"]
    for name in bias_names:
        header = primary_header(name)
        if not (
            header["INSTRUME"] == "GMOS-N"
            and header["OBSCLASS"] == "dayCal"
            and header["OBSTYPE"] == "BIAS"
            and fits.getheader(RAW / name, 1)["CCDSUM"] == "2 2"
        ):
            raise RuntimeError(f"Frozen bias metadata mismatch for {name}")

    raw_gate = bool(
        len(program) == config["download_checks"]["exact_program_associated_file_count"]
        and len(science_names) == config["science_expectation"]["exact_science_frames"]
        and abs(science_exposure - config["science_expectation"]["exact_total_exposure_seconds_nominal"]) < 1.0
        and central_wavelengths == set(config["science_expectation"]["central_wavelengths_um"])
        and position_angles == {163.0}
        and len(flats) == config["calibration_gate"]["program_associated_flats_required"]
        and len(arcs) == config["calibration_gate"]["program_associated_arcs_required"]
        and len(biases) >= config["calibration_gate"]["bias_frame_minimum"]
    )
    report = {
        "report_version": "R1B1-A1689-Gemini-acquisition-0.1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "selection_blind": True,
        "program_id": config["program_id"],
        "program_associated_files": len(program),
        "science_frames": len(science_names),
        "science_exposure_seconds": science_exposure,
        "science_position_angles_deg": sorted(position_angles),
        "central_wavelengths_um": sorted(central_wavelengths),
        "program_associated_flats": len(flats),
        "program_associated_flat_files": flats,
        "program_associated_arcs": len(arcs),
        "program_associated_arc_files": arcs,
        "acquisition_frames": len(acquisitions),
        "matching_bias_frames": len(biases),
        "total_local_raw_bytes": sum((ROOT / record["local_path"]).stat().st_size for record in [*program.values(), *biases.values()]),
        "gates": {
            "download_checksums_passed": True,
            "fits_header_metadata_passed": True,
            "raw_acquisition_gate_passed": raw_gate,
            "reduction_protocol_frozen": False,
            "science_reduction_authorized": False,
            "gravity_response_fit_authorized": False,
        },
        "post_download_manifest_amendment": config["post_download_manifest_amendment"],
        "next_action": "Freeze the GMOS detector processing, sky subtraction, spatial-bin edges, stellar-template resolution matching, bootstrap blocks, and all numerical acceptance thresholds before reducing any A1689 spectrum.",
        "authorization": {
            "freeze_reduction_and_covariance_protocol": raw_gate,
            "start_science_reduction": False,
            "fit_stellar_kinematics": False,
            "infer_dynamical_response": False,
            "fit_gravity_response": False,
            "fit_new_force_or_action": False,
        },
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    print(json.dumps(build_audit(), indent=2))
