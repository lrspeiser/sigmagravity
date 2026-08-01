#!/usr/bin/env python3
"""Run in the isolated DRAGONS environment and audit A2261 raw recognition."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import astrodata
import gemini_instruments  # noqa: F401


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/r1_a2261_gmos_reduction_covariance_protocol.json"
RAW = ROOT / "data/raw/r1_a2261_gemini"
REPORT = ROOT / "results/r1_a2261_dragons_environment/report.json"


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest().upper()


def inspect(name: str) -> dict:
    ad = astrodata.open(RAW / name)
    return {
        "name": name,
        "tags": sorted(ad.tags),
        "extensions": len(ad),
        "instrument": ad.instrument(),
        "detector_name": ad.detector_name(pretty=True),
        "detector_binning": f"{ad.detector_x_bin()}x{ad.detector_y_bin()}",
        "detector_roi_setting": ad.detector_roi_setting(),
        "observation_type": ad.observation_type(),
        "observation_class": ad.observation_class()
    }


def main() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    mapping = config["raw_inputs"]["science_to_flat_arc_mapping"]
    science = [inspect(name) for name in mapping]
    flats = [inspect(name) for name in sorted({pair[0] for pair in mapping.values()})]
    arcs = [inspect(name) for name in sorted({pair[1] for pair in mapping.values()})]
    biases = [inspect(name) for names in config["raw_inputs"]["night_bias_groups"].values() for name in names]
    bpm_name = config["raw_inputs"]["bad_pixel_mask"]["selected_filename"]
    bpm = inspect(bpm_name)
    bpm_provenance = json.loads((RAW / "bpm_provenance.json").read_text())
    bpm_checksum = sha256(RAW / bpm_name) == bpm_provenance["selected"]["local_sha256"] == config["raw_inputs"]["bad_pixel_mask"]["selected_sha256"]
    packages = {}
    for name in ["dragons", "astrodata", "gemini-instruments", "numpy", "scipy", "astropy", "ppxf"]:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    expected = {
        "science": {"GMOS", "SPECT", "LS", "RAW"},
        "flat": {"GMOS", "SPECT", "LS", "FLAT", "RAW"},
        "arc": {"GMOS", "SPECT", "LS", "ARC", "RAW"},
        "bias": {"GMOS", "BIAS", "RAW"},
        "bpm": {"GMOS", "BPM"}
    }
    recognition = {
        "science": all(expected["science"].issubset(row["tags"]) for row in science),
        "flats": all(expected["flat"].issubset(row["tags"]) for row in flats),
        "arcs": all(expected["arc"].issubset(row["tags"]) for row in arcs),
        "biases": all(expected["bias"].issubset(row["tags"]) for row in biases),
        "bpm": expected["bpm"].issubset(bpm["tags"])
    }
    version_gate = packages["dragons"] == "4.2.2" and packages["ppxf"] == "9.4.8"
    raw_gate = all(recognition.values()) and bpm_checksum
    report = {
        "report_version": "R1B2-A2261-DRAGONS-environment-0.1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "selection_blind": True,
        "runtime": {"python": sys.version, "platform": platform.platform(), "packages": packages},
        "bpm_checksum_passed": bpm_checksum,
        "recognition_gates": recognition,
        "files": {"science": science, "flats": flats, "arcs": arcs, "biases": biases, "bpm": bpm},
        "gates": {
            "exact_software_versions_passed": version_gate,
            "all_raw_metadata_recognized_without_edits": raw_gate,
            "P1_environment_and_bpm_gate_passed": version_gate and raw_gate,
            "P2_calibrated_2d_gate_passed": False,
            "P3_profile_covariance_gate_passed": False,
            "gravity_response_fit_authorized": False
        },
        "authorization": {
            "execute_frozen_P2_calibration_reduction": version_gate and raw_gate,
            "fit_stellar_kinematics": False,
            "infer_gravity_response": False,
            "fit_new_force_or_action": False
        }
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
