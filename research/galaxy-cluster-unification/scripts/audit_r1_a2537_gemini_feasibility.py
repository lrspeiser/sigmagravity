#!/usr/bin/env python3
"""Run the frozen, metadata-only A2537 Gemini feasibility gate."""

from __future__ import annotations

import hashlib
import json
import math
import re
from datetime import date, datetime, timezone
from pathlib import Path

import astropy.units as u
import pandas as pd
from astropy.coordinates import SkyCoord
from astroquery.gemini import Observations


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/r1_a2537_gemini_feasibility_protocol.json"
LENS_PATH = ROOT / "data/derived/r1_strong_lens_radial_support.csv"
RANK_PATH = ROOT / "data/derived/r1_lensing_geometric_rank.csv"
REPORT_PATH = ROOT / "results/r1_a2537_gemini_feasibility/report.json"

META_FIELDS = [
    "name", "object", "program_id", "observation_id", "data_label", "ut_datetime",
    "observation_class", "observation_type", "mode", "exposure_time", "qa_state",
    "ra", "dec", "detector_binning", "detector_readspeed_setting", "detector_gain_setting",
    "detector_roi_setting", "disperser", "central_wavelength", "focal_plane_mask",
    "cass_rotator_pa", "file_size", "file_md5",
]


def scalar(value):
    if getattr(value, "mask", False) is True:
        return None
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    return value


def record(row) -> dict:
    return {field: scalar(row[field]) for field in META_FIELDS}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def parse_time(row) -> datetime:
    return datetime.fromisoformat(str(scalar(row["ut_datetime"])).replace(" ", "T"))


def build_report() -> dict:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    science_cfg = config["science_selection"]
    cal_cfg = config["calibration_selection"]
    center_cfg = config["published_bcg_center"]
    source_path = ROOT / center_cfg["source_file"]
    source_text = source_path.read_text(encoding="utf-8")
    center_source_pass = bool(
        sha256(source_path) == center_cfg["source_file_sha256"]
        and re.search(r"Abell\\,2537\s*&23 08 22\.3&\$-\$02 11 32", source_text)
    )

    program = Observations.query_criteria(program_id=config["program_id"], raw_reduced="RAW")
    program_rows = {str(row["name"]): row for row in program}
    bias_frames = []
    for day in (date(2008, 9, 21), date(2008, 9, 22)):
        bias_frames.extend(Observations.query_criteria(
            utc_date=(day, day), instrument="GMOS-S", observation_type="BIAS", raw_reduced="RAW"
        ))
    bias_rows = {str(row["name"]): row for row in bias_frames}
    expected_program = science_cfg["science_filenames"] + cal_cfg["exact_flat_download"] + cal_cfg["exact_arc_download"]
    missing_program = sorted(set(expected_program) - set(program_rows))
    missing_bias = sorted(set(cal_cfg["exact_bias_download"]) - set(bias_rows))

    bcg = SkyCoord(center_cfg["ra_deg"] * u.deg, center_cfg["dec_deg"] * u.deg)
    science_metadata = []
    pointing_offsets = []
    exposure = 0.0
    science_pass = not missing_program
    for name in science_cfg["science_filenames"]:
        if name not in program_rows:
            continue
        row = program_rows[name]
        exposure += float(row["exposure_time"])
        science_metadata.append(record(row))
        pointing = SkyCoord(float(row["ra"]) * u.deg, float(row["dec"]) * u.deg)
        pointing_offsets.append(float(bcg.separation(pointing).arcsec))
        science_pass &= bool(
            str(row["object"]) == config["archive_object"]
            and str(row["program_id"]) == config["program_id"]
            and str(row["observation_class"]) == "science"
            and str(row["observation_type"]) == "OBJECT"
            and str(row["mode"]) == "LS"
            and str(row["qa_state"]) == science_cfg["archive_qa_required"]
            and str(row["detector_binning"]) == science_cfg["binning"]
            and str(row["detector_readspeed_setting"]) == science_cfg["read_speed"]
            and str(row["detector_gain_setting"]) == science_cfg["gain"]
            and str(row["detector_roi_setting"]) == science_cfg["roi"]
            and str(row["disperser"]) == science_cfg["grating"]
            and str(row["focal_plane_mask"]) == science_cfg["slit"]
            and float(row["central_wavelength"]) in science_cfg["central_wavelengths_um"]
        )
    science_pass &= bool(
        len(science_metadata) == science_cfg["exact_science_frames"]
        and math.isclose(exposure, science_cfg["selected_archive_exposure_seconds"], abs_tol=1.0e-6)
        and exposure >= science_cfg["minimum_selected_exposure_seconds"]
        and max(pointing_offsets) <= science_cfg["maximum_archive_pointing_offset_from_published_bcg_arcsec"]
    )

    def cal_pass(names: list[str], observation_type: str, qa: str, rows: dict) -> tuple[bool, list[dict]]:
        metadata = []
        passed = True
        for name in names:
            if name not in rows:
                passed = False
                continue
            row = rows[name]
            metadata.append(record(row))
            passed &= bool(
                str(row["observation_type"]) == observation_type
                and str(row["qa_state"]) == qa
                and str(row["detector_binning"]) == science_cfg["binning"]
                and str(row["detector_readspeed_setting"]) == science_cfg["read_speed"]
                and str(row["detector_gain_setting"]) == science_cfg["gain"]
                and str(row["detector_roi_setting"]) == science_cfg["roi"]
            )
            if observation_type in {"FLAT", "ARC"}:
                passed &= bool(
                    str(row["mode"]) == "LS"
                    and str(row["disperser"]) == science_cfg["grating"]
                    and str(row["focal_plane_mask"]) == science_cfg["slit"]
                    and float(row["central_wavelength"]) in science_cfg["central_wavelengths_um"]
                )
        return passed, metadata

    flat_pass, flat_metadata = cal_pass(cal_cfg["exact_flat_download"], "FLAT", cal_cfg["flat_qa_required"], program_rows)
    arc_pass, arc_metadata = cal_pass(cal_cfg["exact_arc_download"], "ARC", cal_cfg["arc_qa_required"], program_rows)
    bias_pass, bias_metadata = cal_pass(cal_cfg["exact_bias_download"], "BIAS", cal_cfg["bias_qa_required"], bias_rows)
    bias_pass &= len(bias_metadata) >= cal_cfg["minimum_bias_frames"]

    mapping_pass = True
    separations = []
    for science_name, (flat_name, arc_name) in cal_cfg["science_to_flat_arc_mapping"].items():
        if any(name not in program_rows for name in (science_name, flat_name, arc_name)):
            mapping_pass = False
            continue
        science_row = program_rows[science_name]
        flat_row = program_rows[flat_name]
        arc_row = program_rows[arc_name]
        bias_sep = min(abs((parse_time(science_row) - parse_time(bias_rows[name])).total_seconds()) / 86400 for name in cal_cfg["exact_bias_download"] if name in bias_rows)
        flat_sep = abs((parse_time(science_row) - parse_time(flat_row)).total_seconds()) / 86400
        arc_sep = abs((parse_time(science_row) - parse_time(arc_row)).total_seconds()) / 86400
        wavelength_match = float(science_row["central_wavelength"]) == float(flat_row["central_wavelength"]) == float(arc_row["central_wavelength"])
        passed = bool(wavelength_match and bias_sep <= cal_cfg["maximum_bias_separation_days"] and flat_sep <= cal_cfg["maximum_spectroscopy_flat_separation_days"] and arc_sep <= cal_cfg["maximum_arc_separation_days"])
        mapping_pass &= passed
        separations.append({"science": science_name, "flat": flat_name, "arc": arc_name, "nearest_bias_separation_days": bias_sep, "flat_separation_days": flat_sep, "arc_separation_days": arc_sep, "central_wavelength_match": wavelength_match, "passed": passed})

    lens = pd.read_csv(LENS_PATH)
    selected = lens.loc[(lens["system"] == config["system"]) & lens["alternative_metric_likelihood_ready"].astype(bool)].sort_values("bcg_centric_radius_arcsec").head(3)
    expected = config["pre_pixel_overlap_target"]["preidentified_images_sorted_by_radius"]
    lens_pass = len(selected) == 3 and all(
        str(row.image_id) == target["image_id"] and str(row.source_family) == target["family_id"] and math.isclose(float(row.bcg_centric_radius_arcsec), target["radius_arcsec"], abs_tol=1.0e-10)
        for (_, row), target in zip(selected.iterrows(), expected)
    )
    rank = pd.read_csv(RANK_PATH)
    rank_row = rank.loc[rank["system"] == config["system"]].iloc[0]
    rank_pass = bool(float(rank_row["pilot_priority"]) == 3.0 and str(rank_row["pilot_role"]) == "disturbed engineering control only" and str(rank_row["disturbed_control"]).lower() == "true")

    gates = {
        "published_bcg_center_source_traceable": center_source_pass,
        "exact_archive_files_exist": not missing_program and not missing_bias,
        "science_metadata_and_pointing_passed": science_pass,
        "flat_metadata_passed": flat_pass,
        "arc_metadata_passed": arc_pass,
        "bias_metadata_passed": bias_pass,
        "calibration_mapping_and_time_windows_passed": mapping_pass,
        "pre_pixel_lens_target_passed": lens_pass,
        "disturbed_control_rank_and_label_passed": rank_pass,
    }
    passed = all(gates.values())
    report = {
        "report_version": config["protocol_version"],
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "selection_blind": True,
        "science_pixels_downloaded_or_inspected": False,
        "system": config["system"],
        "disturbed_control": True,
        "program_id": config["program_id"],
        "selected_science_frames": len(science_metadata),
        "selected_science_exposure_seconds": exposure,
        "science_pointing_offsets_from_published_bcg_arcsec": pointing_offsets,
        "flat_frames": len(flat_metadata),
        "arc_frames": len(arc_metadata),
        "bias_frames": len(bias_metadata),
        "missing_program_files": missing_program,
        "missing_bias_files": missing_bias,
        "calibration_separations": separations,
        "pre_pixel_overlap_target": config["pre_pixel_overlap_target"],
        "archive_metadata": {"science": science_metadata, "flats": flat_metadata, "arcs": arc_metadata, "biases": bias_metadata},
        "gates": {**gates, "metadata_feasibility_gate_passed": passed},
        "decision": "authorize_exact_raw_acquisition_as_disturbed_control" if passed else "stop_A2537_hard_archive_shortfall",
        "next_action": "Download only the frozen science/calibration/BPM files and verify checksums/headers; do not inspect pixels or reduce until a separate numerical protocol is frozen. Never relabel A2537 as a non-disturbed pilot." if passed else "End this acquisition cycle and invoke the predeclared rethink rule without changing thresholds.",
        "authorization": {
            "download_exact_frozen_raw_files": passed,
            "count_as_non_disturbed_pilot": False,
            "inspect_science_pixels": False,
            "reduce_spectra": False,
            "fit_stellar_kinematics": False,
            "infer_dynamical_or_weyl_response": False,
            "fit_gravity_response": False,
            "fit_new_force_or_action": False,
        }
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    print(json.dumps(build_report(), indent=2))
