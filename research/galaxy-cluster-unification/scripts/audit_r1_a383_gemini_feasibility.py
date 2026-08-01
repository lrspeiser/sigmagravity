#!/usr/bin/env python3
"""Run the frozen, metadata-only A383 GMOS feasibility gate."""

from __future__ import annotations

import json
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd
from astroquery.gemini import Observations


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/r1_a383_gemini_feasibility_protocol.json"
LENS_PATH = ROOT / "data/derived/r1_strong_lens_radial_support.csv"
RANK_PATH = ROOT / "data/derived/r1_lensing_geometric_rank.csv"
REPORT_PATH = ROOT / "results/r1_a383_gemini_feasibility/report.json"

META_FIELDS = [
    "name", "object", "program_id", "observation_id", "data_label", "ut_datetime",
    "observation_class", "observation_type", "mode", "exposure_time", "qa_state",
    "detector_binning", "detector_readspeed_setting", "detector_gain_setting",
    "detector_roi_setting", "disperser", "central_wavelength", "focal_plane_mask",
    "cass_rotator_pa", "file_size", "data_size", "file_md5", "data_md5",
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


def parse_time(row) -> datetime:
    value = scalar(row["ut_datetime"])
    return datetime.fromisoformat(str(value).replace(" ", "T"))


def build_report() -> dict:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    science_cfg = config["science_selection"]
    cal_cfg = config["calibration_selection"]

    program_table = Observations.query_criteria(program_id=config["program_id"], raw_reduced="RAW")
    program_rows = {str(row["name"]): row for row in program_table}
    bias_table = Observations.query_criteria(
        utc_date=(date(2007, 10, 10), date(2007, 10, 10)),
        instrument="GMOS-S",
        observation_type="BIAS",
        raw_reduced="RAW",
    )
    bias_rows = {str(row["name"]): row for row in bias_table}

    science_names = science_cfg["science_filenames"]
    flat_names = cal_cfg["exact_flat_download"]
    arc_names = cal_cfg["exact_arc_download"]
    bias_names = cal_cfg["exact_bias_download"]
    expected_program = [*science_names, *flat_names, *arc_names]
    missing_program = sorted(set(expected_program) - set(program_rows))
    missing_bias = sorted(set(bias_names) - set(bias_rows))

    science_metadata = []
    exposure = 0.0
    science_pass = not missing_program
    for name in science_names:
        if name not in program_rows:
            continue
        row = program_rows[name]
        exposure += float(row["exposure_time"])
        science_metadata.append(record(row))
        science_pass &= bool(
            str(row["object"]) == config["archive_object"]
            and str(row["program_id"]) == config["program_id"]
            and str(row["observation_class"]) == "science"
            and str(row["observation_type"]) == "OBJECT"
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
        and abs(exposure - science_cfg["selected_archive_exposure_seconds"]) < 1.0e-6
        and exposure >= science_cfg["minimum_selected_exposure_seconds"]
    )

    flat_metadata = []
    flat_pass = not missing_program
    for name in flat_names:
        if name not in program_rows:
            continue
        row = program_rows[name]
        flat_metadata.append(record(row))
        flat_pass &= bool(
            str(row["observation_type"]) == "FLAT"
            and str(row["qa_state"]) in cal_cfg["flat_qa_allowed_at_feasibility"]
            and str(row["detector_binning"]) == science_cfg["binning"]
            and str(row["detector_readspeed_setting"]) == science_cfg["read_speed"]
            and str(row["detector_gain_setting"]) == science_cfg["gain"]
            and str(row["detector_roi_setting"]) == science_cfg["roi"]
            and str(row["disperser"]) == science_cfg["grating"]
            and str(row["focal_plane_mask"]) == science_cfg["slit"]
            and float(row["central_wavelength"]) in science_cfg["central_wavelengths_um"]
        )

    arc_metadata = []
    arc_pass = not missing_program
    for name in arc_names:
        if name not in program_rows:
            continue
        row = program_rows[name]
        arc_metadata.append(record(row))
        arc_pass &= bool(
            str(row["observation_type"]) == "ARC"
            and str(row["qa_state"]) == cal_cfg["arc_qa_required"]
            and str(row["detector_binning"]) == science_cfg["binning"]
            and str(row["detector_readspeed_setting"]) == science_cfg["read_speed"]
            and str(row["detector_gain_setting"]) == science_cfg["gain"]
            and str(row["detector_roi_setting"]) == science_cfg["roi"]
            and str(row["disperser"]) == science_cfg["grating"]
            and str(row["focal_plane_mask"]) == science_cfg["slit"]
            and float(row["central_wavelength"]) in science_cfg["central_wavelengths_um"]
        )

    bias_metadata = []
    bias_pass = not missing_bias
    for name in bias_names:
        if name not in bias_rows:
            continue
        row = bias_rows[name]
        bias_metadata.append(record(row))
        bias_pass &= bool(
            str(row["observation_type"]) == "BIAS"
            and str(row["qa_state"]) == cal_cfg["bias_qa_required"]
            and str(row["detector_binning"]) == science_cfg["binning"]
            and str(row["detector_readspeed_setting"]) == science_cfg["read_speed"]
            and str(row["detector_gain_setting"]) == science_cfg["gain"]
            and str(row["detector_roi_setting"]) == science_cfg["roi"]
        )
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
        bias_separation = min(
            abs((parse_time(science_row) - parse_time(bias_rows[name])).total_seconds()) / 86400.0
            for name in bias_names if name in bias_rows
        )
        flat_separation = abs((parse_time(science_row) - parse_time(flat_row)).total_seconds()) / 86400.0
        arc_separation = abs((parse_time(science_row) - parse_time(arc_row)).total_seconds()) / 86400.0
        same_wavelength = float(science_row["central_wavelength"]) == float(flat_row["central_wavelength"]) == float(arc_row["central_wavelength"])
        passed = bool(
            same_wavelength
            and bias_separation <= cal_cfg["maximum_bias_separation_days"]
            and flat_separation <= cal_cfg["maximum_spectroscopy_flat_separation_days"]
            and arc_separation <= cal_cfg["maximum_arc_separation_days"]
        )
        mapping_pass &= passed
        separations.append({
            "science": science_name,
            "flat": flat_name,
            "arc": arc_name,
            "nearest_bias_separation_days": bias_separation,
            "flat_separation_days": flat_separation,
            "arc_separation_days": arc_separation,
            "central_wavelength_match": same_wavelength,
            "passed": passed,
        })

    lens = pd.read_csv(LENS_PATH)
    a383_lens = lens.loc[(lens["system"] == "A383") & lens["alternative_metric_likelihood_ready"]].sort_values("bcg_centric_radius_arcsec")
    expected_images = config["pre_pixel_overlap_target"]["preidentified_images_sorted_by_radius"]
    lens_pass = len(a383_lens) >= 3 and all(
        str(row.image_id) == expected["image_id"]
        and str(row.source_family) == expected["family_id"]
        and abs(float(row.bcg_centric_radius_arcsec) - expected["radius_arcsec"]) < 1.0e-10
        for (_, row), expected in zip(a383_lens.head(3).iterrows(), expected_images)
    )
    rank = pd.read_csv(RANK_PATH)
    rank_row = rank.loc[rank["system"] == "A383"].iloc[0]
    rank_pass = int(rank_row["structural_radial_rank_upper_bound"]) == 2 and str(rank_row["pilot_role"]) == "primary non-disturbed pilot"

    gates = {
        "exact_archive_files_exist": not missing_program and not missing_bias,
        "science_metadata_passed": science_pass,
        "flat_metadata_passed": flat_pass,
        "arc_metadata_passed": arc_pass,
        "bias_metadata_passed": bias_pass,
        "calibration_mapping_and_time_windows_passed": mapping_pass,
        "pre_pixel_lens_target_passed": lens_pass,
        "residual_blind_selection_rank_passed": rank_pass,
    }
    metadata_gate = all(gates.values())
    report = {
        "report_version": config["protocol_version"],
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "selection_blind": True,
        "science_pixels_downloaded_or_inspected": False,
        "system": config["system"],
        "program_id": config["program_id"],
        "selected_science_frames": len(science_metadata),
        "selected_science_exposure_seconds": exposure,
        "excluded_usable_science_frames": science_cfg["excluded_usable_science_frames"],
        "flat_frames": len(flat_metadata),
        "arc_frames": len(arc_metadata),
        "bias_frames": len(bias_metadata),
        "missing_program_files": missing_program,
        "missing_bias_files": missing_bias,
        "calibration_separations": separations,
        "pre_pixel_overlap_target": config["pre_pixel_overlap_target"],
        "archive_metadata": {
            "science": science_metadata,
            "flats": flat_metadata,
            "arcs": arc_metadata,
            "biases": bias_metadata,
        },
        "gates": {**gates, "metadata_feasibility_gate_passed": metadata_gate},
        "decision": "authorize_exact_raw_acquisition" if metadata_gate else "stop_A383_hard_archive_shortfall",
        "next_action": "Download only the frozen science/calibration/BPM files and verify checksums/headers; do not inspect pixels or reduce until a separate numerical reduction/covariance protocol is frozen." if metadata_gate else "Record the exact failed gate and select the next candidate without changing thresholds.",
        "authorization": {
            "download_exact_frozen_raw_files": metadata_gate,
            "inspect_science_pixels": False,
            "reduce_spectra": False,
            "fit_stellar_kinematics": False,
            "infer_dynamical_or_weyl_response": False,
            "fit_gravity_response": False,
            "fit_new_force_or_action": False,
        },
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    print(json.dumps(build_report(), indent=2))
