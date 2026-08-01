#!/usr/bin/env python3
"""Inventory the exact public KCWI night needed for SDSS J1402+6321."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from pyvo.dal import tap


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = ROOT / "data/derived/r1_j1402_kcwi_night_inventory.csv"
REPORT_PATH = ROOT / "results/r1_j1402_kcwi_metadata/report.json"
TAP_URL = "https://koa.ipac.caltech.edu/TAP"
TARGET = "SDSSJ1402+6321"
DATE_TOKEN = "20220408"
PROGRAM = "U020"
SCIENCE_IDS = {
    "KB.20220408.44130.47.fits",
    "KB.20220408.46107.23.fits",
    "KB.20220408.48082.97.fits",
    "KB.20220408.50056.67.fits",
}

COLUMNS = [
    "koaid",
    "object",
    "targname",
    "koaimtyp",
    "imtype",
    "ra",
    "dec",
    "utdatetime",
    "date_obs",
    "elaptime",
    "filesize_mb",
    "waveblue",
    "wavered",
    "camera",
    "bgratnam",
    "bfiltnam",
    "ifunam",
    "binning",
    "ampmode",
    "ccdspeed",
    "gainmode",
    "progid",
    "progpi",
    "progtitl",
    "semester",
    "ofname",
    "filehand",
    "lmp0nam",
    "lmp0stat",
    "lmp1nam",
    "lmp1stat",
    "lmp2nam",
    "lmp2stat",
]


def clean_text(series: pd.Series) -> pd.Series:
    return series.astype("string").fillna("").str.strip()


def classify(frame: pd.DataFrame) -> pd.Series:
    image_type = clean_text(frame["koaimtyp"]).str.lower()
    header_type = clean_text(frame["imtype"]).str.lower()
    objects = clean_text(frame["object"]).str.lower()
    combined = image_type + " " + header_type + " " + objects
    category = pd.Series("other", index=frame.index, dtype=object)
    category.loc[combined.str.contains("bias", regex=False)] = "bias"
    category.loc[combined.str.contains("cont", regex=False) | combined.str.contains("bar", regex=False)] = "continuum_bar"
    category.loc[combined.str.contains("flat", regex=False)] = "flat"
    category.loc[image_type.eq("arclamp") | header_type.eq("arclamp")] = "arc"
    category.loc[image_type.isin(["object", "science"])] = "object"
    return category


def build_report() -> dict:
    query = (
        f"SELECT {','.join(COLUMNS)} FROM koa_kcwi "
        f"WHERE koaid LIKE '%{DATE_TOKEN}%' ORDER BY utdatetime"
    )
    service = tap.TAPService(TAP_URL)
    frame = service.run_sync(query, maxrec=10000).to_table().to_pandas()
    frame.columns = [str(column).lower() for column in frame.columns]
    frame["category"] = classify(frame)
    frame["is_target_science"] = frame["koaid"].astype(str).isin(SCIENCE_IDS)

    target = frame.loc[frame["is_target_science"]].copy()
    if set(target["koaid"].astype(str)) != SCIENCE_IDS:
        missing = sorted(SCIENCE_IDS - set(target["koaid"].astype(str)))
        raise RuntimeError(f"Missing exact target science metadata: {missing}")

    setup_columns = [
        "camera",
        "bgratnam",
        "bfiltnam",
        "ifunam",
        "binning",
        "ampmode",
        "ccdspeed",
        "gainmode",
        "waveblue",
        "wavered",
    ]
    target_setup: dict[str, object] = {}
    setup_consistent = True
    for column in setup_columns:
        values = sorted(set(clean_text(target[column])))
        target_setup[column] = values
        setup_consistent &= len(values) == 1

    optical_match = pd.Series(True, index=frame.index)
    for column in ("camera", "bgratnam", "bfiltnam", "ifunam", "binning"):
        target_value = clean_text(target[column]).iloc[0]
        optical_match &= clean_text(frame[column]).eq(target_value)
    detector_match = optical_match.copy()
    for column in ("ampmode", "ccdspeed", "gainmode"):
        target_value = clean_text(target[column]).iloc[0]
        detector_match &= clean_text(frame[column]).eq(target_value)
    frame["target_optical_configuration_match"] = optical_match
    frame["target_detector_configuration_match"] = detector_match
    frame["target_configuration_match"] = detector_match

    calibrations = frame.loc[
        (
            (frame["category"].eq("bias") & detector_match)
            | (
                frame["category"].isin(["continuum_bar", "arc", "flat"])
                & optical_match
            )
        )
    ].copy()
    category_counts = {
        category: int((calibrations["category"] == category).sum())
        for category in ("bias", "continuum_bar", "arc", "flat")
    }
    calibration_minimum_gate = bool(
        category_counts["bias"] >= 5
        and category_counts["continuum_bar"] >= 1
        and category_counts["arc"] >= 1
        and category_counts["flat"] >= 1
    )

    frame = frame.sort_values(["utdatetime", "koaid"], kind="stable").reset_index(drop=True)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUTPUT_PATH, index=False, lineterminator="\n")

    report = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": "R1-J1402-KCWI-metadata-0.1",
        "metadata_only": True,
        "science_arrays_downloaded": False,
        "archive": {
            "service": TAP_URL,
            "table": "koa_kcwi",
            "date_token": DATE_TOKEN,
            "public_night_rows": len(frame),
        },
        "target": {
            "name": TARGET,
            "program": PROGRAM,
            "exact_science_ids": sorted(SCIENCE_IDS),
            "science_frame_count": len(target),
            "science_exposure_seconds": float(
                pd.to_numeric(target["elaptime"], errors="coerce").sum()
            ),
            "science_size_mb": float(
                pd.to_numeric(target["filesize_mb"], errors="coerce").sum()
            ),
            "setup": target_setup,
            "setup_consistent": bool(setup_consistent),
        },
        "same_configuration_calibrations": {
            "counts": category_counts,
            "minimum_gate_pass": calibration_minimum_gate,
            "exact_ids": {
                category: calibrations.loc[
                    calibrations["category"] == category, "koaid"
                ].astype(str).tolist()
                for category in category_counts
            },
            "total_size_mb": float(
                pd.to_numeric(calibrations["filesize_mb"], errors="coerce").sum()
            ),
        },
        "decision": (
            "calibration_identity_gate_pass_freeze_exact_acquisition_protocol"
            if setup_consistent and calibration_minimum_gate
            else "calibration_identity_shortfall_do_not_download_kcwi_arrays"
        ),
        "output": str(OUTPUT_PATH.relative_to(ROOT)).replace("\\", "/"),
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    print(json.dumps(build_report(), indent=2))


if __name__ == "__main__":
    main()
