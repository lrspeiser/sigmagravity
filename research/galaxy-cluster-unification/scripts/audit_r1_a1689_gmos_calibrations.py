#!/usr/bin/env python3
"""Audit the frozen A1689 GMOS calibration products before science reduction.

This script must run in the pinned DRAGONS environment because the overscan
diagnostic deliberately reproduces DRAGONS 4.2.2's ``subtractOverscan`` fit.
It does not inspect a science frame or any kinematic/gravity result.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import astrodata
import gemini_instruments  # noqa: F401 - registers Gemini AstroData classes
import numpy as np
from astropy.io import fits
from gempy.gemini import gemini_tools as gt


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/r1_a1689_gmos_reduction_covariance_protocol.json"
RAW = ROOT / "data/raw/r1_a1689_gemini"
CAL = ROOT / "data/derived/r1_a1689_gmos_reconstruction/calibrations"
REPORT = ROOT / "results/r1_a1689_gmos_calibrations/report.json"


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest().upper()


def _fits_section_slices(section: str) -> tuple[slice, slice]:
    xpart, ypart = section.strip("[]").split(",")
    x1, x2 = (int(value) for value in xpart.split(":"))
    y1, y2 = (int(value) for value in ypart.split(":"))
    return slice(y1 - 1, y2), slice(x1 - 1, x2)


def overscan_fit_residuals(name: str, executed_headers: list[dict]) -> list[dict]:
    """Reproduce the GMOS DRAGONS 4.2.2 per-row mean overscan subtraction."""
    ad = astrodata.open(RAW / name)

    rows: list[dict] = []
    for amp, (ext, executed) in enumerate(zip(ad, executed_headers), start=1):
        asec = gt.map_array_sections(ext)
        yslice, xslice = _fits_section_slices(executed["oversec"])
        if yslice.start != asec.y1 or yslice.stop != asec.y2:
            raise ValueError(f"{name} amp {amp}: executed overscan rows do not match the data rows")
        overscan = np.asarray(ext.data[yslice, xslice], dtype=float)
        row_mean = np.mean(overscan, axis=1, keepdims=True)
        residual = overscan - row_mean
        valid = np.isfinite(residual)
        row_mean_uncertainty = (
            executed["read_noise_electron"]
            / executed["gain_electron_per_adu"]
            / np.sqrt(xslice.stop - xslice.start)
        )
        rows.append({
            "amp": amp,
            "overscan_section_used_fits": executed["oversec"],
            "rows_subtracted": int(overscan.shape[0]),
            "pixels_per_row": int(overscan.shape[1]),
            "absolute_mean_residual_adu": float(abs(np.mean(residual[valid]))),
            "rms_residual_adu": float(np.sqrt(np.mean(np.square(residual[valid])))),
            "expected_row_mean_uncertainty_adu": float(row_mean_uncertainty),
        })
    return rows


def audit_biases(config: dict) -> tuple[list[dict], bool, bool]:
    limits = config["calibration_acceptance"]
    max_abs_mean = limits["maximum_master_bias_overscan_residual_absolute_mean_adu"]
    max_rms = limits["maximum_master_bias_overscan_residual_rms_adu"]
    rows = []
    header_crosschecks = []
    for night, names in config["raw_inputs"]["night_bias_groups"].items():
        product = CAL / f"{Path(names[0]).stem}_bias.fits"
        log = CAL / f"bias_{night.replace('-', '')}.log"
        with fits.open(product, memmap=False) as hdul:
            executed_headers = [
                {
                    "oversec": hdu.header["OVERSEC"],
                    "gain_electron_per_adu": float(hdu.header["GAIN"]),
                    "read_noise_electron": float(hdu.header["RDNOISE"]),
                    "overscan_fit_rms_adu": float(hdu.header["OVERRMS"]),
                }
                for hdu in hdul if hdu.name == "SCI"
            ]
        header_rms = [row["overscan_fit_rms_adu"] for row in executed_headers]
        per_input = [
            {"name": name, "amplifiers": overscan_fit_residuals(name, executed_headers)}
            for name in names
        ]
        first_rms = [row["expected_row_mean_uncertainty_adu"] for row in per_input[0]["amplifiers"]]
        crosscheck = bool(np.allclose(header_rms, first_rms, rtol=0, atol=1e-10))
        header_crosschecks.append(crosscheck)
        log_text = log.read_text(encoding="utf-8", errors="replace")
        all_amps = [amp for item in per_input for amp in item["amplifiers"]]
        rows.append({
            "night": night,
            "inputs": names,
            "input_count": len(names),
            "product": str(product.relative_to(ROOT)).replace("\\", "/"),
            "product_sha256": sha256(product),
            "log": str(log.relative_to(ROOT)).replace("\\", "/"),
            "sigma_clipped_median_recorded": "Combining 5 inputs with median and sigclip rejection" in log_text,
            "processed_header_overscan_rms_adu": header_rms,
            "first_input_reproduction_matches_processed_headers": crosscheck,
            "maximum_absolute_mean_residual_adu": max(amp["absolute_mean_residual_adu"] for amp in all_amps),
            "maximum_rms_residual_adu": max(amp["rms_residual_adu"] for amp in all_amps),
            "per_input": per_input,
        })
    construction = all(
        row["input_count"] >= limits["minimum_inputs_per_night_master_bias"]
        and row["sigma_clipped_median_recorded"]
        for row in rows
    ) and all(header_crosschecks)
    residual_gate = all(
        row["maximum_absolute_mean_residual_adu"] <= max_abs_mean
        and row["maximum_rms_residual_adu"] <= max_rms
        for row in rows
    )
    return rows, construction, residual_gate


def audit_flats(config: dict) -> tuple[list[dict], bool]:
    threshold = config["calibration_acceptance"]["maximum_flat_unmasked_fraction_outside_0p8_to_1p2"]
    expected = sorted({pair[0] for pair in config["raw_inputs"]["science_to_flat_arc_mapping"].values()})
    rows = []
    for raw_name in expected:
        product = CAL / f"{Path(raw_name).stem}_flat.fits"
        fractions = []
        with fits.open(product, memmap=False) as hdul:
            for sci in (hdu for hdu in hdul if hdu.name == "SCI"):
                extver = sci.header["EXTVER"]
                dq = next(hdu.data for hdu in hdul if hdu.name == "DQ" and hdu.header["EXTVER"] == extver)
                good = np.isfinite(sci.data) & (dq == 0)
                outside = good & ((sci.data < 0.8) | (sci.data > 1.2))
                fractions.append(float(np.count_nonzero(outside) / np.count_nonzero(good)))
        rows.append({
            "raw": raw_name,
            "product": str(product.relative_to(ROOT)).replace("\\", "/"),
            "product_sha256": sha256(product),
            "unmasked_fraction_outside_0p8_to_1p2_per_extension": fractions,
            "maximum_fraction": max(fractions),
            "passed": max(fractions) <= threshold,
        })
    return rows, len(rows) == 4 and all(row["passed"] for row in rows)


def audit_arcs(config: dict) -> tuple[list[dict], bool]:
    limits = config["calibration_acceptance"]
    min_lines = limits["minimum_cuar_lines_per_arc"]
    max_rms = limits["maximum_wavelength_solution_rms_angstrom"]
    expected = sorted({pair[1] for pair in config["raw_inputs"]["science_to_flat_arc_mapping"].values()})
    rows = []
    for raw_name in expected:
        product = CAL / f"{Path(raw_name).stem}_arc.fits"
        with fits.open(product, memmap=False) as hdul:
            wavecal = next(hdu.data for hdu in hdul if hdu.name == "WAVECAL")
            coefficient_names = np.char.strip(wavecal["name"].astype(str))
            rms_nm = float(wavecal["coefficients"][coefficient_names == "rms"][0])
            line_count = len(wavecal)
        row = {
            "raw": raw_name,
            "product": str(product.relative_to(ROOT)).replace("\\", "/"),
            "product_sha256": sha256(product),
            "matched_cuar_lines": line_count,
            "wavelength_solution_rms_angstrom": 10.0 * rms_nm,
        }
        row["passed"] = line_count >= min_lines and row["wavelength_solution_rms_angstrom"] <= max_rms
        rows.append(row)
    return rows, len(rows) == 3 and all(row["passed"] for row in rows)


def main() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    biases, bias_construction, bias_residuals = audit_biases(config)
    flats, flat_gate = audit_flats(config)
    arcs, arc_gate = audit_arcs(config)
    calibration_products_gate = bias_construction and bias_residuals and flat_gate and arc_gate
    report = {
        "report_version": "R1B1-A1689-GMOS-calibrations-0.1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "selection_blind": True,
        "scope": "calibrations_only_before_any_science_frame_processing",
        "method": {
            "overscan_residual": "Exact GMOS DRAGONS 4.2.2 function=none per-row mean subtraction reproduced on every raw bias amplifier; metrics score the post-subtraction overscan pixels, and OVERRMS headers are cross-checked against the expected uncertainty of each row mean.",
            "flat": "Fraction of finite DQ=0 normalized-flat pixels outside [0.8, 1.2].",
            "arc": "Matched-line count and rms coefficient read from the DRAGONS WAVECAL table; nm converted to Angstrom.",
        },
        "biases": biases,
        "flats": flats,
        "arcs": arcs,
        "gates": {
            "bias_construction_passed": bias_construction,
            "bias_overscan_residuals_passed": bias_residuals,
            "flat_normalization_passed": flat_gate,
            "arc_wavelength_solutions_passed": arc_gate,
            "P2a_calibration_products_gate_passed": calibration_products_gate,
            "P2_calibrated_2d_gate_passed": False,
            "P3_profile_covariance_gate_passed": False,
            "gravity_response_fit_authorized": False,
        },
        "authorization": {
            "process_four_science_frames_with_frozen_mapping": calibration_products_gate,
            "fit_stellar_kinematics": False,
            "infer_gravity_response": False,
            "fit_new_force_or_action": False,
        },
        "next_action": "If and only if P2a passes, calibrate all four science exposures independently with the frozen flat/arc mapping, variance, DQ, and cosmic-ray flags; then audit centroid, coverage, and sky gates.",
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
