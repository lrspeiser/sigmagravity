#!/usr/bin/env python3
"""Fit all frozen A1689 sky-window variants before any kinematic fit."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from astropy.io import fits


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/r1_a1689_gmos_reduction_covariance_protocol.json"
CAL2D_REPORT = ROOT / "results/r1_a1689_gmos_science_cal2d/report.json"
CENTER_REPORT = ROOT / "results/r1_a1689_gmos_continuum_center/report.json"
CENTER_DATA = ROOT / "data/derived/r1_a1689_gmos_reconstruction/continuum_center_profiles.npz"
OUT_ROOT = ROOT / "data/derived/r1_a1689_gmos_reconstruction/sky_models"
REPORT = ROOT / "results/r1_a1689_gmos_sky_models/report.json"
NO_DATA_BIT = np.uint16(16)


def variants(config: dict) -> dict[str, list[list[float]]]:
    sky = config["detector_and_2d_reduction"]["sky_model"]
    sensitivity = sky["sensitivity_windows_arcsec"]
    return {
        "inner": sensitivity[0],
        "baseline": sky["baseline_windows_arcsec_from_photometric_center"],
        "outer": sensitivity[1],
    }


def fit_variant(data: np.ndarray, variance: np.ndarray, dq: np.ndarray,
                offset: np.ndarray, windows: list[list[float]], execution: dict) -> dict:
    sky_rows = np.zeros(len(offset), dtype=bool)
    for lower, upper in windows:
        sky_rows |= (offset >= lower) & (offset <= upper)
    nwave = data.shape[1]
    coefficients = np.full((nwave, 2), np.nan, dtype=float)
    coefficient_covariance = np.full((nwave, 2, 2), np.nan, dtype=float)
    valid_counts = np.zeros(nwave, dtype=np.int32)
    residual_chunks = []
    min_rows = execution["minimum_valid_rows_per_wavelength"]
    niter = execution["clipping_iterations"]

    for column in range(nwave):
        use = (
            sky_rows & (dq[:, column] == 0)
            & np.isfinite(data[:, column]) & np.isfinite(variance[:, column])
            & (variance[:, column] > 0)
        )
        indices = np.flatnonzero(use)
        if len(indices) < min_rows:
            continue
        for _ in range(niter):
            x = offset[indices]
            y = data[indices, column]
            var = variance[indices, column]
            design = np.column_stack([np.ones(len(indices)), x])
            weight = 1.0 / var
            normal = design.T @ (weight[:, None] * design)
            if np.linalg.matrix_rank(normal) < 2:
                indices = np.array([], dtype=int)
                break
            cov = np.linalg.inv(normal)
            coeff = cov @ (design.T @ (weight * y))
            standardized = (y - design @ coeff) / np.sqrt(var)
            keep = np.abs(standardized) <= 4.0
            if keep.all():
                break
            if np.count_nonzero(keep) < min_rows:
                indices = np.array([], dtype=int)
                break
            indices = indices[keep]
        if len(indices) < min_rows:
            continue
        x = offset[indices]
        y = data[indices, column]
        var = variance[indices, column]
        design = np.column_stack([np.ones(len(indices)), x])
        weight = 1.0 / var
        normal = design.T @ (weight[:, None] * design)
        cov = np.linalg.inv(normal)
        coeff = cov @ (design.T @ (weight * y))
        residual = y - design @ coeff
        coefficients[column] = coeff
        coefficient_covariance[column] = cov
        valid_counts[column] = len(indices)
        residual_chunks.append(residual)

    residual = np.concatenate(residual_chunks) if residual_chunks else np.array([], dtype=float)
    median = float(np.median(residual)) if residual.size else float("nan")
    robust_sigma = float(1.4826 * np.median(np.abs(residual - median))) if residual.size else float("nan")
    normalized_median = abs(median) / robust_sigma if robust_sigma > 0 else float("inf")
    return {
        "coefficients": coefficients,
        "coefficient_covariance": coefficient_covariance,
        "valid_sky_rows_per_wavelength": valid_counts,
        "sky_row_count": int(np.count_nonzero(sky_rows)),
        "fitted_wavelength_count": int(np.count_nonzero(np.isfinite(coefficients[:, 0]))),
        "total_wavelength_count": nwave,
        "sky_residual_median_electron": median,
        "sky_residual_robust_sigma_electron": robust_sigma,
        "absolute_median_over_robust_sigma": float(normalized_median),
    }


def write_baseline(input_path: Path, output_path: Path, model: dict,
                   offset: np.ndarray, center: float) -> None:
    coeff = model["coefficients"]
    cov = model["coefficient_covariance"]
    finite = np.isfinite(coeff[:, 0])
    sky = coeff[None, :, 0] + offset[:, None] * coeff[None, :, 1]
    sky_var = (
        cov[None, :, 0, 0]
        + 2.0 * offset[:, None] * cov[None, :, 0, 1]
        + np.square(offset[:, None]) * cov[None, :, 1, 1]
    )
    with fits.open(input_path, memmap=False) as hdul:
        sci = next(hdu for hdu in hdul if hdu.name == "SCI")
        var = next(hdu for hdu in hdul if hdu.name == "VAR")
        dq = next(hdu for hdu in hdul if hdu.name == "DQ")
        sci.data[:, finite] = (sci.data[:, finite] - sky[:, finite]).astype(np.float32)
        var.data[:, finite] = (var.data[:, finite] + sky_var[:, finite]).astype(np.float32)
        dq.data[:, ~finite] |= NO_DATA_BIT
        hdul[0].header["SGSKY"] = ("baseline", "SigmaGravity frozen sky-window variant")
        hdul[0].header["SGCENTER"] = (center, "Joint continuum center offset (arcsec)")
        hdul[0].header.add_history("SigmaGravity: inverse-variance linear sky model, 4-sigma clipping")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        hdul.writeto(output_path, overwrite=True, checksum=True)


def main() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    cal2d = json.loads(CAL2D_REPORT.read_text(encoding="utf-8"))
    center_report = json.loads(CENTER_REPORT.read_text(encoding="utf-8"))
    if not center_report["authorization"]["execute_frozen_sky_window_variants"]:
        raise RuntimeError("The continuum-center gate did not authorize sky fitting")
    center = center_report["joint_fit"]["center_arcsec"]
    execution = config["spatial_extraction"]["sky_fit_execution"]
    all_variants = variants(config)
    threshold = config["calibration_acceptance"]["maximum_sky_residual_absolute_median_over_robust_sigma"]
    center_arrays = np.load(CENTER_DATA)
    rows = []

    for product in cal2d["products"]:
        science = product["science"]
        stem = Path(science).stem
        input_path = ROOT / product["product"]
        offset = np.asarray(center_arrays[f"{stem}_offset_arcsec"], dtype=float) - center
        with fits.open(input_path, memmap=False) as hdul:
            data = np.asarray(next(hdu.data for hdu in hdul if hdu.name == "SCI"), dtype=float)
            variance = np.asarray(next(hdu.data for hdu in hdul if hdu.name == "VAR"), dtype=float)
            dq = np.asarray(next(hdu.data for hdu in hdul if hdu.name == "DQ"), dtype=np.uint16)
        model_path = OUT_ROOT / f"{stem}_sky_models.npz"
        arrays = {"signed_offset_arcsec_from_joint_center": offset}
        summaries = []
        fitted = {}
        for name, windows in all_variants.items():
            model = fit_variant(data, variance, dq, offset, windows, execution)
            fitted[name] = model
            arrays[f"{name}_coefficients_intercept_slope"] = model["coefficients"]
            arrays[f"{name}_coefficient_covariance"] = model["coefficient_covariance"]
            arrays[f"{name}_valid_sky_rows_per_wavelength"] = model["valid_sky_rows_per_wavelength"]
            summaries.append({
                "variant": name,
                "windows_arcsec": windows,
                "sky_row_count": model["sky_row_count"],
                "fitted_wavelength_count": model["fitted_wavelength_count"],
                "total_wavelength_count": model["total_wavelength_count"],
                "sky_residual_median_electron": model["sky_residual_median_electron"],
                "sky_residual_robust_sigma_electron": model["sky_residual_robust_sigma_electron"],
                "absolute_median_over_robust_sigma": model["absolute_median_over_robust_sigma"],
                "passed": model["absolute_median_over_robust_sigma"] <= threshold,
            })
        model_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(model_path, **arrays)
        baseline_path = OUT_ROOT / "baseline" / f"{stem}_sky.fits"
        write_baseline(input_path, baseline_path, fitted["baseline"], offset, center)
        rows.append({
            "science": science,
            "input": str(input_path.relative_to(ROOT)).replace("\\", "/"),
            "model_output": str(model_path.relative_to(ROOT)).replace("\\", "/"),
            "baseline_sky_subtracted_output": str(baseline_path.relative_to(ROOT)).replace("\\", "/"),
            "variants": summaries,
            "all_variants_passed": all(item["passed"] for item in summaries),
        })

    gate = bool(len(rows) == 4 and all(row["all_variants_passed"] for row in rows))
    report = {
        "report_version": "R1B1-A1689-GMOS-sky-models-0.1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "selection_blind": True,
        "joint_continuum_center_arcsec": center,
        "sky_residual_threshold": threshold,
        "products": rows,
        "gates": {
            "P2c_continuum_centroid_range_gate_passed": center_report["gates"]["P2c_continuum_centroid_range_gate_passed"],
            "P2d_all_frozen_sky_variants_gate_passed": gate,
            "P2_calibrated_2d_sky_centroid_coverage_gate_passed": False,
            "P3_profile_covariance_gate_passed": False,
            "gravity_response_fit_authorized": False,
        },
        "authorization": {
            "register_and_combine_baseline_exposures_for_coverage_audit": gate,
            "fit_stellar_kinematics": False,
            "infer_gravity_response": False,
            "fit_new_force_or_action": False,
        },
        "next_action": (
            "Register the four baseline sky-subtracted frames by the frozen continuum center, combine on one common grid, and audit >=3-exposure coverage before any pPXF call."
            if gate else
            "Retain the failed frozen sky residual and keep A1689 geometry-only; do not change windows or clipping."
        ),
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
