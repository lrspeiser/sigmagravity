#!/usr/bin/env python3
"""Measure the frozen wavelength-dependent A1689 GMOS LSF from CuAr lines."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.io import fits
from numpy.polynomial import Chebyshev
from scipy.optimize import least_squares


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/r1_a1689_gmos_reduction_covariance_protocol.json"
CAL_REPORT = ROOT / "results/r1_a1689_gmos_calibrations/report.json"
CAL = ROOT / "data/derived/r1_a1689_gmos_reconstruction/calibrations"
OUTPUT = ROOT / "data/derived/r1_a1689_gmos_lsf_lines.csv"
MODEL = ROOT / "data/derived/r1_a1689_gmos_lsf_model.npz"
REPORT = ROOT / "results/r1_a1689_gmos_lsf/report.json"


def wavelength_model(table) -> Chebyshev:
    names = np.char.strip(table["name"].astype(str))
    coefficients = [float(table["coefficients"][names == f"c{i}"][0]) for i in range(4)]
    domain = [
        float(table["coefficients"][names == "domain_start"][0]),
        float(table["coefficients"][names == "domain_end"][0]),
    ]
    return Chebyshev(coefficients, domain=domain)


def gaussian_linear(x: np.ndarray, background: float, slope: float, amplitude: float,
                    center: float, sigma: float, origin: float) -> np.ndarray:
    return background + slope * (x - origin) + amplitude * np.exp(-0.5 * np.square((x - center) / sigma))


def fit_arc(path: Path, execution: dict) -> list[dict]:
    with fits.open(path, memmap=False) as hdul:
        science = np.asarray(next(hdu.data for hdu in hdul if hdu.name == "SCI"), dtype=float)
        variance = np.asarray(next(hdu.data for hdu in hdul if hdu.name == "VAR"), dtype=float)
        wavecal = next(hdu.data for hdu in hdul if hdu.name == "WAVECAL")
    row0, row1 = execution["arc_extraction_rows_zero_based_half_open"]
    spectrum = np.sum(science[row0:row1], axis=0)
    spectrum_variance = np.sum(variance[row0:row1], axis=0)
    wave_model = wavelength_model(wavecal)
    half = execution["line_fit_half_width_pixels"]
    sigma_bounds = execution["gaussian_sigma_bounds_pixels"]
    rows = []
    for peak, reference_nm in zip(wavecal["peaks"], wavecal["wavelengths"]):
        peak = float(peak)
        reference_angstrom = 10.0 * float(reference_nm)
        lower = max(int(np.floor(peak)) - half, 0)
        upper = min(int(np.floor(peak)) + half + 1, len(spectrum))
        x = np.arange(lower, upper, dtype=float)
        y = spectrum[lower:upper]
        var = spectrum_variance[lower:upper]
        good = np.isfinite(y) & np.isfinite(var) & (var > 0)
        accepted = False
        reason = "insufficient_fit_pixels"
        result = None
        if np.count_nonzero(good) >= 9:
            xfit, yfit, vfit = x[good], y[good], var[good]
            background = float(np.median(np.r_[yfit[:3], yfit[-3:]]))
            amplitude = max(float(np.max(yfit) - background), np.finfo(float).eps)
            result = least_squares(
                lambda p: (gaussian_linear(xfit, *p, origin=peak) - yfit) / np.sqrt(vfit),
                x0=[background, 0.0, amplitude, peak, 2.0],
                bounds=(
                    [-np.inf, -np.inf, 0.0, peak - execution["gaussian_center_tolerance_pixels"], sigma_bounds[0]],
                    [np.inf, np.inf, np.inf, peak + execution["gaussian_center_tolerance_pixels"], sigma_bounds[1]],
                ),
            )
            accepted = bool(
                result.success
                and abs(result.x[3] - peak) <= execution["gaussian_center_tolerance_pixels"]
                and sigma_bounds[0] < result.x[4] < sigma_bounds[1]
                and result.x[2] > 0
            )
            reason = "accepted" if accepted else "fit_or_bound_failure"
        if result is None:
            center = sigma = fwhm_angstrom = fwhm_error = np.nan
            chi2 = np.nan
        else:
            center = float(result.x[3])
            sigma = float(result.x[4])
            dispersion = abs(float(wave_model.deriv()(center))) * 10.0
            fwhm_angstrom = 2.354820045 * sigma * dispersion
            dof = max(len(result.fun) - len(result.x), 1)
            chi2 = float(np.sum(np.square(result.fun)) / dof)
            covariance = np.linalg.pinv(result.jac.T @ result.jac) * chi2
            fwhm_error = 2.354820045 * dispersion * np.sqrt(max(covariance[4, 4], 0.0))
        rows.append({
            "arc": path.name,
            "reference_wavelength_angstrom": reference_angstrom,
            "catalog_peak_pixel": peak,
            "fitted_center_pixel": center,
            "fitted_sigma_pixel": sigma,
            "fwhm_angstrom": fwhm_angstrom,
            "fwhm_standard_error_angstrom": fwhm_error,
            "reduced_chi2": chi2,
            "accepted": accepted,
            "disposition": reason,
        })
    return rows


def wavecal_covariance(path: Path) -> dict:
    with fits.open(path, memmap=False) as hdul:
        table = next(hdu.data for hdu in hdul if hdu.name == "WAVECAL")
    names = np.char.strip(table["name"].astype(str))
    domain = np.asarray([
        float(table["coefficients"][names == "domain_start"][0]),
        float(table["coefficients"][names == "domain_end"][0]),
    ])
    peak = np.asarray(table["peaks"], dtype=float)
    wavelength_nm = np.asarray(table["wavelengths"], dtype=float)
    normalized = 2.0 * (peak - domain[0]) / (domain[1] - domain[0]) - 1.0
    design = np.polynomial.chebyshev.chebvander(normalized, 3)
    coefficients, _, _, _ = np.linalg.lstsq(design, wavelength_nm, rcond=None)
    residual = wavelength_nm - design @ coefficients
    dof = max(len(wavelength_nm) - len(coefficients), 1)
    covariance = np.linalg.inv(design.T @ design) * np.sum(np.square(residual)) / dof
    return {
        "arc": path.name,
        "domain_pixel": domain,
        "coefficients_nm": coefficients,
        "coefficient_covariance_nm2": covariance,
        "matched_lines": len(wavelength_nm),
        "reconstructed_rms_angstrom": float(np.sqrt(np.mean(np.square(residual))) * 10.0),
    }


def main() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    calibration = json.loads(CAL_REPORT.read_text(encoding="utf-8"))
    if not calibration["gates"]["arc_wavelength_solutions_passed"]:
        raise RuntimeError("Arc wavelength gate did not authorize LSF measurement")
    execution = config["stellar_kinematic_fit"]["lsf_execution"]
    rows = []
    wavecal_models = []
    for arc in calibration["arcs"]:
        path = ROOT / arc["product"]
        rows.extend(fit_arc(path, execution))
        wavecal_models.append(wavecal_covariance(path))
    ledger = pd.DataFrame(rows)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    ledger.to_csv(OUTPUT, index=False)

    edges = np.asarray(execution["observed_wavelength_bin_edges_angstrom"], dtype=float)
    summaries = []
    centers = 0.5 * (edges[:-1] + edges[1:])
    values = []
    errors = []
    for lower, upper, center in zip(edges[:-1], edges[1:], centers):
        use = (
            ledger["accepted"]
            & (ledger["reference_wavelength_angstrom"] >= lower)
            & (ledger["reference_wavelength_angstrom"] < upper)
        )
        widths = ledger.loc[use, "fwhm_angstrom"].to_numpy(dtype=float)
        value = float(np.median(widths)) if len(widths) else np.nan
        robust = float(1.4826 * np.median(np.abs(widths - value))) if len(widths) else np.nan
        error = robust / np.sqrt(len(widths)) if len(widths) else np.nan
        passed = len(widths) >= execution["minimum_accepted_lines_per_200_angstrom_bin"]
        summaries.append({
            "lower_angstrom": lower,
            "upper_angstrom": upper,
            "center_angstrom": center,
            "accepted_lines": len(widths),
            "median_fwhm_angstrom": value,
            "robust_standard_error_angstrom": error,
            "passed": passed,
        })
        values.append(value)
        errors.append(error)
    gate = all(row["passed"] and np.isfinite(row["median_fwhm_angstrom"]) for row in summaries)
    np.savez_compressed(
        MODEL,
        observed_wavelength_angstrom=centers,
        fwhm_angstrom=np.asarray(values),
        fwhm_standard_error_angstrom=np.asarray(errors),
        wavecal_arc_names=np.asarray([row["arc"] for row in wavecal_models]),
        wavecal_domain_pixel=np.asarray([row["domain_pixel"] for row in wavecal_models]),
        wavecal_coefficients_nm=np.asarray([row["coefficients_nm"] for row in wavecal_models]),
        wavecal_coefficient_covariance_nm2=np.asarray([row["coefficient_covariance_nm2"] for row in wavecal_models]),
    )
    report = {
        "report_version": "R1B1-A1689-GMOS-LSF-0.1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "selection_blind": True,
        "arc_count": len(calibration["arcs"]),
        "fitted_line_count": len(ledger),
        "accepted_line_count": int(ledger["accepted"].sum()),
        "wavelength_bins": summaries,
        "wavelength_solution_covariances": [
            {
                "arc": row["arc"],
                "matched_lines": row["matched_lines"],
                "reconstructed_rms_angstrom": row["reconstructed_rms_angstrom"],
                "coefficient_covariance_retained": True,
            }
            for row in wavecal_models
        ],
        "outputs": {
            "line_ledger": str(OUTPUT.relative_to(ROOT)).replace("\\", "/"),
            "lsf_model": str(MODEL.relative_to(ROOT)).replace("\\", "/"),
        },
        "gates": {
            "P3b_wavelength_dependent_lsf_gate_passed": bool(gate),
            "P3_profile_covariance_gate_passed": False,
            "gravity_response_fit_authorized": False,
        },
        "authorization": {
            "forward_convolve_xsl_and_run_baseline_ppxf": bool(gate),
            "infer_gravity_response": False,
            "fit_new_force_or_action": False,
        },
        "next_action": (
            "Forward-convolve XSL with this LSF and run the frozen baseline pPXF fits."
            if gate else
            "Retain the LSF shortfall and keep A1689 geometry-only; do not replace the measured LSF after seeing kinematics."
        ),
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
