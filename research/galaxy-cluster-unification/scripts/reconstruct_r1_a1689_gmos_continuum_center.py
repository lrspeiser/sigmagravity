#!/usr/bin/env python3
"""Fit the frozen continuum-only A1689 GMOS center before any sky or pPXF fit."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import astrodata
import gemini_instruments  # noqa: F401 - registers Gemini AstroData classes
import numpy as np
from scipy.optimize import least_squares


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/r1_a1689_gmos_reduction_covariance_protocol.json"
INPUT_REPORT = ROOT / "results/r1_a1689_gmos_science_cal2d/report.json"
OUTPUT = ROOT / "data/derived/r1_a1689_gmos_reconstruction/continuum_center_profiles.npz"
REPORT = ROOT / "results/r1_a1689_gmos_continuum_center/report.json"


def moffat(offset: np.ndarray, center: float, alpha: float, beta: float,
           amplitude: float, background: float) -> np.ndarray:
    return background + amplitude * (1.0 + np.square((offset - center) / alpha)) ** (-beta)


def covariance(result, ndata: int) -> np.ndarray:
    dof = max(ndata - result.x.size, 1)
    scale = float(np.sum(np.square(result.fun)) / dof)
    return np.linalg.pinv(result.jac.T @ result.jac) * scale


def spatial_offset_arcsec(ad, config: dict) -> np.ndarray:
    ext = ad[0]
    ny, nx = ext.data.shape
    y = np.arange(ny, dtype=float)
    _, ra, dec = ext.wcs(np.full(ny, 0.5 * (nx - 1)), y)
    zero = config["spatial_extraction"]["photometric_coordinate_zero"]
    dra = (np.asarray(ra) - zero["ra_deg"]) * np.cos(np.deg2rad(zero["dec_deg"])) * 3600.0
    ddec = (np.asarray(dec) - zero["dec_deg"]) * 3600.0
    pa = np.deg2rad(float(ad.position_angle()))
    return dra * np.sin(pa) + ddec * np.cos(pa)


def continuum_profile(ad, config: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ext = ad[0]
    data = np.asarray(ext.data, dtype=float)
    variance = np.asarray(ext.variance, dtype=float)
    dq = np.asarray(ext.mask, dtype=np.uint16)
    ny, nx = data.shape
    wavelength_nm, _, _ = ext.wcs(np.arange(nx, dtype=float), np.full(nx, 0.5 * (ny - 1)))
    window = config["spatial_extraction"]["center_estimator"]
    if "4800-5400" not in window:
        raise RuntimeError("Continuum window is not the frozen 4800-5400 Angstrom interval")
    use_wave = (np.asarray(wavelength_nm) >= 480.0) & (np.asarray(wavelength_nm) <= 540.0)
    good = (
        (dq[:, use_wave] == 0)
        & np.isfinite(data[:, use_wave])
        & np.isfinite(variance[:, use_wave])
        & (variance[:, use_wave] > 0)
    )
    weight = np.divide(
        1.0,
        variance[:, use_wave],
        out=np.zeros_like(variance[:, use_wave], dtype=float),
        where=good,
    )
    weight_sum = weight.sum(axis=1)
    profile = np.divide(
        np.where(good, data[:, use_wave] * weight, 0.0).sum(axis=1),
        weight_sum,
        out=np.full(ny, np.nan),
        where=weight_sum > 0,
    )
    error = np.divide(
        1.0, np.sqrt(weight_sum), out=np.full(ny, np.inf), where=weight_sum > 0
    )
    return spatial_offset_arcsec(ad, config), profile, error


def fit_individual(offset: np.ndarray, profile: np.ndarray, error: np.ndarray,
                   execution: dict) -> tuple[np.ndarray, np.ndarray, int]:
    support = execution["fit_support_arcsec_from_photometric_coordinate_zero"]
    use = (
        (offset >= support[0]) & (offset <= support[1])
        & np.isfinite(profile) & np.isfinite(error) & (error > 0)
    )
    x = offset[use]
    y = profile[use]
    e = error[use]
    background = float(np.median(y))
    amplitude = max(float(np.max(y) - background), np.finfo(float).eps)
    lower = [execution["center_bound_arcsec"][0], execution["alpha_bound_arcsec"][0],
             execution["beta_bound"][0], 0.0, -np.inf]
    upper = [execution["center_bound_arcsec"][1], execution["alpha_bound_arcsec"][1],
             execution["beta_bound"][1], np.inf, np.inf]
    result = least_squares(
        lambda p: (moffat(x, *p) - y) / e,
        x0=[0.0, 0.8, 2.5, amplitude, background],
        bounds=(lower, upper),
        method="trf",
    )
    if not result.success:
        raise RuntimeError(result.message)
    return result.x, covariance(result, len(x)), len(x)


def fit_joint(profiles: list[dict], execution: dict) -> tuple[np.ndarray, np.ndarray, int]:
    selected = []
    x0_tail = []
    support = execution["fit_support_arcsec_from_photometric_coordinate_zero"]
    for item in profiles:
        use = (
            (item["offset"] >= support[0]) & (item["offset"] <= support[1])
            & np.isfinite(item["profile"]) & np.isfinite(item["error"]) & (item["error"] > 0)
        )
        selected.append((item["offset"][use], item["profile"][use], item["error"][use]))
        x0_tail.extend([item["individual_parameters"][3], item["individual_parameters"][4]])

    def residual(parameters: np.ndarray) -> np.ndarray:
        center, alpha, beta = parameters[:3]
        values = []
        for index, (x, y, e) in enumerate(selected):
            amp, bg = parameters[3 + 2 * index:5 + 2 * index]
            values.append((moffat(x, center, alpha, beta, amp, bg) - y) / e)
        return np.concatenate(values)

    lower = [execution["center_bound_arcsec"][0], execution["alpha_bound_arcsec"][0],
             execution["beta_bound"][0]] + [value for _ in profiles for value in (0.0, -np.inf)]
    upper = [execution["center_bound_arcsec"][1], execution["alpha_bound_arcsec"][1],
             execution["beta_bound"][1]] + [value for _ in profiles for value in (np.inf, np.inf)]
    result = least_squares(
        residual,
        x0=[0.0, 0.8, 2.5] + x0_tail,
        bounds=(lower, upper),
        method="trf",
    )
    if not result.success:
        raise RuntimeError(result.message)
    ndata = sum(len(item[0]) for item in selected)
    return result.x, covariance(result, ndata), ndata


def main() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    input_report = json.loads(INPUT_REPORT.read_text(encoding="utf-8"))
    if not input_report["gates"]["P2b_individual_calibrated_2d_gate_passed"]:
        raise RuntimeError("P2b did not authorize the continuum-only center fit")
    execution = config["spatial_extraction"]["continuum_center_fit"]
    profiles = []
    arrays = {}
    for item in input_report["products"]:
        path = ROOT / item["product"]
        ad = astrodata.open(path)
        offset, profile, error = continuum_profile(ad, config)
        parameters, cov, npoints = fit_individual(offset, profile, error, execution)
        profiles.append({
            "science": item["science"],
            "offset": offset,
            "profile": profile,
            "error": error,
            "individual_parameters": parameters,
            "individual_covariance": cov,
            "npoints": npoints,
        })
        stem = Path(item["science"]).stem
        arrays[f"{stem}_offset_arcsec"] = offset
        arrays[f"{stem}_continuum_electron"] = profile
        arrays[f"{stem}_continuum_error_electron"] = error

    joint, joint_cov, joint_npoints = fit_joint(profiles, execution)
    centers = np.asarray([item["individual_parameters"][0] for item in profiles])
    center_range = float(np.ptp(centers))
    max_range = config["calibration_acceptance"]["maximum_continuum_centroid_range_between_exposures_arcsec"]
    gate = bool(center_range <= max_range)
    arrays["joint_parameters_center_alpha_beta_then_amplitude_background_per_exposure"] = joint
    arrays["joint_covariance"] = joint_cov
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUTPUT, **arrays)

    fwhm = 2.0 * joint[1] * np.sqrt(np.power(2.0, 1.0 / joint[2]) - 1.0)
    report = {
        "report_version": "R1B1-A1689-GMOS-continuum-center-0.1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "selection_blind": True,
        "continuum_window_angstrom": [4800.0, 5400.0],
        "photometric_coordinate_zero": config["spatial_extraction"]["photometric_coordinate_zero"],
        "individual_fits": [
            {
                "science": item["science"],
                "center_arcsec": float(item["individual_parameters"][0]),
                "center_standard_error_arcsec": float(np.sqrt(max(item["individual_covariance"][0, 0], 0.0))),
                "alpha_arcsec": float(item["individual_parameters"][1]),
                "beta": float(item["individual_parameters"][2]),
                "fitted_spatial_rows": item["npoints"],
            }
            for item in profiles
        ],
        "joint_fit": {
            "center_arcsec": float(joint[0]),
            "center_standard_error_arcsec": float(np.sqrt(max(joint_cov[0, 0], 0.0))),
            "alpha_arcsec": float(joint[1]),
            "beta": float(joint[2]),
            "fwhm_arcsec": float(fwhm),
            "fitted_spatial_rows_across_exposures": joint_npoints,
        },
        "individual_center_range_arcsec": center_range,
        "maximum_allowed_center_range_arcsec": max_range,
        "output": str(OUTPUT.relative_to(ROOT)).replace("\\", "/"),
        "gates": {
            "P2b_individual_calibrated_2d_gate_passed": True,
            "P2c_continuum_centroid_range_gate_passed": gate,
            "P2_calibrated_2d_sky_centroid_coverage_gate_passed": False,
            "P3_profile_covariance_gate_passed": False,
            "gravity_response_fit_authorized": False,
        },
        "authorization": {
            "execute_frozen_sky_window_variants": gate,
            "fit_stellar_kinematics": False,
            "infer_gravity_response": False,
            "fit_new_force_or_action": False,
        },
        "next_action": (
            "Apply and audit all three preregistered sky-window variants before any pPXF call."
            if gate else
            "Retain the failed centroid-range gate and keep A1689 geometry-only; do not register away the discrepancy."
        ),
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
