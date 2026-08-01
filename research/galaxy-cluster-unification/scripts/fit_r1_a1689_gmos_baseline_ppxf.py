#!/usr/bin/env python3
"""Run the frozen baseline XSL pPXF fit on A1689's nine signed spectra."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from ppxf.ppxf import ppxf
import ppxf.ppxf_util as ppxf_util
from ppxf import sps_util


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/r1_a1689_gmos_reduction_covariance_protocol.json"
SPECTRA_REPORT = ROOT / "results/r1_a1689_gmos_signed_spectra/report.json"
LSF_REPORT = ROOT / "results/r1_a1689_gmos_lsf/report.json"
OUTPUT = ROOT / "data/derived/r1_a1689_gmos_baseline_ppxf_profile.csv"
FITS_OUTPUT = ROOT / "data/derived/r1_a1689_gmos_baseline_ppxf_fits.npz"
REPORT = ROOT / "results/r1_a1689_gmos_baseline_ppxf/report.json"


def fit_bin(spectrum: np.ndarray, variance: np.ndarray, retained: np.ndarray,
            wavelength: np.ndarray, config: dict, lsf: dict, sps=None) -> tuple[dict, object, dict]:
    fit_cfg = config["stellar_kinematic_fit"]
    execution = fit_cfg["ppxf_execution"]
    wave_range = fit_cfg["observed_fit_range_angstrom"]
    window = (wavelength >= wave_range[0]) & (wavelength <= wave_range[1])
    wave = wavelength[window]
    flux = spectrum[window].astype(float)
    var = variance[window].astype(float)
    mask = retained[window] & np.isfinite(flux) & np.isfinite(var) & (var > 0)
    if np.count_nonzero(mask) < 500:
        raise ValueError(f"only {np.count_nonzero(mask)} retained linear pixels")
    pixel = np.arange(len(flux))
    filled_flux = np.interp(pixel, pixel[mask], flux[mask])
    filled_variance = np.interp(pixel, pixel[mask], var[mask])
    redshift = fit_cfg["redshift_initial"]
    rest_range = wave[[0, -1]] / (1.0 + redshift)
    galaxy, log_wave, velocity_scale = ppxf_util.log_rebin(rest_range, filled_flux)
    variance_log, _, _ = ppxf_util.log_rebin(rest_range, filled_variance, velscale=velocity_scale)
    mask_log, _, _ = ppxf_util.log_rebin(rest_range, mask.astype(float), velscale=velocity_scale)
    common = min(len(galaxy), len(variance_log), len(mask_log), len(log_wave))
    galaxy, variance_log, mask_log, log_wave = (
        galaxy[:common], variance_log[:common], mask_log[:common], log_wave[:common]
    )
    log_retained = mask_log >= execution["input_mask_log_rebin_retention_threshold"]
    normalization = float(np.median(galaxy[log_retained]))
    if not np.isfinite(normalization) or normalization <= 0:
        raise ValueError("nonpositive pPXF normalization")
    galaxy /= normalization
    noise = np.sqrt(np.maximum(variance_log, 1e-30)) / normalization

    fwhm_rest = {
        "lam": lsf["observed_wavelength_angstrom"] / (1.0 + redshift),
        "fwhm": lsf["fwhm_angstrom"] / (1.0 + redshift),
    }
    if sps is None:
        sps = sps_util.sps_lib(
            ROOT / fit_cfg["template_path"], velocity_scale, fwhm_gal=fwhm_rest
        )
    stellar_templates = sps.templates.reshape(sps.templates.shape[0], -1)
    gas_templates, gas_names, gas_waves = ppxf_util.emission_lines(
        sps.ln_lam_temp, rest_range, fwhm_rest, tie_balmer=False, limit_doublets=False
    )
    wanted = np.array([
        name.startswith("Hdelta") or name.startswith("Hgamma") or name.startswith("[NeIII]")
        for name in gas_names
    ])
    gas_templates = gas_templates[:, wanted]
    gas_names = np.asarray(gas_names)[wanted].tolist()
    gas_waves = np.asarray(gas_waves)[wanted]
    if gas_templates.shape[1]:
        templates = np.column_stack([stellar_templates, gas_templates])
        component = np.r_[np.zeros(stellar_templates.shape[1], dtype=int), np.ones(gas_templates.shape[1], dtype=int)]
        gas_component = component > 0
        moments = [fit_cfg["moments"], 2]
        start = [
            [execution["stellar_start_velocity_km_s_relative_to_initial_redshift"], fit_cfg["initial_sigma_km_s"]],
            [execution["gas_start_velocity_km_s_relative_to_initial_redshift"], execution["gas_start_sigma_km_s"]],
        ]
    else:
        templates = stellar_templates
        component = None
        gas_component = None
        moments = fit_cfg["moments"]
        start = [execution["stellar_start_velocity_km_s_relative_to_initial_redshift"], fit_cfg["initial_sigma_km_s"]]
    good = ppxf_util.determine_goodpixels(
        log_wave, [sps.lam_temp.min(), sps.lam_temp.max()], 0
    )
    good = np.intersect1d(good, np.flatnonzero(log_retained))
    fit = ppxf(
        templates,
        galaxy,
        noise,
        velocity_scale,
        start,
        goodpixels=good,
        moments=moments,
        component=component,
        gas_component=gas_component,
        gas_names=gas_names if gas_names else None,
        lam=np.exp(log_wave),
        lam_temp=sps.lam_temp,
        degree=fit_cfg["additive_polynomial_degree"],
        mdegree=fit_cfg["multiplicative_polynomial_degree"],
        regul=execution["regularization"],
        quiet=True,
    )
    stellar_solution = fit.sol[0] if isinstance(fit.sol, list) else fit.sol
    stellar_error = fit.error[0] if isinstance(fit.error, list) else fit.error
    scaled_error = np.asarray(stellar_error) * np.sqrt(fit.chi2)
    result = {
        "velocity_km_s": float(stellar_solution[0]),
        "sigma_km_s": float(stellar_solution[1]),
        "velocity_formal_error_km_s": float(scaled_error[0]),
        "sigma_formal_error_km_s": float(scaled_error[1]),
        "reduced_chi2": float(fit.chi2),
        "linear_retained_pixels": int(np.count_nonzero(mask)),
        "log_retained_pixels": int(np.count_nonzero(log_retained)),
        "fitted_pixels": int(len(good)),
        "gas_templates": gas_names,
        "gas_rest_wavelengths_angstrom": gas_waves.tolist(),
    }
    arrays = {
        "log_wavelength": log_wave,
        "galaxy_normalized": galaxy,
        "noise_normalized": noise,
        "bestfit_normalized": fit.bestfit,
        "goodpixels": good,
    }
    return result, sps, arrays


def main() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    spectra_report = json.loads(SPECTRA_REPORT.read_text(encoding="utf-8"))
    lsf_report = json.loads(LSF_REPORT.read_text(encoding="utf-8"))
    if not spectra_report["authorization"]["run_frozen_baseline_ppxf_on_retained_signed_bins"]:
        raise RuntimeError("Signed-spectrum S/N gate did not authorize pPXF")
    if not lsf_report["authorization"]["forward_convolve_xsl_and_run_baseline_ppxf"]:
        raise RuntimeError("LSF gate did not authorize pPXF")
    product = np.load(ROOT / spectra_report["outputs"]["spectra"])
    lsf = dict(np.load(ROOT / lsf_report["outputs"]["lsf_model"]))
    wavelength = np.asarray(product["wavelength_angstrom"], dtype=float)
    rows = []
    fit_arrays = {}
    sps = None
    for index, snr_row in enumerate(spectra_report["signed_bins"], start=1):
        row = dict(snr_row)
        row.update({"status": "not_fit_snr_failure", "error": ""})
        if snr_row["snr_gate_passed"]:
            try:
                result, sps, arrays = fit_bin(
                    product["spectrum_electron"][index - 1],
                    product["variance_electron2"][index - 1],
                    product["retained_mask"][index - 1],
                    wavelength,
                    config,
                    lsf,
                    sps,
                )
                row.update(result)
                row["status"] = "success"
                for name, value in arrays.items():
                    fit_arrays[f"bin{index}_{name}"] = value
            except Exception as error:
                row.update({"status": "failed", "error": f"{type(error).__name__}: {error}"})
        rows.append(row)

    profile = pd.DataFrame(rows)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    profile.to_csv(OUTPUT, index=False)
    np.savez_compressed(FITS_OUTPUT, **fit_arrays)
    successful = profile["status"] == "success"
    pair_rows = []
    for negative, positive in [(1, 9), (2, 8), (3, 7), (4, 6)]:
        neg = profile.loc[profile["signed_bin"] == negative].iloc[0]
        pos = profile.loc[profile["signed_bin"] == positive].iloc[0]
        if neg["status"] == pos["status"] == "success":
            velocity_difference = abs(float(neg["velocity_km_s"] - pos["velocity_km_s"]))
            sigma_difference = abs(float(neg["sigma_km_s"] - pos["sigma_km_s"])) / np.mean([neg["sigma_km_s"], pos["sigma_km_s"]])
        else:
            velocity_difference = sigma_difference = np.nan
        pair_rows.append({
            "negative_signed_bin": negative,
            "positive_signed_bin": positive,
            "velocity_difference_km_s": velocity_difference,
            "sigma_difference_fraction": sigma_difference,
        })
    finite_pairs = [row for row in pair_rows if np.isfinite(row["velocity_difference_km_s"])]
    acceptance = config["profile_acceptance"]
    successful_count = int(successful.sum())
    formal_fractional = (
        profile.loc[successful, "sigma_formal_error_km_s"] / profile.loc[successful, "sigma_km_s"]
        if successful_count else pd.Series(dtype=float)
    )
    side_velocity_median = float(np.median([row["velocity_difference_km_s"] for row in finite_pairs])) if finite_pairs else np.inf
    side_velocity_max = float(np.max([row["velocity_difference_km_s"] for row in finite_pairs])) if finite_pairs else np.inf
    side_sigma_median = float(np.median([row["sigma_difference_fraction"] for row in finite_pairs])) if finite_pairs else np.inf
    side_sigma_max = float(np.max([row["sigma_difference_fraction"] for row in finite_pairs])) if finite_pairs else np.inf
    checks = {
        "minimum_successful_signed_bins": successful_count >= acceptance["minimum_finite_signed_bins"],
        "formal_fractional_sigma_uncertainty": bool(len(formal_fractional) and np.all(formal_fractional <= acceptance["maximum_fractional_sigma_uncertainty_each_retained_radial_bin"])),
        "median_opposite_side_velocity": side_velocity_median <= acceptance["maximum_median_opposite_side_velocity_difference_km_s"],
        "maximum_opposite_side_velocity": side_velocity_max <= acceptance["maximum_any_opposite_side_velocity_difference_km_s"],
        "median_opposite_side_sigma": side_sigma_median <= acceptance["maximum_median_opposite_side_sigma_difference_fraction"],
        "maximum_opposite_side_sigma": side_sigma_max <= acceptance["maximum_any_opposite_side_sigma_difference_fraction"],
    }
    baseline_internal = all(checks.values())
    report = {
        "report_version": "R1B1-A1689-GMOS-baseline-ppxf-0.1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "selection_blind": True,
        "successful_signed_bins": successful_count,
        "signed_bin_fits": rows,
        "opposite_side_pairs": pair_rows,
        "summary": {
            "median_opposite_side_velocity_difference_km_s": side_velocity_median,
            "maximum_opposite_side_velocity_difference_km_s": side_velocity_max,
            "median_opposite_side_sigma_difference_fraction": side_sigma_median,
            "maximum_opposite_side_sigma_difference_fraction": side_sigma_max,
        },
        "checks": checks,
        "outputs": {
            "profile": str(OUTPUT.relative_to(ROOT)).replace("\\", "/"),
            "fit_arrays": str(FITS_OUTPUT.relative_to(ROOT)).replace("\\", "/"),
        },
        "gates": {
            "P3c_baseline_ppxf_minimum_fits_gate_passed": checks["minimum_successful_signed_bins"],
            "P3c_baseline_internal_consistency_gate_passed": baseline_internal,
            "P3_profile_covariance_gate_passed": False,
            "gravity_response_fit_authorized": False,
        },
        "authorization": {
            "run_frozen_200_replicate_covariance_and_systematic_grid": checks["minimum_successful_signed_bins"],
            "infer_gravity_response": False,
            "fit_lens_mass_model": False,
            "fit_new_force_or_action": False,
        },
        "next_action": (
            "Run the frozen joint exposure/block bootstrap and systematic grid; retain the baseline side checks as final acceptance inputs."
            if checks["minimum_successful_signed_bins"] else
            "Keep A1689 geometry-only because fewer than seven baseline signed fits succeeded; do not change the masks or templates."
        ),
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
