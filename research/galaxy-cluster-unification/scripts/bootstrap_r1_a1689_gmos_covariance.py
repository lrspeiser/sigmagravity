#!/usr/bin/env python3
"""Run the frozen 200-replicate joint A1689 exposure/wild bootstrap."""

from __future__ import annotations

import contextlib
import io
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import medfilt

from fit_r1_a1689_gmos_baseline_ppxf import fit_bin


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/r1_a1689_gmos_reduction_covariance_protocol.json"
BASELINE_REPORT = ROOT / "results/r1_a1689_gmos_baseline_ppxf/report.json"
COMBINATION_REPORT = ROOT / "results/r1_a1689_gmos_combination/report.json"
CENTER_DATA = ROOT / "data/derived/r1_a1689_gmos_reconstruction/continuum_center_profiles.npz"
LSF_REPORT = ROOT / "results/r1_a1689_gmos_lsf/report.json"
OUTPUT = ROOT / "data/derived/r1_a1689_gmos_bootstrap.csv"
COVARIANCE = ROOT / "data/derived/r1_a1689_gmos_bootstrap_covariance.npz"
REPORT = ROOT / "results/r1_a1689_gmos_bootstrap/report.json"
NO_DATA_BIT = np.uint16(16)


def linear_resample_x(data: np.ndarray, variance: np.ndarray, dq: np.ndarray,
                      coordinate: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    finite_coordinate = np.isfinite(coordinate)
    safe_coordinate = np.where(finite_coordinate, coordinate, 0.0)
    x0 = np.floor(safe_coordinate).astype(int)
    fraction = safe_coordinate - x0
    inside = finite_coordinate & (x0 >= 0) & (x0 + 1 < data.shape[1])
    safe = np.clip(x0, 0, data.shape[1] - 2)
    left = data[:, safe]
    right = data[:, safe + 1]
    out = (1.0 - fraction)[None, :] * left + fraction[None, :] * right
    out_var = np.square(1.0 - fraction)[None, :] * variance[:, safe] + np.square(fraction)[None, :] * variance[:, safe + 1]
    nearest = np.clip(np.rint(safe_coordinate).astype(int), 0, data.shape[1] - 1)
    out_dq = dq[:, nearest].copy()
    invalid = ~inside[None, :] | ~np.isfinite(out) | ~np.isfinite(out_var)
    out[invalid] = np.nan
    out_var[invalid] = np.nan
    out_dq[invalid] |= NO_DATA_BIT
    return out, out_var, out_dq


def linear_resample_y(data: np.ndarray, variance: np.ndarray, dq: np.ndarray,
                      coordinate: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    finite_coordinate = np.isfinite(coordinate)
    safe_coordinate = np.where(finite_coordinate, coordinate, 0.0)
    y0 = np.floor(safe_coordinate).astype(int)
    fraction = safe_coordinate - y0
    inside = finite_coordinate & (y0 >= 0) & (y0 + 1 < data.shape[0])
    safe = np.clip(y0, 0, data.shape[0] - 2)
    out = (1.0 - fraction)[:, None] * data[safe] + fraction[:, None] * data[safe + 1]
    out_var = np.square(1.0 - fraction)[:, None] * variance[safe] + np.square(fraction)[:, None] * variance[safe + 1]
    nearest = np.clip(np.rint(safe_coordinate).astype(int), 0, data.shape[0] - 1)
    out_dq = dq[nearest].copy()
    invalid = ~inside[:, None] | ~np.isfinite(out) | ~np.isfinite(out_var)
    out[invalid] = np.nan
    out_var[invalid] = np.nan
    out_dq[invalid] |= NO_DATA_BIT
    return out, out_var, out_dq


def wavelength_delta(wavelength: np.ndarray, arc_index: int, lsf: dict,
                     drawn_coefficients: np.ndarray) -> np.ndarray:
    domain = lsf["wavecal_domain_pixel"][arc_index]
    baseline = lsf["wavecal_coefficients_nm"][arc_index]
    pixels = np.linspace(domain[0], domain[1], 4096)
    normalized = 2.0 * (pixels - domain[0]) / (domain[1] - domain[0]) - 1.0
    model_nm = np.polynomial.chebyshev.chebval(normalized, baseline)
    target_pixel = np.interp(wavelength / 10.0, model_nm[::-1], pixels[::-1])
    target_normalized = 2.0 * (target_pixel - domain[0]) / (domain[1] - domain[0]) - 1.0
    design = np.polynomial.chebyshev.chebvander(target_normalized, 3)
    return 10.0 * (design @ (drawn_coefficients - baseline))


def recombine(draw: np.ndarray, arrays: dict, wavelength: np.ndarray, offset: np.ndarray,
              lsf: dict, coefficient_draws: np.ndarray, arc_for_exposure: list[int],
              center_delta: float, minimum_coverage: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    selected_data = []
    selected_var = []
    selected_dq = []
    for exposure in draw:
        delta = wavelength_delta(wavelength, arc_for_exposure[exposure], lsf, coefficient_draws[arc_for_exposure[exposure]])
        source_wave = wavelength - delta
        coordinate = np.interp(source_wave, wavelength, np.arange(len(wavelength)), left=np.nan, right=np.nan)
        source_dq = arrays["exposure_dq"][exposure].copy()
        source_dq[~arrays["exposure_accepted_after_clipping"][exposure]] |= NO_DATA_BIT
        data, var, dq = linear_resample_x(
            arrays["exposure_science_electron"][exposure],
            arrays["exposure_variance_electron2"][exposure],
            source_dq,
            coordinate,
        )
        selected_data.append(data)
        selected_var.append(var)
        selected_dq.append(dq)
    data = np.asarray(selected_data)
    var = np.asarray(selected_var)
    dq = np.asarray(selected_dq)
    valid = (dq == 0) & np.isfinite(data) & np.isfinite(var) & (var > 0)
    weight = np.divide(1.0, var, out=np.zeros_like(var), where=valid)
    weight_sum = weight.sum(axis=0)
    coverage = valid.sum(axis=0)
    good = coverage >= minimum_coverage
    combined = np.divide(
        (weight * np.where(valid, data, 0.0)).sum(axis=0), weight_sum,
        out=np.full(weight_sum.shape, np.nan), where=good,
    )
    combined_var = np.divide(1.0, weight_sum, out=np.full(weight_sum.shape, np.nan), where=good)
    combined_dq = np.where(good, 0, NO_DATA_BIT).astype(np.uint16)
    ycoord = np.interp(offset + center_delta, offset, np.arange(len(offset)), left=np.nan, right=np.nan)
    return linear_resample_y(combined, combined_var, combined_dq, ycoord)


def extract_signed(data: np.ndarray, variance: np.ndarray, dq: np.ndarray,
                   wavelength: np.ndarray, offset: np.ndarray, config: dict) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    edges = config["spatial_extraction"]["signed_bin_edges_arcsec"]
    min_fraction = config["spatial_extraction"]["minimum_valid_pixel_fraction_per_bin"]
    results = []
    for index, (lower, upper) in enumerate(zip(edges[:-1], edges[1:]), start=1):
        spatial = (offset >= lower) & ((offset < upper) if index < len(edges) - 1 else (offset <= upper))
        values, errors, flags = data[spatial], variance[spatial], dq[spatial]
        valid = np.isfinite(values) & np.isfinite(errors) & (errors > 0) & (flags == 0)
        count = valid.sum(axis=0)
        retained = count >= min_fraction * np.count_nonzero(spatial)
        spectrum = np.where(valid, values, 0.0).sum(axis=0)
        summed_var = np.where(valid, errors, 0.0).sum(axis=0)
        noise = np.sqrt(np.maximum(summed_var, 0.0))
        local = medfilt(noise, kernel_size=21)
        ratio = np.divide(noise, local, out=np.full_like(noise, np.inf), where=local > 0)
        finite = retained & np.isfinite(ratio)
        center = np.median(ratio[finite])
        robust = 1.4826 * np.median(np.abs(ratio[finite] - center))
        retained &= ratio <= center + 6.0 * max(robust, 0.02)
        retained &= ~((wavelength >= 5569.0) & (wavelength <= 5585.0))
        results.append((spectrum, summed_var, retained))
    return results


def baseline_residuals(wavelength: np.ndarray, baseline_report: dict) -> np.ndarray:
    fits = np.load(ROOT / baseline_report["outputs"]["fit_arrays"])
    redshift = json.loads(CONFIG.read_text())["stellar_kinematic_fit"]["redshift_initial"]
    rest = wavelength / (1.0 + redshift)
    rows = []
    for index in range(1, 10):
        log_wave = fits[f"bin{index}_log_wavelength"]
        residual = fits[f"bin{index}_galaxy_normalized"] - fits[f"bin{index}_bestfit_normalized"]
        rows.append(np.interp(rest, np.exp(log_wave), residual, left=0.0, right=0.0))
    return np.asarray(rows)


def main() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    baseline = json.loads(BASELINE_REPORT.read_text(encoding="utf-8"))
    combination = json.loads(COMBINATION_REPORT.read_text(encoding="utf-8"))
    lsf_report = json.loads(LSF_REPORT.read_text(encoding="utf-8"))
    if not baseline["authorization"]["run_frozen_200_replicate_covariance_and_systematic_grid"]:
        raise RuntimeError("Baseline fits did not authorize covariance")
    arrays_file = np.load(ROOT / combination["output"])
    arrays = {name: arrays_file[name] for name in arrays_file.files}
    wavelength = np.asarray(arrays["wavelength_angstrom"], dtype=float)
    offset = np.asarray(arrays["signed_offset_arcsec"], dtype=float)
    lsf_file = np.load(ROOT / lsf_report["outputs"]["lsf_model"])
    lsf = {name: lsf_file[name] for name in lsf_file.files}
    center_cov = np.load(CENTER_DATA)["joint_covariance"]
    center_sigma = float(np.sqrt(max(center_cov[0, 0], 0.0)))
    residual = baseline_residuals(wavelength, baseline)
    bootstrap = config["covariance_protocol"]
    rng = np.random.default_rng(bootstrap["random_seed"])
    arc_names = lsf["wavecal_arc_names"].astype(str).tolist()
    arc_for_exposure = [
        arc_names.index("N20090615S0080_arc.fits"),
        arc_names.index("N20090621S0037_arc.fits"),
        arc_names.index("N20090621S0037_arc.fits"),
        arc_names.index("N20090621S0040_arc.fits"),
    ]
    rows = []
    for replicate in range(1, bootstrap["replicates"] + 1):
        draw = rng.integers(0, 4, size=4)
        center_delta = rng.normal(0.0, center_sigma)
        coefficient_draws = np.asarray([
            rng.multivariate_normal(mean, covariance)
            for mean, covariance in zip(lsf["wavecal_coefficients_nm"], lsf["wavecal_coefficient_covariance_nm2"])
        ])
        fwhm = rng.normal(lsf["fwhm_angstrom"], lsf["fwhm_standard_error_angstrom"])
        replicate_lsf = {
            "observed_wavelength_angstrom": lsf["observed_wavelength_angstrom"],
            "fwhm_angstrom": np.maximum(fwhm, 0.5),
        }
        data, var, dq = recombine(
            draw, arrays, wavelength, offset, lsf, coefficient_draws, arc_for_exposure,
            center_delta, config["calibration_acceptance"]["minimum_unmasked_exposure_coverage_per_combined_pixel"],
        )
        spectra = extract_signed(data, var, dq, wavelength, offset, config)
        signs = np.repeat(rng.choice([-1.0, 1.0], size=int(np.ceil(len(wavelength) / 8))), 8)[:len(wavelength)]
        sps = None
        for bin_index, (spectrum, summed_var, retained) in enumerate(spectra, start=1):
            row = {
                "replicate": replicate,
                "signed_bin": bin_index,
                "draw": " ".join(str(int(value)) for value in draw),
                "center_delta_arcsec": center_delta,
                "status": "success",
                "error": "",
            }
            try:
                fit_window = (wavelength >= 4600.0) & (wavelength <= 5550.0) & retained
                normalization = float(np.median(spectrum[fit_window]))
                wild_spectrum = spectrum + signs * residual[bin_index - 1] * normalization
                with contextlib.redirect_stdout(io.StringIO()):
                    result, sps, _ = fit_bin(
                        wild_spectrum, summed_var, retained, wavelength, config, replicate_lsf, sps
                    )
                row.update(result)
            except Exception as error:
                row.update({"status": "failed", "error": f"{type(error).__name__}: {error}"})
            rows.append(row)
        if replicate % 10 == 0:
            print(json.dumps({"completed_replicates": replicate}), flush=True)

    frame = pd.DataFrame(rows)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUTPUT, index=False)
    matrix = frame.pivot(index="replicate", columns="signed_bin", values="sigma_km_s")
    complete = matrix.dropna()
    covariance = np.cov(complete.to_numpy(), rowvar=False, ddof=1) if len(complete) >= 2 else np.full((9, 9), np.nan)
    velocity_matrix = frame.pivot(index="replicate", columns="signed_bin", values="velocity_km_s").loc[complete.index]
    velocity_covariance = np.cov(velocity_matrix.to_numpy(), rowvar=False, ddof=1) if len(complete) >= 2 else np.full((9, 9), np.nan)
    np.savez_compressed(
        COVARIANCE,
        complete_replicates=complete.index.to_numpy(dtype=int),
        sigma_bootstrap=complete.to_numpy(),
        velocity_bootstrap=velocity_matrix.to_numpy(),
        sigma_covariance=covariance,
        velocity_covariance=velocity_covariance,
    )
    required = bootstrap["minimum_successful_replicates"]
    gate = len(complete) >= required and np.isfinite(covariance).all()
    report = {
        "report_version": "R1B1-A1689-GMOS-bootstrap-0.1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "selection_blind": True,
        "requested_replicates": bootstrap["replicates"],
        "complete_nine_bin_replicates": len(complete),
        "minimum_required_complete_replicates": required,
        "failed_bin_fits": int((frame["status"] != "success").sum()),
        "center_sigma_arcsec": center_sigma,
        "outputs": {
            "ledger": str(OUTPUT.relative_to(ROOT)).replace("\\", "/"),
            "covariance": str(COVARIANCE.relative_to(ROOT)).replace("\\", "/"),
        },
        "gates": {
            "P3d_bootstrap_covariance_gate_passed": bool(gate),
            "P3_profile_covariance_gate_passed": False,
            "gravity_response_fit_authorized": False,
        },
        "authorization": {
            "run_frozen_systematic_grid_and_final_covariance_assembly": bool(gate),
            "infer_gravity_response": False,
            "fit_new_force_or_action": False,
        },
        "next_action": (
            "Run the frozen polynomial/range/sky systematic grid and assemble signed plus symmetrized covariance."
            if gate else
            "Keep A1689 geometry-only because fewer than 180 complete covariance replicates survived; do not change the bootstrap."
        ),
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
