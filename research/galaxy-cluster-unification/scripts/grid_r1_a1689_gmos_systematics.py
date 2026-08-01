#!/usr/bin/env python3
"""Run the frozen A1689 polynomial/range/sky pPXF systematic grid."""

from __future__ import annotations

import contextlib
import copy
import io
import itertools
import json
from datetime import datetime, timezone
from pathlib import Path

import astrodata
import gemini_instruments  # noqa: F401 - registers Gemini AstroData classes
import numpy as np
import pandas as pd

from bootstrap_r1_a1689_gmos_covariance import extract_signed
from combine_r1_a1689_gmos_baseline import bilinear_resample, input_axes, pixel_coordinate
from fit_r1_a1689_gmos_baseline_ppxf import fit_bin


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/r1_a1689_gmos_reduction_covariance_protocol.json"
CAL2D_REPORT = ROOT / "results/r1_a1689_gmos_science_cal2d/report.json"
SKY_REPORT = ROOT / "results/r1_a1689_gmos_sky_models/report.json"
CENTER_REPORT = ROOT / "results/r1_a1689_gmos_continuum_center/report.json"
CENTER_DATA = ROOT / "data/derived/r1_a1689_gmos_reconstruction/continuum_center_profiles.npz"
COMBINATION_REPORT = ROOT / "results/r1_a1689_gmos_combination/report.json"
BASELINE_REPORT = ROOT / "results/r1_a1689_gmos_baseline_ppxf/report.json"
BOOTSTRAP_REPORT = ROOT / "results/r1_a1689_gmos_bootstrap/report.json"
LSF_REPORT = ROOT / "results/r1_a1689_gmos_lsf/report.json"
OUTPUT = ROOT / "data/derived/r1_a1689_gmos_systematic_grid.csv"
REPORT = ROOT / "results/r1_a1689_gmos_systematics/report.json"
NO_DATA_BIT = np.uint16(16)


def variant_spectra(name: str, config: dict, cal2d: dict, sky: dict,
                    center: dict, center_arrays, combination: dict) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    combined_product = np.load(ROOT / combination["output"])
    wavelength = np.asarray(combined_product["wavelength_angstrom"], dtype=float)
    signed_offset = np.asarray(combined_product["signed_offset_arcsec"], dtype=float)
    centers = {item["science"]: item["center_arcsec"] for item in center["individual_fits"]}
    sky_by_science = {item["science"]: item for item in sky["products"]}
    cal_by_science = {item["science"]: item for item in cal2d["products"]}
    joint_center = center["joint_fit"]["center_arcsec"]
    data_rows, var_rows, dq_rows = [], [], []
    for science in [item["science"] for item in cal2d["products"]]:
        ad = astrodata.open(ROOT / cal_by_science[science]["product"])
        ext = ad[0]
        data = np.asarray(ext.data, dtype=float)
        variance = np.asarray(ext.variance, dtype=float)
        dq = np.asarray(ext.mask, dtype=np.uint16).copy()
        stem = Path(science).stem
        absolute_offset = np.asarray(center_arrays[f"{stem}_offset_arcsec"], dtype=float)
        model = np.load(ROOT / sky_by_science[science]["model_output"])
        coeff = model[f"{name}_coefficients_intercept_slope"]
        cov = model[f"{name}_coefficient_covariance"]
        sky_offset = absolute_offset - joint_center
        finite = np.isfinite(coeff[:, 0])
        sky_value = coeff[None, :, 0] + sky_offset[:, None] * coeff[None, :, 1]
        sky_variance = (
            cov[None, :, 0, 0]
            + 2 * sky_offset[:, None] * cov[None, :, 0, 1]
            + np.square(sky_offset[:, None]) * cov[None, :, 1, 1]
        )
        data[:, finite] -= sky_value[:, finite]
        variance[:, finite] += sky_variance[:, finite]
        dq[:, ~finite] |= NO_DATA_BIT
        input_wave, input_offset = input_axes(ad, absolute_offset)
        xcoord = pixel_coordinate(input_wave, wavelength)
        ycoord = pixel_coordinate(input_offset, signed_offset + centers[science])
        out = bilinear_resample(data, variance, dq, ycoord, xcoord)
        data_rows.append(out[0])
        var_rows.append(out[1])
        dq_rows.append(out[2])
    data = np.asarray(data_rows)
    variance = np.asarray(var_rows)
    dq = np.asarray(dq_rows)
    accepted = (dq == 0) & np.isfinite(data) & np.isfinite(variance) & (variance > 0)
    for _ in range(config["spatial_extraction"]["combination_execution"]["combination_clipping_iterations"]):
        weight = np.divide(1.0, variance, out=np.zeros_like(variance), where=accepted)
        total = weight.sum(axis=0)
        mean = np.divide((weight * np.where(accepted, data, 0)).sum(axis=0), total,
                         out=np.full(total.shape, np.nan), where=total > 0)
        standardized = np.divide(np.abs(data - mean), np.sqrt(variance),
                                 out=np.full_like(variance, np.inf), where=variance > 0)
        accepted &= standardized <= 4.0
    coverage = accepted.sum(axis=0)
    good = coverage >= config["calibration_acceptance"]["minimum_unmasked_exposure_coverage_per_combined_pixel"]
    weight = np.divide(1.0, variance, out=np.zeros_like(variance), where=accepted)
    total = weight.sum(axis=0)
    combined = np.divide((weight * np.where(accepted, data, 0)).sum(axis=0), total,
                         out=np.full(total.shape, np.nan), where=good)
    combined_variance = np.divide(1.0, total, out=np.full(total.shape, np.nan), where=good)
    combined_dq = np.where(good, 0, NO_DATA_BIT).astype(np.uint16)
    return extract_signed(combined, combined_variance, combined_dq, wavelength, signed_offset, config)


def main() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    bootstrap = json.loads(BOOTSTRAP_REPORT.read_text(encoding="utf-8"))
    if not bootstrap["authorization"]["run_frozen_systematic_grid_and_final_covariance_assembly"]:
        raise RuntimeError("Bootstrap gate did not authorize the systematic grid")
    cal2d = json.loads(CAL2D_REPORT.read_text(encoding="utf-8"))
    sky = json.loads(SKY_REPORT.read_text(encoding="utf-8"))
    center = json.loads(CENTER_REPORT.read_text(encoding="utf-8"))
    combination = json.loads(COMBINATION_REPORT.read_text(encoding="utf-8"))
    baseline = json.loads(BASELINE_REPORT.read_text(encoding="utf-8"))
    lsf_report = json.loads(LSF_REPORT.read_text(encoding="utf-8"))
    lsf_file = np.load(ROOT / lsf_report["outputs"]["lsf_model"])
    lsf = {"observed_wavelength_angstrom": lsf_file["observed_wavelength_angstrom"],
           "fwhm_angstrom": lsf_file["fwhm_angstrom"]}
    center_arrays = np.load(CENTER_DATA)
    wavelength = np.load(ROOT / combination["output"])["wavelength_angstrom"]
    spectra_by_sky = {
        name: variant_spectra(name, config, cal2d, sky, center, center_arrays, combination)
        for name in ("inner", "baseline", "outer")
    }
    grid = config["covariance_protocol"]["pre_registered_systematic_grid"]
    baseline_sigma = {
        item["signed_bin"]: item["sigma_km_s"] for item in baseline["signed_bin_fits"]
    }
    rows = []
    combinations = list(itertools.product(
        grid["additive_polynomial_degree"], grid["observed_fit_range_angstrom"], grid["sky_window_variant"]
    ))
    for run_index, (degree, fit_range, sky_name) in enumerate(combinations, start=1):
        run_config = copy.deepcopy(config)
        run_config["stellar_kinematic_fit"]["additive_polynomial_degree"] = degree
        run_config["stellar_kinematic_fit"]["observed_fit_range_angstrom"] = fit_range
        sps = None
        for bin_index, (spectrum, variance, retained) in enumerate(spectra_by_sky[sky_name], start=1):
            row = {
                "run": run_index,
                "degree": degree,
                "fit_start_angstrom": fit_range[0],
                "fit_stop_angstrom": fit_range[1],
                "sky_variant": sky_name,
                "signed_bin": bin_index,
                "status": "success",
                "error": "",
            }
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    result, sps, _ = fit_bin(
                        spectrum, variance, retained, wavelength, run_config, lsf, sps
                    )
                row.update(result)
                row["sigma_shift_fraction_from_frozen_baseline"] = (
                    result["sigma_km_s"] - baseline_sigma[bin_index]
                ) / baseline_sigma[bin_index]
            except Exception as error:
                row.update({"status": "failed", "error": f"{type(error).__name__}: {error}",
                            "sigma_shift_fraction_from_frozen_baseline": np.nan})
            rows.append(row)
        print(json.dumps({"completed_systematic_runs": run_index, "total": len(combinations)}), flush=True)

    frame = pd.DataFrame(rows)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUTPUT, index=False)
    complete = frame.groupby("run")["status"].apply(lambda values: bool((values == "success").all()))
    successful = frame[frame["status"] == "success"]
    max_shift_by_bin = successful.groupby("signed_bin")["sigma_shift_fraction_from_frozen_baseline"].apply(
        lambda values: float(np.max(np.abs(values)))
    )
    threshold = config["profile_acceptance"]["maximum_sigma_shift_fraction_over_systematic_grid"]
    all_complete = int(complete.sum()) == len(combinations)
    shift_gate = len(max_shift_by_bin) == 9 and bool((max_shift_by_bin <= threshold).all())
    baseline_row = frame[
        (frame["degree"] == 8) & (frame["fit_start_angstrom"] == 4600.0)
        & (frame["fit_stop_angstrom"] == 5550.0) & (frame["sky_variant"] == "baseline")
    ]
    reproduction = float(np.max(np.abs(baseline_row["sigma_shift_fraction_from_frozen_baseline"])))
    gate = all_complete and shift_gate and reproduction <= 1e-6
    report = {
        "report_version": "R1B1-A1689-GMOS-systematics-0.1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "selection_blind": True,
        "requested_grid_runs": len(combinations),
        "complete_grid_runs": int(complete.sum()),
        "failed_bin_fits": int((frame["status"] != "success").sum()),
        "baseline_reproduction_max_fractional_sigma_difference": reproduction,
        "maximum_absolute_sigma_shift_fraction_by_signed_bin": {
            str(int(index)): float(value) for index, value in max_shift_by_bin.items()
        },
        "maximum_allowed_shift_fraction": threshold,
        "output": str(OUTPUT.relative_to(ROOT)).replace("\\", "/"),
        "gates": {
            "all_systematic_grid_runs_complete": all_complete,
            "baseline_reproduction_passed": reproduction <= 1e-6,
            "P3e_systematic_shift_gate_passed": bool(gate),
            "P3_profile_covariance_gate_passed": False,
            "gravity_response_fit_authorized": False,
        },
        "authorization": {
            "assemble_final_signed_and_symmetrized_covariance": bool(gate),
            "infer_gravity_response": False,
            "fit_new_force_or_action": False,
        },
        "next_action": (
            "Assemble bootstrap plus systematic covariance, transform to five radial bins, and apply all final P3 gates."
            if gate else
            "Keep A1689 geometry-only because the frozen systematic grid failed; do not remove a sky/range/degree variant."
        ),
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
