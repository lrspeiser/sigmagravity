#!/usr/bin/env python3
"""Register/combine the frozen baseline A1689 frames and audit coverage."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import astrodata
import gemini_instruments  # noqa: F401 - registers Gemini AstroData classes
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/r1_a1689_gmos_reduction_covariance_protocol.json"
SKY_REPORT = ROOT / "results/r1_a1689_gmos_sky_models/report.json"
CENTER_REPORT = ROOT / "results/r1_a1689_gmos_continuum_center/report.json"
CENTER_DATA = ROOT / "data/derived/r1_a1689_gmos_reconstruction/continuum_center_profiles.npz"
OUTPUT = ROOT / "data/derived/r1_a1689_gmos_reconstruction/a1689_baseline_registered_combined.npz"
REPORT = ROOT / "results/r1_a1689_gmos_combination/report.json"
NO_DATA_BIT = np.uint16(16)


def bilinear_resample(data: np.ndarray, variance: np.ndarray, dq: np.ndarray,
                      ycoord: np.ndarray, xcoord: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    yy, xx = np.meshgrid(ycoord, xcoord, indexing="ij")
    y0 = np.floor(yy).astype(int)
    x0 = np.floor(xx).astype(int)
    fy = yy - y0
    fx = xx - x0
    inside = (y0 >= 0) & (x0 >= 0) & (y0 + 1 < data.shape[0]) & (x0 + 1 < data.shape[1])
    safe_y = np.clip(y0, 0, data.shape[0] - 2)
    safe_x = np.clip(x0, 0, data.shape[1] - 2)
    weights = [
        (1 - fy) * (1 - fx),
        (1 - fy) * fx,
        fy * (1 - fx),
        fy * fx,
    ]
    coordinates = [
        (safe_y, safe_x),
        (safe_y, safe_x + 1),
        (safe_y + 1, safe_x),
        (safe_y + 1, safe_x + 1),
    ]
    out_data = np.zeros_like(xx, dtype=float)
    out_variance = np.zeros_like(xx, dtype=float)
    finite_neighbors = np.ones_like(inside)
    for weight, (iy, ix) in zip(weights, coordinates):
        value = data[iy, ix]
        var = variance[iy, ix]
        finite_neighbors &= np.isfinite(value) & np.isfinite(var) & (var >= 0)
        out_data += weight * value
        out_variance += np.square(weight) * var
    nearest_y = np.clip(np.rint(yy).astype(int), 0, data.shape[0] - 1)
    nearest_x = np.clip(np.rint(xx).astype(int), 0, data.shape[1] - 1)
    out_dq = dq[nearest_y, nearest_x].copy()
    invalid = ~inside | ~finite_neighbors
    out_data[invalid] = np.nan
    out_variance[invalid] = np.nan
    out_dq[invalid] |= NO_DATA_BIT
    return out_data, out_variance, out_dq


def input_axes(ad, offset: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ext = ad[0]
    ny, nx = ext.data.shape
    wavelength_nm, _, _ = ext.wcs(np.arange(nx, dtype=float), np.full(nx, 0.5 * (ny - 1)))
    return np.asarray(wavelength_nm, dtype=float) * 10.0, offset


def pixel_coordinate(values: np.ndarray, targets: np.ndarray) -> np.ndarray:
    order = np.argsort(values)
    return np.interp(targets, values[order], np.arange(len(values), dtype=float)[order], left=np.nan, right=np.nan)


def main() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    sky = json.loads(SKY_REPORT.read_text(encoding="utf-8"))
    center = json.loads(CENTER_REPORT.read_text(encoding="utf-8"))
    if not sky["authorization"]["register_and_combine_baseline_exposures_for_coverage_audit"]:
        raise RuntimeError("Frozen sky gates did not authorize combination")
    execution = config["spatial_extraction"]["combination_execution"]
    wave_min, wave_max = execution["common_wavelength_range_angstrom"]
    wave_step = execution["common_wavelength_step_angstrom"]
    wavelength = np.arange(wave_min, wave_max + 0.25 * wave_step, wave_step)
    spatial_max = min(abs(value) for value in execution["common_signed_spatial_range_arcsec"])
    spatial_step = execution["common_signed_spatial_step_arcsec"]
    spatial_half_count = int(np.floor(spatial_max / spatial_step))
    signed_offset = np.arange(-spatial_half_count, spatial_half_count + 1) * spatial_step
    center_arrays = np.load(CENTER_DATA)
    individual_centers = {
        item["science"]: item["center_arcsec"] for item in center["individual_fits"]
    }

    resampled_data = []
    resampled_variance = []
    resampled_dq = []
    input_rows = []
    for item in sky["products"]:
        science = item["science"]
        path = ROOT / item["baseline_sky_subtracted_output"]
        ad = astrodata.open(path)
        stem = Path(science).stem
        absolute_offset = np.asarray(center_arrays[f"{stem}_offset_arcsec"], dtype=float)
        input_wavelength, input_offset = input_axes(ad, absolute_offset)
        xcoord = pixel_coordinate(input_wavelength, wavelength)
        # Sampling target+individual_center shifts only the continuum centroid.
        ycoord = pixel_coordinate(input_offset, signed_offset + individual_centers[science])
        ext = ad[0]
        data, variance, dq = bilinear_resample(
            np.asarray(ext.data, dtype=float),
            np.asarray(ext.variance, dtype=float),
            np.asarray(ext.mask, dtype=np.uint16),
            ycoord,
            xcoord,
        )
        resampled_data.append(data)
        resampled_variance.append(variance)
        resampled_dq.append(dq)
        input_rows.append({
            "science": science,
            "baseline_input": item["baseline_sky_subtracted_output"],
            "individual_center_shift_arcsec": individual_centers[science],
            "input_wavelength_range_angstrom": [float(np.nanmin(input_wavelength)), float(np.nanmax(input_wavelength))],
            "input_spatial_range_arcsec": [float(np.nanmin(input_offset)), float(np.nanmax(input_offset))],
        })

    data = np.asarray(resampled_data)
    variance = np.asarray(resampled_variance)
    dq = np.asarray(resampled_dq, dtype=np.uint16)
    accepted = (dq == 0) & np.isfinite(data) & np.isfinite(variance) & (variance > 0)
    for _ in range(execution["combination_clipping_iterations"]):
        weight = np.divide(1.0, variance, out=np.zeros_like(variance), where=accepted)
        weight_sum = weight.sum(axis=0)
        mean = np.divide(
            (weight * np.where(accepted, data, 0.0)).sum(axis=0),
            weight_sum,
            out=np.full(weight_sum.shape, np.nan),
            where=weight_sum > 0,
        )
        standardized = np.divide(
            np.abs(data - mean[None, :, :]),
            np.sqrt(variance),
            out=np.full_like(variance, np.inf),
            where=variance > 0,
        )
        accepted &= ~(standardized > 4.0)

    coverage = accepted.sum(axis=0).astype(np.uint8)
    minimum_coverage = config["calibration_acceptance"]["minimum_unmasked_exposure_coverage_per_combined_pixel"]
    combined_good = coverage >= minimum_coverage
    weight = np.divide(1.0, variance, out=np.zeros_like(variance), where=accepted)
    weight_sum = weight.sum(axis=0)
    combined = np.divide(
        (weight * np.where(accepted, data, 0.0)).sum(axis=0),
        weight_sum,
        out=np.full(weight_sum.shape, np.nan),
        where=combined_good,
    )
    combined_variance = np.divide(
        1.0, weight_sum, out=np.full(weight_sum.shape, np.nan), where=combined_good
    )
    combined_dq = np.where(combined_good, 0, NO_DATA_BIT).astype(np.uint16)
    support = np.abs(signed_offset) <= max(abs(value) for value in config["spatial_extraction"]["signed_bin_edges_arcsec"])
    support_coverage = coverage[support]
    unmasked_support = combined_dq[support] == 0
    coverage_integrity = bool(
        np.all(support_coverage[unmasked_support] >= minimum_coverage)
        and np.all(combined_dq[support][support_coverage < minimum_coverage] & NO_DATA_BIT)
    )
    retained_fraction = float(np.mean(unmasked_support))

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUTPUT,
        wavelength_angstrom=wavelength,
        signed_offset_arcsec=signed_offset,
        exposure_science_electron=data.astype(np.float32),
        exposure_variance_electron2=variance.astype(np.float32),
        exposure_dq=dq,
        exposure_accepted_after_clipping=accepted,
        combined_science_electron=combined.astype(np.float32),
        combined_variance_electron2=combined_variance.astype(np.float32),
        combined_dq=combined_dq,
        exposure_coverage=coverage,
    )
    p2_gate = bool(
        sky["gates"]["P2d_all_frozen_sky_variants_gate_passed"]
        and coverage_integrity
    )
    unique, counts = np.unique(support_coverage, return_counts=True)
    report = {
        "report_version": "R1B1-A1689-GMOS-combination-0.1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "selection_blind": True,
        "inputs": input_rows,
        "common_grid": {
            "wavelength_start_angstrom": float(wavelength[0]),
            "wavelength_stop_angstrom": float(wavelength[-1]),
            "wavelength_step_angstrom": wave_step,
            "wavelength_pixels": len(wavelength),
            "signed_spatial_start_arcsec": float(signed_offset[0]),
            "signed_spatial_stop_arcsec": float(signed_offset[-1]),
            "signed_spatial_step_arcsec": spatial_step,
            "spatial_pixels": len(signed_offset),
        },
        "coverage_counts_within_frozen_signed_support": {
            str(int(value)): int(count) for value, count in zip(unique, counts)
        },
        "retained_unmasked_fraction_within_frozen_signed_support": retained_fraction,
        "minimum_coverage_for_unmasked_pixel": minimum_coverage,
        "all_subthreshold_pixels_masked": coverage_integrity,
        "output": str(OUTPUT.relative_to(ROOT)).replace("\\", "/"),
        "gates": {
            "P2d_all_frozen_sky_variants_gate_passed": sky["gates"]["P2d_all_frozen_sky_variants_gate_passed"],
            "P2e_combined_coverage_mask_gate_passed": coverage_integrity,
            "P2_calibrated_2d_sky_centroid_coverage_gate_passed": p2_gate,
            "P3_profile_covariance_gate_passed": False,
            "gravity_response_fit_authorized": False,
        },
        "authorization": {
            "fit_frozen_nine_signed_stellar_kinematic_bins": p2_gate,
            "infer_gravity_response": False,
            "fit_lens_mass_model": False,
            "fit_new_force_or_action": False,
        },
        "next_action": (
            "Extract the nine frozen signed bins, apply the S/N gate, and run the preregistered pPXF/covariance protocol without changing bin edges or masks."
            if p2_gate else
            "Retain the failed P2 coverage mask and keep A1689 geometry-only; do not relax coverage or binning."
        ),
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
