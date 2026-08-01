#!/usr/bin/env python3
"""Execute the frozen MS2137 P2 centroid, mask, validity, and S/N gate."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.wcs import WCS
from scipy.ndimage import binary_dilation, gaussian_filter, label

try:
    from scripts.reconstruct_m1206_ppxf import (
        _elliptical_coordinates,
        _register_center,
        _wavelength,
    )
except ModuleNotFoundError:
    from reconstruct_m1206_ppxf import (
        _elliptical_coordinates,
        _register_center,
        _wavelength,
    )


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/r1_ms2137_ppxf_covariance_protocol.json"
PROTOCOL_REPORT_PATH = ROOT / "results/r1_ms2137_ppxf_protocol/report.json"
REPORT_PATH = ROOT / "results/r1_ms2137_ppxf_geometry/report.json"
PLOT_PATH = ROOT / "results/r1_ms2137_ppxf_geometry/mask_diagnostic.png"


def source_mask(white_light: np.ndarray, radius: np.ndarray, config: dict) -> tuple[np.ndarray, dict]:
    filled = np.nan_to_num(white_light, nan=float(np.nanmedian(white_light)))
    high_pass = white_light - gaussian_filter(filled, 3)
    reference = (radius > 3.0) & (radius < 17.0) & np.isfinite(high_pass)
    center = float(np.median(high_pass[reference]))
    robust_sigma = float(1.4826 * np.median(np.abs(high_pass[reference] - center)))
    components, count = label((high_pass > center + 6.0 * robust_sigma) & reference)
    mask = np.zeros_like(white_light, dtype=bool)
    accepted = 0
    component_areas = []
    for component_id in range(1, count + 1):
        component = components == component_id
        area = int(component.sum())
        if 2 <= area <= 300:
            mask |= component
            accepted += 1
            component_areas.append(area)
    mask = binary_dilation(mask, iterations=8)
    mask[radius < 3.0] = False
    return mask, {
        "reference_inner_arcsec": 3.0,
        "reference_outer_arcsec": 17.0,
        "detection_sigma": 6.0,
        "detection_robust_sigma": robust_sigma,
        "accepted_components": accepted,
        "accepted_component_areas_pixels": component_areas,
        "masked_pixels_after_dilation": int(mask.sum()),
        "dilation_pixels": 8,
    }


def annulus_metrics(
    data: np.ndarray,
    variance: np.ndarray,
    wavelength: np.ndarray,
    spatial_mask: np.ndarray,
    raw_mask: np.ndarray,
    half_coordinate: np.ndarray,
    lower: float,
    upper: float,
    config: dict,
) -> dict:
    spatial_count = int(spatial_mask.sum())
    raw_count = int(raw_mask.sum())
    masked_fraction = 1.0 - spatial_count / raw_count if raw_count else 1.0
    positive_half = int((spatial_mask & (half_coordinate >= 0)).sum())
    negative_half = int((spatial_mask & (half_coordinate < 0)).sum())
    values = data[:, spatial_mask]
    variances = variance[:, spatial_mask]
    valid = np.isfinite(values) & np.isfinite(variances) & (variances > 0)
    valid_count = valid.sum(axis=1)
    minimum_valid_spaxels = config["spatial_extraction"]["minimum_valid_wavelength_fraction_per_bin"] * spatial_count
    enough = valid_count >= minimum_valid_spaxels
    wavelength_fraction = float(enough.mean())
    spectrum = np.where(valid, values, 0.0).sum(axis=1)
    summed_variance = np.where(valid, variances, 0.0).sum(axis=1)
    selected = enough & np.isfinite(spectrum) & np.isfinite(summed_variance) & (summed_variance > 0)
    signal_to_noise = spectrum[selected] / np.sqrt(summed_variance[selected])
    median_signal_to_noise = float(np.median(signal_to_noise)) if signal_to_noise.size else float("nan")
    gates_cfg = config["stage_gates"]["P2_geometry_and_signal"]
    gates = {
        "exists": spatial_count > 0,
        "valid_wavelength_fraction_passed": wavelength_fraction >= gates_cfg["minimum_valid_wavelength_fraction_each_annulus"],
        "masked_spaxel_fraction_passed": masked_fraction <= gates_cfg["maximum_masked_spaxel_fraction_each_annulus"],
        "median_signal_to_noise_passed": median_signal_to_noise >= gates_cfg["minimum_median_signal_to_noise_each_annulus"],
        "both_opposite_halves_populated": positive_half >= config["spatial_extraction"]["minimum_unmasked_spaxels_per_opposite_half"] and negative_half >= config["spatial_extraction"]["minimum_unmasked_spaxels_per_opposite_half"],
    }
    return {
        "inner_arcsec": lower,
        "outer_arcsec": upper,
        "raw_spaxels": raw_count,
        "unmasked_spaxels": spatial_count,
        "masked_spaxel_fraction": masked_fraction,
        "positive_half_spaxels": positive_half,
        "negative_half_spaxels": negative_half,
        "valid_wavelength_fraction": wavelength_fraction,
        "median_signal_to_noise_per_native_pixel": median_signal_to_noise,
        "valid_wavelength_planes": int(selected.sum()),
        "gates": {**gates, "annulus_P2_passed": all(gates.values())},
    }


def build_report() -> dict:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    protocol = json.loads(PROTOCOL_REPORT_PATH.read_text(encoding="utf-8"))
    if not protocol["authorization"]["execute_P2_geometry_and_signal"]:
        raise RuntimeError("frozen protocol audit did not authorize P2")

    cube_path = ROOT / config["input"]["cube_path"]
    with fits.open(cube_path, mode="readonly", memmap=True) as hdul:
        data = hdul[config["input"]["data_extension"]].data
        variance = hdul[config["input"]["variance_extension"]].data
        header = hdul[config["input"]["data_extension"]].header
        wavelength = _wavelength(header)
        celestial = WCS(header).celestial
        initial_x, initial_y = celestial.world_to_pixel(
            SkyCoord(
                config["input"]["bcg_center_ra_deg"] * u.deg,
                config["input"]["bcg_center_dec_deg"] * u.deg,
            )
        )
        white_cfg = config["spatial_extraction"]["white_light_range_angstrom"]
        white_selection = (wavelength >= white_cfg[0]) & (wavelength <= white_cfg[1])
        white_light = np.nanmedian(data[white_selection], axis=0)

        pixel_scale = abs(float(header["CD2_2"])) * 3600.0
        spatial = config["spatial_extraction"]
        registered, registration = _register_center(
            white_light,
            float(initial_x),
            float(initial_y),
            spatial["position_angle_deg_east_of_north"],
            spatial["axis_ratio_b_over_a"],
            pixel_scale,
        )
        y, x = np.mgrid[: white_light.shape[0], : white_light.shape[1]]
        radius, major = _elliptical_coordinates(
            x,
            y,
            float(registered[0]),
            float(registered[1]),
            pixel_scale,
            spatial["position_angle_deg_east_of_north"],
            spatial["axis_ratio_b_over_a"],
        )
        compact_mask, mask_summary = source_mask(white_light, radius, config)
        edges = spatial["annulus_edges_arcsec"]
        annuli = []
        for lower, upper in zip(edges[:-1], edges[1:]):
            raw_mask = (radius >= lower) & (radius < upper) & np.isfinite(white_light)
            accepted_mask = raw_mask & ~compact_mask
            annuli.append(
                annulus_metrics(
                    data,
                    variance,
                    wavelength,
                    accepted_mask,
                    raw_mask,
                    major,
                    lower,
                    upper,
                    config,
                )
            )

    registration_pass = registration["registration_offset_arcsec"] <= config["stage_gates"]["P2_geometry_and_signal"]["registration_offset_arcsec_max"]
    all_annuli_pass = len(annuli) == config["spatial_extraction"]["annulus_count"] and all(item["gates"]["annulus_P2_passed"] for item in annuli)
    outer_support_pass = annuli[-1]["outer_arcsec"] >= config["structural_target"]["frozen_outer_edge_arcsec"] and annuli[-1]["gates"]["annulus_P2_passed"]
    gates = {
        "protocol_freeze_gate_passed": protocol["gates"]["protocol_freeze_gate_passed"],
        "registration_offset_passed": registration_pass,
        "all_nine_annuli_geometry_and_signal_passed": all_annuli_pass,
        "outer_8p5_to_14arcsec_annulus_passed": outer_support_pass,
    }
    passed = all(gates.values())

    PLOT_PATH.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(7, 6))
    image = axis.imshow(white_light, origin="lower", cmap="gray", vmin=np.nanpercentile(white_light, 5), vmax=np.nanpercentile(white_light, 99))
    axis.contour(compact_mask.astype(float), levels=[0.5], colors="tab:red", linewidths=0.5)
    theta = np.linspace(0, 2 * np.pi, 400)
    for edge in spatial["annulus_edges_arcsec"][1:]:
        radius_pixels = edge / pixel_scale
        axis.plot(registered[0] + radius_pixels * np.cos(theta), registered[1] + radius_pixels * np.sin(theta), color="tab:cyan", linewidth=0.45)
    axis.plot(initial_x, initial_y, marker="+", color="yellow", markersize=10, label="published/WCS")
    axis.plot(registered[0], registered[1], marker="x", color="lime", markersize=8, label="registered")
    axis.set(title="MS2137 frozen annuli and compact-source mask", xlabel="x pixel", ylabel="y pixel")
    axis.legend(loc="upper right", fontsize=8)
    figure.colorbar(image, ax=axis, label="median flux (6000-7000 A)")
    figure.tight_layout()
    figure.savefig(PLOT_PATH, dpi=170)
    plt.close(figure)

    report = {
        "report_version": "R1B2-MS2137-P2-geometry-signal-0.1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "system": config["input"]["system_name"],
        "cube_sha256": config["input"]["cube_sha256"],
        "data_and_variance_arrays_read": True,
        "ppxf_run": False,
        "registration": registration,
        "mask_summary": mask_summary,
        "annuli": annuli,
        "gates": {**gates, "P2_geometry_and_signal_gate_passed": passed},
        "decision": "authorize_P3_baseline_ppxf" if passed else "stop_MS2137_at_P2_geometry_and_signal",
        "next_action": "Run the frozen nine-bin XSL baseline and opposite-half pPXF fits; do not run covariance until P3 passes." if passed else "Record the failed annulus or centroid gate. Do not shrink support, merge bins, lower S/N, or run pPXF.",
        "authorization": {
            "execute_P3_baseline_ppxf": passed,
            "execute_P4_covariance": False,
            "change_support_or_thresholds": False,
            "infer_dynamical_or_weyl_response": False,
            "fit_gravity_response": False,
            "fit_new_force_or_action": False,
        },
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    print(json.dumps(build_report(), indent=2))
