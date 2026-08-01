from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
from astropy.io import fits
import astropy.units as u
from astropy.wcs import WCS

try:
    from scripts.reconstruct_m1206_ppxf import (
        _elliptical_coordinates,
        _fit_spectrum,
        _register_center,
        _sha256,
        _source_mask,
        _wavelength,
    )
except ModuleNotFoundError:  # Direct execution from the scripts directory.
    from reconstruct_m1206_ppxf import (  # type: ignore[no-redef]
        _elliptical_coordinates,
        _fit_spectrum,
        _register_center,
        _sha256,
        _source_mask,
        _wavelength,
    )


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INVENTORY = ROOT / "configs/r1_m1206_level2_products.json"
DEFAULT_PROTOCOL = ROOT / "configs/r1_m1206_ppxf_protocol.json"


def _extract_spectrum(
    data: np.ndarray,
    variance: np.ndarray,
    wavelength: np.ndarray,
    spatial_mask: np.ndarray,
    wavelength_range: list[float],
    minimum_valid_fraction: float,
) -> dict:
    pixel_count = int(spatial_mask.sum())
    selected_data = data[:, spatial_mask]
    selected_variance = variance[:, spatial_mask]
    valid = (
        np.isfinite(selected_data)
        & np.isfinite(selected_variance)
        & (selected_variance > 0)
    )
    valid_count = valid.sum(axis=1)
    enough = valid_count >= minimum_valid_fraction * pixel_count
    spectrum = np.where(valid, selected_data, 0).sum(axis=1, dtype=np.float64)
    summed_variance = np.where(valid, selected_variance, 0).sum(
        axis=1, dtype=np.float64
    )
    keep = (
        (wavelength >= wavelength_range[0])
        & (wavelength <= wavelength_range[1])
        & enough
        & np.isfinite(spectrum)
        & np.isfinite(summed_variance)
        & (summed_variance > 0)
    )
    if keep.sum() < 1000:
        raise ValueError("too few valid wavelength planes in an extracted spectrum")
    return {
        "wavelength": wavelength[keep],
        "spectrum": spectrum[keep],
        "variance": summed_variance[keep],
        "spaxels": pixel_count,
    }


def _common_wavelength_grid(extractions: list[dict]) -> np.ndarray:
    step = float(
        np.median(
            np.concatenate(
                [np.diff(item["wavelength"]) for item in extractions]
            )
        )
    )
    lower = max(float(item["wavelength"][0]) for item in extractions)
    upper = min(float(item["wavelength"][-1]) for item in extractions)
    start = np.ceil(lower / step) * step
    stop = np.floor(upper / step) * step
    return start + np.arange(int(np.floor((stop - start) / step)) + 1) * step


def _coadd_spectra(
    extractions: list[dict],
    normalization_window: list[float],
    minimum_contributors: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    common = _common_wavelength_grid(extractions)
    weighted_flux = np.zeros_like(common)
    weight_sum = np.zeros_like(common)
    contributors = np.zeros_like(common, dtype=int)
    normalizations: list[float] = []
    for item in extractions:
        wavelength = item["wavelength"]
        spectrum = item["spectrum"]
        variance = item["variance"]
        window = (
            (wavelength >= normalization_window[0])
            & (wavelength <= normalization_window[1])
        )
        normalization = float(np.median(spectrum[window]))
        if not np.isfinite(normalization) or normalization <= 0:
            raise ValueError("invalid continuum normalization")
        normalizations.append(normalization)
        flux_interp = np.interp(common, wavelength, spectrum / normalization)
        # Linear variance interpolation is conservative relative to squared
        # interpolation weights; resampling covariance remains explicitly missing.
        variance_interp = np.interp(
            common, wavelength, variance / normalization**2
        )
        usable = (
            (common >= wavelength[0])
            & (common <= wavelength[-1])
            & np.isfinite(flux_interp)
            & np.isfinite(variance_interp)
            & (variance_interp > 0)
        )
        weight = np.zeros_like(common)
        weight[usable] = 1.0 / variance_interp[usable]
        weighted_flux += weight * np.nan_to_num(flux_interp)
        weight_sum += weight
        contributors += usable.astype(int)
    if np.any(contributors < minimum_contributors):
        raise ValueError("common grid has too few contributing level-2 products")
    coadd = weighted_flux / weight_sum
    coadd_variance = 1.0 / weight_sum
    metadata = {
        "wavelength_min_angstrom": float(common[0]),
        "wavelength_max_angstrom": float(common[-1]),
        "wavelength_step_angstrom": float(np.median(np.diff(common))),
        "wavelength_pixels": int(common.size),
        "minimum_contributors": int(contributors.min()),
        "maximum_contributors": int(contributors.max()),
        "continuum_normalizations": normalizations,
    }
    return common, coadd, coadd_variance, metadata


def build_level2_reconstruction(
    inventory_path: Path,
    protocol_path: Path,
    profile_output: Path,
    report_output: Path,
    mask_plot_output: Path,
) -> dict:
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    provenance_path = (
        ROOT
        / "data/raw/r1_muse_bcg_cubes/macs_j1206_level2/provenance.json"
    )
    provenance = json.loads(provenance_path.read_text(encoding="utf-8-sig"))
    provenance_by_id = {item["dp_id"]: item for item in provenance["files"]}
    template_path = ROOT / protocol["spectral_fit"]["template_path"]
    if _sha256(template_path) != protocol["spectral_fit"]["template_sha256"]:
        raise ValueError("template checksum does not match the frozen protocol")

    spatial = protocol["spatial_extraction"]
    coadd_protocol = inventory["coadd_protocol"]
    extraction_by_region: dict[str, list[dict]] = {}
    product_summaries = []
    figure, axes = plt.subplots(2, 3, figsize=(13, 8.5))
    edges = spatial["annulus_semimajor_edges_arcsec"]

    for product_index, (product, axis) in enumerate(
        zip(inventory["products"], axes.flat)
    ):
        cube_path = ROOT / product["local_path"]
        provenance_item = provenance_by_id[product["dp_id"]]
        checksum_pass = _sha256(cube_path) == provenance_item["sha256"]
        if not checksum_pass:
            raise ValueError(f"checksum mismatch for {product['dp_id']}")
        with fits.open(cube_path, memmap=True) as hdul:
            data = np.asarray(
                hdul[protocol["input"]["data_extension"]].data,
                dtype=np.float32,
            )
            variance = np.asarray(
                hdul[protocol["input"]["variance_extension"]].data,
                dtype=np.float32,
            )
            header = hdul[protocol["input"]["data_extension"]].header
            wavelength = _wavelength(header)
            initial_x, initial_y = WCS(header).celestial.world_to_pixel(
                SkyCoord(
                    protocol["input"]["bcg_center_ra_deg"] * u.deg,
                    protocol["input"]["bcg_center_dec_deg"] * u.deg,
                )
            )
            white_light = np.nanmedian(
                data[(wavelength >= 6000) & (wavelength <= 7000)], axis=0
            )
            pixel_scale = abs(header["CD2_2"]) * 3600
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
            source_mask, mask_summary = _source_mask(white_light, radius)
            for bin_index, (lower, upper) in enumerate(
                zip(edges[:-1], edges[1:]), start=1
            ):
                annulus = (
                    (radius >= lower)
                    & (radius < upper)
                    & ~source_mask
                    & np.isfinite(white_light)
                )
                regions = {f"bin_{bin_index}_full": annulus}
                if bin_index >= 5:
                    regions[f"bin_{bin_index}_positive_major"] = annulus & (major >= 0)
                    regions[f"bin_{bin_index}_negative_major"] = annulus & (major < 0)
                for name, region_mask in regions.items():
                    extracted = _extract_spectrum(
                        data,
                        variance,
                        wavelength,
                        region_mask,
                        protocol["input"]["observed_wavelength_range_angstrom"],
                        spatial["minimum_valid_spaxel_fraction_per_bin"],
                    )
                    extracted["product_index"] = product_index
                    extraction_by_region.setdefault(name, []).append(extracted)

        registration.update(
            {
                "dp_id": product["dp_id"],
                "checksum_pass": checksum_pass,
                "wavelength_min_angstrom": float(wavelength[0]),
                "wavelength_max_angstrom": float(wavelength[-1]),
                "wavelength_step_angstrom": float(np.median(np.diff(wavelength))),
                "source_mask": mask_summary,
            }
        )
        product_summaries.append(registration)
        low, high = np.nanpercentile(white_light, [5, 99.5])
        display = np.arcsinh(
            np.clip((white_light - low) / (high - low), 0, None) * 5
        )
        axis.imshow(display, origin="lower", cmap="gray")
        axis.contour(radius, levels=edges[1:], colors="cyan", linewidths=0.4)
        overlay = np.zeros((*source_mask.shape, 4))
        overlay[source_mask] = [1, 0, 0, 0.4]
        axis.imshow(overlay, origin="lower")
        axis.plot(registered[0], registered[1], "c+", markersize=10)
        axis.set_title(product["dp_id"].replace("ADP.", ""), fontsize=8)
        axis.set_xticks([])
        axis.set_yticks([])

    mask_plot_output.parent.mkdir(parents=True, exist_ok=True)
    figure.suptitle("MACS J1206 level-2 registered annuli and source masks")
    figure.tight_layout()
    figure.savefig(mask_plot_output, dpi=170)
    plt.close(figure)

    normalization_window = coadd_protocol[
        "relative_flux_normalization_angstrom"
    ]
    minimum_contributors = coadd_protocol[
        "minimum_contributing_products_each_wavelength"
    ]
    coadded: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, dict]] = {}
    fit_by_region = {}
    sps = None
    for region_name, extractions in extraction_by_region.items():
        coadded[region_name] = _coadd_spectra(
            extractions, normalization_window, minimum_contributors
        )
        wavelength, spectrum, variance, _ = coadded[region_name]
        fit_by_region[region_name], sps = _fit_spectrum(
            spectrum,
            variance,
            wavelength,
            protocol["input"]["bcg_redshift_initial"],
            template_path,
            protocol["spectral_fit"]["galaxy_fwhm_angstrom_baseline"],
            protocol["spectral_fit"]["additive_polynomial_degree"],
            sps,
        )

    rows = []
    for bin_index, (lower, upper) in enumerate(zip(edges[:-1], edges[1:]), start=1):
        row = dict(fit_by_region[f"bin_{bin_index}_full"])
        _, spectrum, variance, metadata = coadded[f"bin_{bin_index}_full"]
        row.update(
            {
                "bin": bin_index,
                "semimajor_min_arcsec": lower,
                "semimajor_max_arcsec": upper,
                "median_signal_to_noise_per_angstrom": float(
                    np.median(spectrum / np.sqrt(variance))
                ),
                "minimum_contributing_products": metadata["minimum_contributors"],
            }
        )
        if bin_index >= 5:
            positive = fit_by_region[f"bin_{bin_index}_positive_major"]
            negative = fit_by_region[f"bin_{bin_index}_negative_major"]
            row["opposite_half_velocity_difference_km_s"] = abs(
                positive["velocity_km_s"] - negative["velocity_km_s"]
            )
            row["opposite_half_sigma_difference_fraction"] = abs(
                positive["sigma_km_s"] - negative["sigma_km_s"]
            ) / np.mean([positive["sigma_km_s"], negative["sigma_km_s"]])
        rows.append(row)
    profile = pd.DataFrame(rows)
    profile_output.parent.mkdir(parents=True, exist_ok=True)
    profile.to_csv(profile_output, index=False)

    leave_one_out = []
    outer_region_names = [
        "bin_6_full",
        "bin_6_positive_major",
        "bin_6_negative_major",
    ]
    for omitted_index, product in enumerate(inventory["products"]):
        for region_name in outer_region_names:
            subset = [
                item
                for item in extraction_by_region[region_name]
                if item["product_index"] != omitted_index
            ]
            wavelength, spectrum, variance, _ = _coadd_spectra(
                subset,
                normalization_window,
                min(minimum_contributors, len(subset)),
            )
            fit, sps = _fit_spectrum(
                spectrum,
                variance,
                wavelength,
                protocol["input"]["bcg_redshift_initial"],
                template_path,
                protocol["spectral_fit"]["galaxy_fwhm_angstrom_baseline"],
                protocol["spectral_fit"]["additive_polynomial_degree"],
                sps,
            )
            baseline = fit_by_region[region_name]
            leave_one_out.append(
                {
                    "omitted_dp_id": product["dp_id"],
                    "region": region_name,
                    **fit,
                    "absolute_velocity_shift_km_s": abs(
                        fit["velocity_km_s"] - baseline["velocity_km_s"]
                    ),
                    "absolute_sigma_shift_fraction": abs(
                        fit["sigma_km_s"] - baseline["sigma_km_s"]
                    )
                    / baseline["sigma_km_s"],
                }
            )

    base_thresholds = protocol["success_thresholds"]
    level2_thresholds = inventory["success_thresholds"]
    outer_rows = profile.iloc[-2:]
    registration_pass = all(
        item["registration_offset_arcsec"]
        <= level2_thresholds["maximum_registration_offset_arcsec_each_product"]
        for item in product_summaries
    )
    finite_pass = bool(np.isfinite(profile["sigma_km_s"]).all())
    outer_signal_pass = bool(
        profile.iloc[-1]["median_signal_to_noise_per_angstrom"]
        >= base_thresholds[
            "minimum_outer_bin_median_signal_to_noise_per_angstrom"
        ]
    )
    fractional_error_pass = bool(
        (
            profile["sigma_formal_error_km_s"] / profile["sigma_km_s"]
            <= base_thresholds["maximum_fractional_sigma_uncertainty_each_bin"]
        ).all()
    )
    half_velocity_pass = bool(
        (
            outer_rows["opposite_half_velocity_difference_km_s"]
            <= base_thresholds[
                "maximum_opposite_half_velocity_difference_km_s"
            ]
        ).all()
    )
    half_sigma_pass = bool(
        (
            outer_rows["opposite_half_sigma_difference_fraction"]
            <= base_thresholds[
                "maximum_opposite_half_sigma_difference_fraction"
            ]
        ).all()
    )
    loo_velocity_pass = all(
        item["absolute_velocity_shift_km_s"]
        <= level2_thresholds["maximum_leave_one_out_velocity_shift_km_s"]
        for item in leave_one_out
    )
    loo_sigma_pass = all(
        item["absolute_sigma_shift_fraction"]
        <= level2_thresholds["maximum_leave_one_out_sigma_shift_fraction"]
        for item in leave_one_out
    )
    baseline_pass = all(
        [
            registration_pass,
            finite_pass,
            outer_signal_pass,
            fractional_error_pass,
            half_velocity_pass,
            half_sigma_pass,
            loo_velocity_pass,
            loo_sigma_pass,
        ]
    )
    report_output.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "inventory_version": inventory["inventory_version"],
        "kinematic_protocol_version": protocol["protocol_version"],
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "method": "per-product registered annular extraction followed by common-grid inverse-variance spectral coaddition",
        "products": product_summaries,
        "coadd_metadata": {
            name: values[3] for name, values in coadded.items()
        },
        "coadd_half_fits": {
            name: fit_by_region[name]
            for name in fit_by_region
            if "positive_major" in name or "negative_major" in name
        },
        "leave_one_product_out_outer_diagnostics": leave_one_out,
        "gates": {
            "six_product_checksum_pass": True,
            "all_registration_offsets_pass": registration_pass,
            "six_finite_sigma_pass": finite_pass,
            "outer_signal_to_noise_pass": outer_signal_pass,
            "formal_fractional_uncertainty_pass": fractional_error_pass,
            "opposite_half_velocity_pass": half_velocity_pass,
            "opposite_half_sigma_pass": half_sigma_pass,
            "leave_one_product_out_velocity_pass": loo_velocity_pass,
            "leave_one_product_out_sigma_pass": loo_sigma_pass,
            "baseline_internal_consistency_pass": baseline_pass,
            "full_covariance_protocol_complete": False,
            "r1_profile_ready": False,
        },
        "decision": (
            "Level-2 baseline checks pass; proceed to the frozen covariance and template-systematic stage, without fitting gravity."
            if baseline_pass
            else "Level-2 baseline checks fail; do not fit gravity or average through the failed internal diagnostic."
        ),
        "outputs": {
            "profile": str(profile_output.relative_to(ROOT)).replace("\\", "/"),
            "mask_plot": str(mask_plot_output.relative_to(ROOT)).replace("\\", "/"),
        },
    }
    report_output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inventory", type=Path, default=DEFAULT_INVENTORY)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument(
        "--profile-output",
        type=Path,
        default=ROOT / "data/derived/r1_m1206_level2_ppxf_profile.csv",
    )
    parser.add_argument(
        "--report-output",
        type=Path,
        default=ROOT / "results/r1_m1206_level2_ppxf/report.json",
    )
    parser.add_argument(
        "--mask-plot-output",
        type=Path,
        default=ROOT / "results/r1_m1206_level2_ppxf/source_masks.png",
    )
    args = parser.parse_args()
    report = build_level2_reconstruction(
        args.inventory,
        args.protocol,
        args.profile_output,
        args.report_output,
        args.mask_plot_output,
    )
    print(json.dumps(report["gates"], indent=2))
    print(report["decision"])


if __name__ == "__main__":
    main()
