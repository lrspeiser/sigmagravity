from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.io import fits
from ppxf.ppxf import ppxf
import ppxf.ppxf_util as ppxf_util
import ppxf.sps_util as sps_util
from scipy.ndimage import binary_dilation, gaussian_filter, label
from scipy.optimize import least_squares
from scipy.signal import medfilt


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/r1_m1206_ppxf_protocol.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def _wavelength(header: fits.Header) -> np.ndarray:
    index = np.arange(header["NAXIS3"])
    return header["CRVAL3"] + (index + 1 - header["CRPIX3"]) * header["CD3_3"]


def _elliptical_coordinates(
    x: np.ndarray,
    y: np.ndarray,
    center_x: float,
    center_y: float,
    pixel_scale_arcsec: float,
    pa_deg: float,
    axis_ratio: float,
) -> tuple[np.ndarray, np.ndarray]:
    east = -(x - center_x) * pixel_scale_arcsec
    north = (y - center_y) * pixel_scale_arcsec
    angle = np.deg2rad(pa_deg)
    major = east * np.sin(angle) + north * np.cos(angle)
    minor = east * np.cos(angle) - north * np.sin(angle)
    elliptical_radius = np.sqrt(major**2 + (minor / axis_ratio) ** 2)
    return elliptical_radius, major


def _sersic_model(
    parameters: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    pixel_scale_arcsec: float,
    pa_deg: float,
    axis_ratio: float,
) -> np.ndarray:
    center_x, center_y, effective_radius, intensity, background, grad_x, grad_y = parameters
    radius, _ = _elliptical_coordinates(
        x, y, center_x, center_y, pixel_scale_arcsec, pa_deg, axis_ratio
    )
    index = 4.05
    b_n = 2 * index - 1 / 3
    profile = intensity * np.exp(
        -b_n * ((np.maximum(radius, 0.02) / effective_radius) ** (1 / index) - 1)
    )
    return profile + background + grad_x * (x - center_x) + grad_y * (y - center_y)


def _register_center(
    white_light: np.ndarray,
    initial_x: float,
    initial_y: float,
    pa_deg: float,
    axis_ratio: float,
    pixel_scale_arcsec: float,
) -> tuple[np.ndarray, dict]:
    y, x = np.mgrid[: white_light.shape[0], : white_light.shape[1]]
    fit_region = (
        (np.abs(x - initial_x) < 35)
        & (np.abs(y - initial_y) < 40)
        & np.isfinite(white_light)
    )
    xx = x[fit_region]
    yy = y[fit_region]
    values = white_light[fit_region]
    smooth = gaussian_filter(np.nan_to_num(white_light, nan=np.nanmedian(white_light)), 5)
    peak_y, peak_x = np.unravel_index(np.nanargmax(smooth), smooth.shape)
    initial = np.array([peak_x, peak_y, 4.8, 0.3, 0.0, 0.0, 0.0])

    def residual(parameters: np.ndarray) -> np.ndarray:
        return (
            _sersic_model(
                parameters, xx, yy, pixel_scale_arcsec, pa_deg, axis_ratio
            )
            - values
        )

    result = least_squares(
        residual,
        initial,
        bounds=(
            [initial_x - 10, initial_y - 20, 2, 0.001, -10, -1, -1],
            [initial_x + 20, initial_y + 10, 10, 10, 10, 1, 1],
        ),
        loss="soft_l1",
        f_scale=0.5,
        max_nfev=3000,
    )
    offset = np.hypot(result.x[0] - initial_x, result.x[1] - initial_y) * pixel_scale_arcsec
    return result.x, {
        "initial_pixel_x": initial_x,
        "initial_pixel_y": initial_y,
        "registered_pixel_x": float(result.x[0]),
        "registered_pixel_y": float(result.x[1]),
        "registration_offset_arcsec": float(offset),
        "fitted_effective_radius_arcsec": float(result.x[2]),
        "optimizer_cost": float(result.cost),
    }


def _source_mask(
    white_light: np.ndarray,
    radius: np.ndarray,
    detection_sigma: float = 6.0,
    dilation_pixels: int = 8,
) -> tuple[np.ndarray, dict]:
    filled = np.nan_to_num(white_light, nan=np.nanmedian(white_light))
    high_pass = white_light - gaussian_filter(filled, 3)
    reference = (radius > 3) & (radius < 15) & np.isfinite(high_pass)
    center = float(np.median(high_pass[reference]))
    robust_sigma = float(
        1.4826 * np.median(np.abs(high_pass[reference] - center))
    )
    components, count = label(
        (high_pass > center + detection_sigma * robust_sigma) & reference
    )
    mask = np.zeros_like(white_light, dtype=bool)
    accepted = 0
    for component_id in range(1, count + 1):
        component = components == component_id
        area = int(component.sum())
        if 2 <= area <= 300:
            mask |= component
            accepted += 1
    mask = binary_dilation(mask, iterations=dilation_pixels)
    mask[radius < 3] = False
    return mask, {
        "detection_robust_sigma": robust_sigma,
        "accepted_components": accepted,
        "masked_pixels": int(mask.sum()),
        "dilation_pixels": dilation_pixels,
    }


def _fit_spectrum(
    spectrum: np.ndarray,
    variance: np.ndarray,
    wavelength: np.ndarray,
    redshift: float,
    template_path: Path,
    galaxy_fwhm: float,
    degree: int,
    sps: sps_util.sps_lib | None,
    variance_mask_sigma: float = 6.0,
) -> tuple[dict, sps_util.sps_lib]:
    rest_range = wavelength[[0, -1]] / (1 + redshift)
    galaxy, log_wavelength, velocity_scale = ppxf_util.log_rebin(rest_range, spectrum)
    variance_log, _, _ = ppxf_util.log_rebin(
        rest_range, variance, velscale=velocity_scale
    )
    common_length = min(galaxy.size, variance_log.size, log_wavelength.size)
    galaxy = galaxy[:common_length]
    variance_log = variance_log[:common_length]
    log_wavelength = log_wavelength[:common_length]
    noise = np.sqrt(np.maximum(variance_log, 1e-30))
    normalization = float(np.median(galaxy))
    galaxy /= normalization
    noise /= normalization
    if sps is None:
        sps = sps_util.sps_lib(
            template_path, velocity_scale, galaxy_fwhm / (1 + redshift)
        )
    templates = sps.templates.reshape(sps.templates.shape[0], -1)
    good_pixels = ppxf_util.determine_goodpixels(
        log_wavelength, [sps.lam_temp.min(), sps.lam_temp.max()], 0
    )
    local_noise = medfilt(noise, kernel_size=21)
    ratio = noise / np.maximum(local_noise, 1e-12)
    robust_sigma = 1.4826 * np.median(np.abs(ratio - np.median(ratio)))
    bad_variance = ratio > (
        np.median(ratio) + variance_mask_sigma * max(robust_sigma, 0.02)
    )
    good_pixels = np.intersect1d(good_pixels, np.where(~bad_variance)[0])
    fit = ppxf(
        templates,
        galaxy,
        noise,
        velocity_scale,
        [0, 300],
        goodpixels=good_pixels,
        moments=2,
        lam=np.exp(log_wavelength),
        lam_temp=sps.lam_temp,
        degree=degree,
        quiet=True,
    )
    scaled_error = fit.error * np.sqrt(fit.chi2)
    return (
        {
            "velocity_km_s": float(fit.sol[0]),
            "sigma_km_s": float(fit.sol[1]),
            "velocity_formal_error_km_s": float(scaled_error[0]),
            "sigma_formal_error_km_s": float(scaled_error[1]),
            "reduced_chi2": float(fit.chi2),
            "masked_wavelength_pixels": int(bad_variance.sum()),
            "fitted_wavelength_pixels": int(len(good_pixels)),
        },
        sps,
    )


def _extract_fit(
    data: np.ndarray,
    variance: np.ndarray,
    wavelength: np.ndarray,
    spatial_mask: np.ndarray,
    config: dict,
    template_path: Path,
    sps: sps_util.sps_lib | None,
) -> tuple[dict, sps_util.sps_lib]:
    valid = (
        np.isfinite(data[:, spatial_mask])
        & np.isfinite(variance[:, spatial_mask])
        & (variance[:, spatial_mask] > 0)
    )
    valid_count = valid.sum(axis=1)
    enough = valid_count >= (
        config["spatial_extraction"]["minimum_valid_spaxel_fraction_per_bin"]
        * int(spatial_mask.sum())
    )
    spectrum = np.where(valid, data[:, spatial_mask], 0).sum(axis=1)
    summed_variance = np.where(valid, variance[:, spatial_mask], 0).sum(axis=1)
    wave_range = config["input"]["observed_wavelength_range_angstrom"]
    selected = (
        (wavelength >= wave_range[0])
        & (wavelength <= wave_range[1])
        & enough
        & (summed_variance > 0)
        & np.isfinite(spectrum)
    )
    signal_to_noise = float(
        np.median(spectrum[selected] / np.sqrt(summed_variance[selected]))
    )
    result, sps = _fit_spectrum(
        spectrum[selected],
        summed_variance[selected],
        wavelength[selected],
        config["input"]["bcg_redshift_initial"],
        template_path,
        config["spectral_fit"]["galaxy_fwhm_angstrom_baseline"],
        config["spectral_fit"]["additive_polynomial_degree"],
        sps,
    )
    result.update(
        {
            "spaxels": int(spatial_mask.sum()),
            "median_signal_to_noise_per_angstrom": signal_to_noise,
        }
    )
    return result, sps


def build_reconstruction(
    config_path: Path,
    profile_output: Path,
    report_output: Path,
    mask_plot_output: Path,
) -> dict:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    cube_path = ROOT / config["input"]["cube_path"]
    template_path = ROOT / config["spectral_fit"]["template_path"]
    if _sha256(cube_path) != config["input"]["cube_sha256"]:
        raise ValueError("cube checksum does not match the frozen protocol")
    if _sha256(template_path) != config["spectral_fit"]["template_sha256"]:
        raise ValueError("template checksum does not match the frozen protocol")

    with fits.open(cube_path, memmap=True, checksum=True) as hdul:
        data = np.asarray(hdul[config["input"]["data_extension"]].data, dtype=float)
        variance = np.asarray(
            hdul[config["input"]["variance_extension"]].data, dtype=float
        )
        header = hdul[config["input"]["data_extension"]].header
        wavelength = _wavelength(header)
        from astropy.coordinates import SkyCoord
        from astropy.wcs import WCS
        import astropy.units as u

        celestial = WCS(header).celestial
        initial_x, initial_y = celestial.world_to_pixel(
            SkyCoord(
                config["input"]["bcg_center_ra_deg"] * u.deg,
                config["input"]["bcg_center_dec_deg"] * u.deg,
            )
        )
        white_light = np.nanmedian(
            data[(wavelength >= 6000) & (wavelength <= 7000)], axis=0
        )

    pixel_scale = abs(header["CD2_2"]) * 3600
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
    source_mask, mask_summary = _source_mask(white_light, radius)

    rows = []
    half_diagnostics = []
    sps = None
    edges = spatial["annulus_semimajor_edges_arcsec"]
    half_check_start_bin = max(1, len(edges) - 2)
    for bin_index, (lower, upper) in enumerate(zip(edges[:-1], edges[1:]), start=1):
        annulus = (
            (radius >= lower)
            & (radius < upper)
            & ~source_mask
            & np.isfinite(white_light)
        )
        fitted, sps = _extract_fit(
            data, variance, wavelength, annulus, config, template_path, sps
        )
        fitted.update(
            {
                "bin": bin_index,
                "semimajor_min_arcsec": lower,
                "semimajor_max_arcsec": upper,
            }
        )
        rows.append(fitted)

        if bin_index >= half_check_start_bin:
            half_results = []
            for half_name, side in (("positive_major", major >= 0), ("negative_major", major < 0)):
                half_fit, sps = _extract_fit(
                    data,
                    variance,
                    wavelength,
                    annulus & side,
                    config,
                    template_path,
                    sps,
                )
                half_fit.update({"bin": bin_index, "half": half_name})
                half_results.append(half_fit)
                half_diagnostics.append(half_fit)
            velocity_difference = abs(
                half_results[0]["velocity_km_s"] - half_results[1]["velocity_km_s"]
            )
            sigma_difference_fraction = abs(
                half_results[0]["sigma_km_s"] - half_results[1]["sigma_km_s"]
            ) / np.mean(
                [half_results[0]["sigma_km_s"], half_results[1]["sigma_km_s"]]
            )
            rows[-1]["opposite_half_velocity_difference_km_s"] = velocity_difference
            rows[-1]["opposite_half_sigma_difference_fraction"] = sigma_difference_fraction

    profile = pd.DataFrame(rows)
    profile_output.parent.mkdir(parents=True, exist_ok=True)
    report_output.parent.mkdir(parents=True, exist_ok=True)
    mask_plot_output.parent.mkdir(parents=True, exist_ok=True)
    profile.to_csv(profile_output, index=False)

    display_low, display_high = np.nanpercentile(white_light, [5, 99.5])
    display = np.arcsinh(
        np.clip((white_light - display_low) / (display_high - display_low), 0, None) * 5
    )
    overlay = np.zeros((*source_mask.shape, 4))
    overlay[source_mask] = [1, 0, 0, 0.45]
    figure, axis = plt.subplots(figsize=(8, 8))
    axis.imshow(display, origin="lower", cmap="gray")
    axis.imshow(overlay, origin="lower")
    axis.contour(radius, levels=edges[1:], colors="cyan", linewidths=0.6)
    axis.plot(registered[0], registered[1], "c+", markersize=15, markeredgewidth=2)
    axis.set_title(
        f"{config['input'].get('system_name', 'MACS J1206')} registered annuli "
        "and frozen compact-source mask"
    )
    figure.tight_layout()
    figure.savefig(mask_plot_output, dpi=180)
    plt.close(figure)

    thresholds = config["success_thresholds"]
    finite_sigma_bins = int(np.isfinite(profile["sigma_km_s"]).sum())
    minimum_finite_sigma_bins = int(thresholds["minimum_finite_sigma_bins"])
    finite = finite_sigma_bins >= minimum_finite_sigma_bins
    outer_signal = bool(
        profile.iloc[-1]["median_signal_to_noise_per_angstrom"]
        >= thresholds["minimum_outer_bin_median_signal_to_noise_per_angstrom"]
    )
    fractional_error = bool(
        (
            profile["sigma_formal_error_km_s"] / profile["sigma_km_s"]
            <= thresholds["maximum_fractional_sigma_uncertainty_each_bin"]
        ).all()
    )
    outer_rows = profile.iloc[-2:]
    half_velocity = bool(
        (
            outer_rows["opposite_half_velocity_difference_km_s"]
            <= thresholds["maximum_opposite_half_velocity_difference_km_s"]
        ).all()
    )
    half_sigma = bool(
        (
            outer_rows["opposite_half_sigma_difference_fraction"]
            <= thresholds["maximum_opposite_half_sigma_difference_fraction"]
        ).all()
    )
    registration_ok = (
        registration["registration_offset_arcsec"]
        <= config["input"]["maximum_allowed_registration_offset_arcsec"]
    )
    covariance_complete = False
    baseline_pass = all(
        [finite, outer_signal, fractional_error, half_velocity, half_sigma, registration_ok]
    )
    report = {
        "protocol_version": config["protocol_version"],
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "registration": registration,
        "source_mask": mask_summary,
        "gates": {
            "registration_offset_pass": registration_ok,
            "finite_sigma_bins": finite_sigma_bins,
            "minimum_finite_sigma_bins": minimum_finite_sigma_bins,
            "minimum_finite_sigma_bins_pass": finite,
            "outer_signal_to_noise_pass": outer_signal,
            "formal_fractional_uncertainty_pass": fractional_error,
            "opposite_half_velocity_pass": half_velocity,
            "opposite_half_sigma_pass": half_sigma,
            "baseline_internal_consistency_pass": baseline_pass,
            "full_covariance_protocol_complete": covariance_complete,
            "r1_profile_ready": baseline_pass and covariance_complete,
        },
        "half_diagnostics": half_diagnostics,
        "decision": (
            "Proceed to frozen covariance and template-systematic runs; do not fit gravity."
            if baseline_pass
            else "Do not average through the failed internal check. Revise the residual-blind source/background mask protocol before covariance runs or gravity fitting."
        ),
        "outputs": {
            "profile": str(profile_output.resolve().relative_to(ROOT.resolve())).replace("\\", "/"),
            "mask_plot": str(mask_plot_output.resolve().relative_to(ROOT.resolve())).replace("\\", "/"),
        },
    }
    report_output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--profile-output",
        type=Path,
        default=ROOT / "data/derived/r1_m1206_ppxf_engineering_profile.csv",
    )
    parser.add_argument(
        "--report-output",
        type=Path,
        default=ROOT / "results/r1_m1206_ppxf/report.json",
    )
    parser.add_argument(
        "--mask-plot-output",
        type=Path,
        default=ROOT / "results/r1_m1206_ppxf/source_mask.png",
    )
    args = parser.parse_args()
    report = build_reconstruction(
        args.config, args.profile_output, args.report_output, args.mask_plot_output
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
