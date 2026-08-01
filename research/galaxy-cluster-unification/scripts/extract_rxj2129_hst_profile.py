"""Extract the frozen, masked F125W/F814W RX J2129 radial light profile."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.wcs import WCS


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/r1_rxj2129_bcg_icl_protocol.json"


def _resolve(path: str) -> Path:
    return ROOT / path


def _read_catalog(path: Path) -> pd.DataFrame:
    header = next(
        line[2:].strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.startswith("# CLASHID")
    )
    catalog = pd.read_csv(
        path,
        sep=r"\s+",
        comment="#",
        names=header.split(),
        low_memory=False,
    )
    for column in ("x", "y", "a", "b", "theta"):
        catalog[column] = pd.to_numeric(catalog[column], errors="coerce")
    return catalog


def _center_pixel(header: fits.Header, geometry: dict[str, Any]) -> tuple[float, float]:
    wcs = WCS(header)
    # CLASH mosaics are already drizzled; retaining the detector SIP terms would
    # apply a second distortion correction despite the missing -SIP CTYPE suffix.
    wcs.sip = None
    center = SkyCoord(geometry["center_ra_deg"] * u.deg, geometry["center_dec_deg"] * u.deg)
    x, y = wcs.world_to_pixel(center)
    return float(np.asarray(x).item()), float(np.asarray(y).item())


def _make_source_mask(
    shape: tuple[int, int],
    catalog: pd.DataFrame,
    x_min: int,
    y_min: int,
    bcg_id: str,
    axis_multiplier: float,
) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    height, width = shape
    for source in catalog.itertuples(index=False):
        if str(source.CLASHID) == bcg_id:
            continue
        values = (source.x, source.y, source.a, source.b, source.theta)
        if not all(np.isfinite(value) for value in values):
            continue
        center_x = float(source.x) - 1.0 - x_min
        center_y = float(source.y) - 1.0 - y_min
        semimajor = max(4.0, axis_multiplier * float(source.a))
        semiminor = max(4.0, axis_multiplier * float(source.b))
        extent = int(np.ceil(max(semimajor, semiminor))) + 1
        left = max(0, int(np.floor(center_x)) - extent)
        right = min(width, int(np.floor(center_x)) + extent + 2)
        bottom = max(0, int(np.floor(center_y)) - extent)
        top = min(height, int(np.floor(center_y)) + extent + 2)
        if left >= right or bottom >= top:
            continue
        yy, xx = np.indices((top - bottom, right - left), dtype=float)
        dx = xx + left - center_x
        dy = yy + bottom - center_y
        theta = np.deg2rad(float(source.theta))
        major_coordinate = np.cos(theta) * dx + np.sin(theta) * dy
        minor_coordinate = -np.sin(theta) * dx + np.cos(theta) * dy
        ellipse = (
            (major_coordinate / semimajor) ** 2 + (minor_coordinate / semiminor) ** 2
            <= 1.0
        )
        mask[bottom:top, left:right] |= ellipse
    return mask


def _refine_center(
    image: np.ndarray,
    weight: np.ndarray,
    source_mask: np.ndarray,
    expected_x: float,
    expected_y: float,
    pixel_scale: float,
    background_annulus: list[float],
) -> tuple[float, float, float, float]:
    yy, xx = np.indices(image.shape, dtype=float)
    radius_arcsec = np.hypot(xx - expected_x, yy - expected_y) * pixel_scale
    valid = np.isfinite(image) & np.isfinite(weight) & (weight > 0) & ~source_mask
    background_selection = (
        valid
        & (radius_arcsec >= background_annulus[0])
        & (radius_arcsec <= background_annulus[1])
    )
    background = float(np.median(image[background_selection]))
    centroid_selection = valid & (radius_arcsec <= 1.5)
    signal = np.clip(image - background, 0.0, None)
    flux = float(signal[centroid_selection].sum())
    if flux <= 0:
        raise ValueError("Central F125W centroid flux is non-positive")
    center_x = float((signal[centroid_selection] * xx[centroid_selection]).sum() / flux)
    center_y = float((signal[centroid_selection] * yy[centroid_selection]).sum() / flux)
    offset = float(np.hypot(center_x - expected_x, center_y - expected_y) * pixel_scale)
    return center_x, center_y, offset, background


def _profile_edges(pixel_scale: float, fit_radius: float) -> np.ndarray:
    return np.concatenate([[0.0], np.geomspace(pixel_scale, fit_radius, 60)])


def _extract_sector_profile(
    image: np.ndarray,
    weight: np.ndarray,
    source_mask: np.ndarray,
    center_x: float,
    center_y: float,
    edges: np.ndarray,
    pixel_scale: float,
    sectors: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    yy, xx = np.indices(image.shape, dtype=float)
    dx = xx - center_x
    dy = yy - center_y
    radius = np.hypot(dx, dy) * pixel_scale
    angle = np.mod(np.arctan2(dy, dx), 2.0 * np.pi)
    radial_index = np.searchsorted(edges, radius, side="right") - 1
    sector_index = np.floor(angle / (2.0 * np.pi) * sectors).astype(int)
    sector_index = np.clip(sector_index, 0, sectors - 1)
    valid = (
        np.isfinite(image)
        & np.isfinite(weight)
        & (weight > 0)
        & ~source_mask
        & (radial_index >= 0)
        & (radial_index < len(edges) - 1)
    )
    total = np.bincount(
        radial_index[(radial_index >= 0) & (radial_index < len(edges) - 1)].ravel(),
        minlength=len(edges) - 1,
    ).astype(float)
    retained = np.bincount(radial_index[valid].ravel(), minlength=len(edges) - 1).astype(float)
    unmasked_fraction = np.divide(retained, total, out=np.zeros_like(retained), where=total > 0)

    sector_profile = np.full((sectors, len(edges) - 1), np.nan)
    sector_noise_variance = np.full_like(sector_profile, np.nan)
    for sector in range(sectors):
        for radial_bin in range(len(edges) - 1):
            selection = valid & (sector_index == sector) & (radial_index == radial_bin)
            if not np.any(selection):
                continue
            weights = weight[selection]
            weight_sum = float(weights.sum())
            sector_profile[sector, radial_bin] = float(
                np.sum(weights * image[selection]) / weight_sum
            )
            sector_noise_variance[sector, radial_bin] = 1.15**2 / weight_sum
    finite_sectors = np.sum(np.isfinite(sector_profile), axis=0)
    return sector_profile, sector_noise_variance, unmasked_fraction, finite_sectors


def _joint_bootstrap_covariance(
    sector_profiles: list[np.ndarray],
    sector_noise: list[np.ndarray],
    usable: np.ndarray,
    replicates: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
    rng = np.random.default_rng(seed)
    sectors = sector_profiles[0].shape[0]
    draws = []
    for _ in range(replicates):
        indices = rng.integers(0, sectors, size=sectors)
        values = []
        for profile in sector_profiles:
            values.append(np.nanmean(profile[indices][:, usable], axis=0))
        draws.append(np.concatenate(values))
    bootstrap = np.asarray(draws)
    covariance = np.cov(bootstrap, rowvar=False, ddof=1)

    central_profiles = []
    diagonal_noise = []
    for profile, noise in zip(sector_profiles, sector_noise, strict=True):
        central_profiles.append(np.nanmean(profile[:, usable], axis=0))
        valid_count = np.sum(np.isfinite(noise[:, usable]), axis=0)
        summed = np.nansum(noise[:, usable], axis=0)
        diagonal_noise.append(
            np.divide(
                summed,
                valid_count**2,
                out=np.full_like(summed, np.nan),
                where=valid_count > 0,
            )
        )
    joint_noise = np.concatenate(diagonal_noise)
    covariance += np.diag(joint_noise)
    return covariance, np.concatenate(central_profiles), central_profiles


def _write_covariance(path: Path, covariance: np.ndarray, labels: list[str]) -> None:
    frame = pd.DataFrame(covariance, columns=labels)
    frame.insert(0, "row", labels)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def _plot_diagnostic(
    path: Path,
    f125: np.ndarray,
    source_mask: np.ndarray,
    center_x: float,
    center_y: float,
    pixel_scale: float,
    profile: pd.DataFrame,
    backgrounds: dict[str, float],
) -> None:
    display_radius_pixels = int(round(30.0 / pixel_scale))
    x0 = int(round(center_x))
    y0 = int(round(center_y))
    cut = np.asarray(
        f125[
            y0 - display_radius_pixels : y0 + display_radius_pixels + 1,
            x0 - display_radius_pixels : x0 + display_radius_pixels + 1,
        ],
        dtype=float,
    )
    cut_mask = source_mask[
        y0 - display_radius_pixels : y0 + display_radius_pixels + 1,
        x0 - display_radius_pixels : x0 + display_radius_pixels + 1,
    ]
    display = np.where(cut_mask, np.nan, cut - backgrounds["F125W"])
    scale = np.nanpercentile(np.clip(display, 0.0, None), 99.5)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4))
    axes[0].imshow(
        np.arcsinh(np.clip(display, 0.0, None) / max(scale, 1e-12) * 30.0),
        origin="lower",
        cmap="gray",
    )
    axes[0].set_title("masked F125W, 60-arcsec field")
    axes[0].set_xticks([])
    axes[0].set_yticks([])

    usable = profile["profile_gate_usable"].astype(bool)
    for label, color in (("F125W", "tab:blue"), ("F814W", "tab:orange")):
        signal = profile[f"{label.lower()}_surface_brightness"] - backgrounds[label]
        error = profile[f"{label.lower()}_surface_brightness_error"]
        positive = usable & (signal > 0) & np.isfinite(error)
        axes[1].errorbar(
            profile.loc[positive, "radius_mid_arcsec"],
            signal[positive],
            yerr=error[positive],
            fmt=".",
            ms=3,
            color=color,
            label=label,
        )
    axes[1].set(xscale="log", yscale="log", xlabel="radius (arcsec)", ylabel="sky-subtracted image units")
    axes[1].grid(alpha=0.25)
    axes[1].legend()

    axes[2].plot(
        profile["radius_mid_arcsec"],
        profile["minimum_unmasked_fraction"],
        color="tab:purple",
        label="unmasked fraction",
    )
    axes[2].plot(
        profile["radius_mid_arcsec"],
        profile["minimum_finite_sectors"] / 12.0,
        color="tab:green",
        label="finite sector fraction",
    )
    axes[2].axhline(0.5, color="black", linestyle="--", linewidth=1)
    axes[2].set(xscale="log", ylim=(0, 1.05), xlabel="radius (arcsec)", ylabel="coverage fraction")
    axes[2].grid(alpha=0.25)
    axes[2].legend()
    fig.suptitle("RX J2129 frozen HST profile extraction")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def extract(config_path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config["authorization"]["gravity_response_fit"]:
        raise ValueError("HST profile protocol must not authorize gravity fitting")
    psf_report = json.loads(_resolve(config["inputs"]["empirical_psf_report"]).read_text())
    if not psf_report["both_filters_gate_pass"]:
        raise ValueError("Empirical PSF gate did not pass")

    geometry = config["geometry"]
    pixel_scale = geometry["pixel_scale_arcsec"]
    cutout_radius = max(max(pair) for pair in geometry["background_annulus_sensitivity_arcsec"])
    half_size = int(np.ceil(cutout_radius / pixel_scale)) + 2
    paths = {
        "F125W": (
            _resolve(config["inputs"]["f125w_science"]),
            _resolve(config["inputs"]["f125w_weight"]),
        ),
        "F814W": (
            _resolve(config["inputs"]["f814w_science"]),
            _resolve(config["inputs"]["f814w_weight"]),
        ),
    }
    images: dict[str, np.ndarray] = {}
    weights: dict[str, np.ndarray] = {}
    units: dict[str, str] = {}
    expected_global: tuple[float, float] | None = None
    x_min = y_min = 0
    for label, (science_path, weight_path) in paths.items():
        with fits.open(science_path, memmap=True) as science_hdul, fits.open(
            weight_path, memmap=True
        ) as weight_hdul:
            global_center = _center_pixel(science_hdul[0].header, geometry)
            if expected_global is None:
                expected_global = global_center
                x_center_integer = int(round(global_center[0]))
                y_center_integer = int(round(global_center[1]))
                x_min = x_center_integer - half_size
                y_min = y_center_integer - half_size
            else:
                if np.hypot(
                    global_center[0] - expected_global[0],
                    global_center[1] - expected_global[1],
                ) > 0.1:
                    raise ValueError("F125W and F814W mosaic WCS centers disagree")
            x_max = x_min + 2 * half_size + 1
            y_max = y_min + 2 * half_size + 1
            images[label] = np.asarray(
                science_hdul[0].data[y_min:y_max, x_min:x_max], dtype=float
            )
            weights[label] = np.asarray(
                weight_hdul[0].data[y_min:y_max, x_min:x_max], dtype=float
            )
            units[label] = science_hdul[0].header.get("BUNIT", "unknown")
    assert expected_global is not None
    expected_x = expected_global[0] - x_min
    expected_y = expected_global[1] - y_min

    catalog = _read_catalog(_resolve(config["inputs"]["catalog"]))
    source_mask = _make_source_mask(
        images["F125W"].shape,
        catalog,
        x_min,
        y_min,
        config["mask"]["bcg_id"],
        3.0,
    )
    center_x, center_y, center_offset, f125_background = _refine_center(
        images["F125W"],
        weights["F125W"],
        source_mask,
        expected_x,
        expected_y,
        pixel_scale,
        geometry["background_annulus_arcsec"],
    )
    yy, xx = np.indices(images["F814W"].shape, dtype=float)
    radius = np.hypot(xx - center_x, yy - center_y) * pixel_scale
    f814_valid = (
        np.isfinite(images["F814W"])
        & np.isfinite(weights["F814W"])
        & (weights["F814W"] > 0)
        & ~source_mask
        & (radius >= geometry["background_annulus_arcsec"][0])
        & (radius <= geometry["background_annulus_arcsec"][1])
    )
    backgrounds = {
        "F125W": f125_background,
        "F814W": float(np.median(images["F814W"][f814_valid])),
    }

    edges = _profile_edges(pixel_scale, geometry["fit_radius_arcsec"])
    sector_profiles = []
    sector_noise = []
    fractions = []
    finite_counts = []
    for label in ("F125W", "F814W"):
        profile, noise, fraction, finite = _extract_sector_profile(
            images[label],
            weights[label],
            source_mask,
            center_x,
            center_y,
            edges,
            pixel_scale,
            config["nonparametric_profile"]["azimuthal_sectors"],
        )
        sector_profiles.append(profile)
        sector_noise.append(noise)
        fractions.append(fraction)
        finite_counts.append(finite)
    fractions_array = np.stack(fractions)
    finite_array = np.stack(finite_counts)
    usable = (
        np.min(fractions_array, axis=0)
        >= config["mask"]["minimum_unmasked_fraction_per_radial_bin"]
    ) & (
        np.min(finite_array, axis=0)
        >= config["nonparametric_profile"]["minimum_finite_sectors_per_bin"]
    )
    covariance, _, central = _joint_bootstrap_covariance(
        sector_profiles,
        sector_noise,
        usable,
        config["nonparametric_profile"]["bootstrap_replicates"],
        config["nonparametric_profile"]["random_seed"],
    )
    errors = np.sqrt(np.diag(covariance))
    usable_indices = np.flatnonzero(usable)
    labels = [f"F125W_bin_{index + 1}" for index in usable_indices] + [
        f"F814W_bin_{index + 1}" for index in usable_indices
    ]

    profile_frame = pd.DataFrame(
        {
            "radial_bin": np.arange(1, len(edges)),
            "radius_min_arcsec": edges[:-1],
            "radius_max_arcsec": edges[1:],
            "radius_mid_arcsec": 0.5 * (edges[:-1] + edges[1:]),
            "f125w_surface_brightness": np.nanmean(sector_profiles[0], axis=0),
            "f814w_surface_brightness": np.nanmean(sector_profiles[1], axis=0),
            "f125w_unmasked_fraction": fractions_array[0],
            "f814w_unmasked_fraction": fractions_array[1],
            "minimum_unmasked_fraction": np.min(fractions_array, axis=0),
            "f125w_finite_sectors": finite_array[0],
            "f814w_finite_sectors": finite_array[1],
            "minimum_finite_sectors": np.min(finite_array, axis=0),
            "profile_gate_usable": usable,
        }
    )
    profile_frame["f125w_surface_brightness_error"] = np.nan
    profile_frame["f814w_surface_brightness_error"] = np.nan
    profile_frame.loc[usable, "f125w_surface_brightness_error"] = errors[: len(usable_indices)]
    profile_frame.loc[usable, "f814w_surface_brightness_error"] = errors[len(usable_indices) :]

    outputs = config["outputs"]
    profile_path = _resolve(outputs["nonparametric_profile"])
    profile_path.parent.mkdir(parents=True, exist_ok=True)
    profile_frame.to_csv(profile_path, index=False)
    _write_covariance(_resolve(outputs["profile_covariance"]), covariance, labels)
    _plot_diagnostic(
        _resolve(outputs["profile_diagnostic"]),
        images["F125W"],
        source_mask,
        center_x,
        center_y,
        pixel_scale,
        profile_frame,
        backgrounds,
    )

    eigenvalues = np.linalg.eigvalsh(covariance)
    report = {
        "protocol_version": config["protocol_version"],
        "status": "nonparametric_profile_extraction_complete_decomposition_pending",
        "gravity_or_lens_residual_read": False,
        "empirical_psf_gate_pass": True,
        "image_units": units,
        "expected_center_cutout_pixels": [expected_x, expected_y],
        "refined_center_cutout_pixels": [center_x, center_y],
        "refined_center_offset_arcsec": center_offset,
        "center_gate_pass": bool(
            center_offset <= geometry["maximum_fitted_centroid_offset_arcsec"]
        ),
        "background_image_units": backgrounds,
        "catalog_sources_masked": int(len(catalog) - 1),
        "radial_bins_total": int(len(edges) - 1),
        "radial_bins_usable": int(usable.sum()),
        "minimum_required_usable_bins": config["advance_thresholds"][
            "finite_nonparametric_profile_bins_minimum"
        ],
        "profile_coverage_gate_pass": bool(
            usable.sum()
            >= config["advance_thresholds"]["finite_nonparametric_profile_bins_minimum"]
        ),
        "joint_covariance_shape": list(covariance.shape),
        "joint_covariance_minimum_eigenvalue": float(eigenvalues.min()),
        "joint_covariance_positive_semidefinite": bool(eigenvalues.min() >= -1e-30),
        "profile_extraction_gate_pass": bool(
            center_offset <= geometry["maximum_fitted_centroid_offset_arcsec"]
            and usable.sum()
            >= config["advance_thresholds"]["finite_nonparametric_profile_bins_minimum"]
            and eigenvalues.min() >= -1e-30
        ),
        "component_identifiability_test_complete": False,
        "stellar_mass_mapping_complete": False,
        "strict_baryonic_component_gate_pass": False,
        "strict_r1_ready": False,
        "outputs": {
            "profile": outputs["nonparametric_profile"],
            "covariance": outputs["profile_covariance"],
            "diagnostic": outputs["profile_diagnostic"],
        },
        "next_action": "Fit the frozen one- and two-component PSF-convolved models with radial cross-validation and all mask/background/leave-one-star-out variants; do not inspect gravity residuals.",
    }
    report_path = _resolve(outputs["profile_extraction_report"])
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    arguments = parser.parse_args()
    print(json.dumps(extract(arguments.config), indent=2))


if __name__ == "__main__":
    main()
