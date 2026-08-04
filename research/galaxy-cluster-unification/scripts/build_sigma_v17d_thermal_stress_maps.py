#!/usr/bin/env python3
"""Build frozen, target-blind v17D gas thermal-stress source maps."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
from astropy.constants import G, c, m_p
from astropy.coordinates import SkyCoord
from astropy.cosmology import Planck18
from astropy.io import fits
from astropy.wcs import WCS
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import gaussian_filter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from voidscreen.sigma_covariant_feature_inference import convergence_to_shear

DEFAULT_CONFIG = ROOT / "configs" / "sigma_v17d_thermal_stress_map.json"
DEFAULT_SPECTRAL_CONFIG = ROOT / "configs" / "sigma_v17c_spectral_temperature.json"
DEFAULT_REGIONAL_TEMPERATURES = (
    ROOT / "results" / "sigma_v17c_regional_temperatures" / "report.json"
)
DEFAULT_INTEGRATED_TEMPERATURES = (
    ROOT / "results" / "sigma_v17c_integrated_temperatures" / "report.json"
)
DEFAULT_OUTPUT = ROOT / "results" / "sigma_v17d_thermal_stress_maps"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def critical_surface_density_msun_kpc2(
    lens_redshift: float,
    source_redshift: float,
) -> float:
    lens_distance = Planck18.angular_diameter_distance(lens_redshift)
    source_distance = Planck18.angular_diameter_distance(source_redshift)
    lens_source_distance = Planck18.angular_diameter_distance_z1z2(
        lens_redshift,
        source_redshift,
    )
    value = c**2 / (4.0 * np.pi * G) * source_distance / (
        lens_distance * lens_source_distance
    )
    return float(value.to_value(u.Msun / u.kpc**2))


def energy_ratio_per_kev(mu: float) -> float:
    return float((1.0 * u.keV / (mu * m_p * c**2)).to_value(u.dimensionless_unscaled))


def resample_grid(
    source_axis: np.ndarray,
    values: np.ndarray,
    target_axis: np.ndarray,
) -> np.ndarray:
    interpolator = RegularGridInterpolator(
        (source_axis, source_axis),
        values,
        bounds_error=True,
    )
    east, north = np.meshgrid(target_axis, target_axis)
    points = np.column_stack([north.ravel(), east.ravel()])
    return interpolator(points).reshape(east.shape)


def one_metric_triplet(
    source_axis: np.ndarray,
    scalar_source: np.ndarray,
    target_axis: np.ndarray,
    padding_factor: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    shear_1, shear_2 = convergence_to_shear(
        scalar_source,
        padding_factor=padding_factor,
    )
    return (
        resample_grid(source_axis, scalar_source, target_axis),
        resample_grid(source_axis, shear_1, target_axis),
        resample_grid(source_axis, shear_2, target_axis),
    )


def assign_temperature_fields(
    bin_ids: np.ndarray,
    temperatures_by_region: dict[int, float],
    global_temperature: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not math.isfinite(global_temperature) or global_temperature <= 0:
        raise RuntimeError("integrated temperature is not finite and positive")
    if not temperatures_by_region or any(
        not math.isfinite(value) or value <= 0 for value in temperatures_by_region.values()
    ):
        raise RuntimeError("a regional best-fit temperature is not finite and positive")
    unexpected = {
        int(value) for value in np.unique(bin_ids) if value >= 0
    }.difference(temperatures_by_region)
    if unexpected:
        raise RuntimeError(f"temperature binmap contains unknown regions: {sorted(unexpected)}")
    total = np.full(bin_ids.shape, global_temperature, dtype=float)
    contrast = np.zeros(bin_ids.shape, dtype=float)
    resolved = bin_ids >= 0
    for rid, temperature in temperatures_by_region.items():
        mask = bin_ids == rid
        total[mask] = temperature
        contrast[mask] = temperature - global_temperature
    return total, contrast, resolved


def bin_ids_on_baryon_grid(
    axis_kpc: np.ndarray,
    center: SkyCoord,
    lens_redshift: float,
    binmap_path: Path,
    sampling_half_width_kpc: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    with fits.open(binmap_path, memmap=True) as hdus:
        binmap = np.asarray(hdus[0].data, dtype=int)
        wcs = WCS(hdus[0].header)
    ids = np.full((axis_kpc.size, axis_kpc.size), -1, dtype=np.int16)
    inner_indices = np.flatnonzero(np.abs(axis_kpc) <= sampling_half_width_kpc)
    east, north = np.meshgrid(axis_kpc[inner_indices], axis_kpc[inner_indices])
    kpc_per_arcsec = float(
        Planck18.kpc_proper_per_arcmin(lens_redshift).to_value(u.kpc / u.arcmin) / 60.0
    )
    offset_frame = center.skyoffset_frame()
    sky = SkyCoord(
        lon=(east / kpc_per_arcsec) * u.arcsec,
        lat=(north / kpc_per_arcsec) * u.arcsec,
        frame=offset_frame,
    ).icrs
    pixel_x, pixel_y = wcs.world_to_pixel(sky)
    ix = np.rint(pixel_x).astype(int)
    iy = np.rint(pixel_y).astype(int)
    inside = (
        np.isfinite(pixel_x)
        & np.isfinite(pixel_y)
        & (ix >= 0)
        & (ix < binmap.shape[1])
        & (iy >= 0)
        & (iy < binmap.shape[0])
    )
    sampled = np.full(east.shape, -1, dtype=np.int16)
    sampled[inside] = binmap[iy[inside], ix[inside]].astype(np.int16)
    ids[np.ix_(inner_indices, inner_indices)] = sampled
    finite_ids = sorted(int(value) for value in np.unique(ids) if value >= 0)
    return ids, {
        "binmap_shape": list(binmap.shape),
        "baryon_grid_points_sampled_per_axis": int(inner_indices.size),
        "kpc_per_arcsec": kpc_per_arcsec,
        "mapped_region_ids": finite_ids,
        "resolved_baryon_pixels": int(np.count_nonzero(ids >= 0)),
        "nearest_pixel_sampling": True,
    }


def feature_inventory(
    source_axis: np.ndarray,
    q_total: np.ndarray,
    q_contrast: np.ndarray,
    config: dict,
) -> dict[str, np.ndarray]:
    feature_config = config["feature_construction"]
    target_axis = np.linspace(
        -float(feature_config["target_half_width_kpc"]),
        float(feature_config["target_half_width_kpc"]),
        int(feature_config["target_grid_points"]),
    )
    spacing = float(source_axis[1] - source_axis[0])
    padding = int(feature_config["fourier_padding_factor"])
    arrays: dict[str, np.ndarray] = {"target_axis_kpc": target_axis}
    for family, source in (("thermal_total", q_total), ("thermal_contrast", q_contrast)):
        for raw_scale in feature_config["gaussian_scales_kpc"]:
            scale = float(raw_scale)
            smoothed = gaussian_filter(source, scale / spacing, mode="constant")
            triplet = one_metric_triplet(source_axis, smoothed, target_axis, padding)
            for channel, values in zip(
                ("convergence", "shear_1", "shear_2"), triplet, strict=True
            ):
                key = f"{family}_smooth_{scale:g}kpc_{channel}"
                if not np.isfinite(values).all():
                    raise RuntimeError(f"non-finite thermal feature: {key}")
                arrays[key] = values
    return arrays


def validate_authorization(
    config_path: Path,
    config: dict,
    spectral_config_path: Path,
    regional_path: Path,
    regional: dict,
    integrated_path: Path,
    integrated: dict,
) -> None:
    for key, relative_key in (
        ("dynamical_stress_gate_sha256", "dynamical_stress_gate"),
        ("static_spent_baseline_sha256", "static_spent_baseline"),
        ("spectral_protocol_sha256", "spectral_protocol"),
    ):
        path = ROOT / config["parents"][relative_key]
        if config["parents"][key] != sha256(path):
            raise RuntimeError(f"frozen thermal-map parent changed: {relative_key}")
    if sha256(spectral_config_path) != config["parents"]["spectral_protocol_sha256"]:
        raise RuntimeError("supplied spectral protocol is not the frozen parent")
    if regional["status"] != config["authorization"][
        "required_regional_temperature_status"
    ]:
        raise RuntimeError("both regional temperature gates have not passed")
    if regional.get("thermal_stress_construction_authorized") is not True:
        raise RuntimeError("thermal-stress construction is not authorized")
    if integrated["status"] != "both_integrated_temperature_gates_passed":
        raise RuntimeError("integrated temperature gate has not passed")
    spectral_hash = sha256(spectral_config_path)
    if regional["config_sha256"] != spectral_hash or integrated["config_sha256"] != spectral_hash:
        raise RuntimeError("spectral protocol changed before thermal-map construction")
    if regional["integrated_temperatures_report_sha256"] != sha256(integrated_path):
        raise RuntimeError("regional fit used another integrated-temperature report")
    if not config_path.is_file() or not regional_path.is_file():
        raise RuntimeError("missing thermal-stress construction input")


def build_cluster(
    cluster_name: str,
    cluster_config: dict,
    config: dict,
    regional_cluster: dict,
    integrated_cluster: dict,
    output: Path,
) -> dict[str, Any]:
    baryon_path = ROOT / cluster_config["baryon_map"]
    binmap_path = ROOT / cluster_config["temperature_binmap"]
    if sha256(baryon_path) != cluster_config["baryon_map_sha256"]:
        raise RuntimeError(f"registered baryon map changed for {cluster_name}")
    if sha256(binmap_path) != cluster_config["temperature_binmap_sha256"]:
        raise RuntimeError(f"frozen temperature binmap changed for {cluster_name}")
    with np.load(baryon_path) as data:
        axis = data["axis_kpc"].astype(float)
        gas_surface = data["gas_surface_density_msun_kpc2"].astype(float)
        lens_redshift = float(data["redshift"])
        center = SkyCoord(
            float(data["center_ra_deg"]) * u.deg,
            float(data["center_dec_deg"]) * u.deg,
        )
    map_radius = float(config["feature_construction"]["registered_baryon_map_radius_kpc"])
    if (
        axis.ndim != 1
        or gas_surface.shape != (axis.size, axis.size)
        or not np.isclose(axis[0], -map_radius)
        or not np.isclose(axis[-1], map_radius)
    ):
        raise RuntimeError(f"registered baryon map geometry changed for {cluster_name}")
    expected_regions = int(cluster_config["expected_region_count"])
    temperatures = {
        int(row["region_id"]): float(row["parameters"]["temperature_keV"])
        for row in regional_cluster["regions"]
    }
    if set(temperatures) != set(range(expected_regions)):
        raise RuntimeError(f"regional temperature inventory changed for {cluster_name}")
    global_temperature = float(integrated_cluster["parameters"]["temperature_keV"])
    bin_ids, mapping = bin_ids_on_baryon_grid(
        axis,
        center,
        lens_redshift,
        binmap_path,
        float(config["feature_construction"]["target_half_width_kpc"]) * 1.1,
    )
    if mapping["mapped_region_ids"] != list(range(expected_regions)):
        raise RuntimeError(f"not every frozen temperature bin mapped for {cluster_name}")
    temperature_total, temperature_contrast, resolved = assign_temperature_fields(
        bin_ids,
        temperatures,
        global_temperature,
    )
    sigma_critical = critical_surface_density_msun_kpc2(
        lens_redshift,
        float(config["physical_proxy"]["source_redshift"]),
    )
    gas_convergence = gas_surface / sigma_critical
    ratio_per_kev = energy_ratio_per_kev(
        float(config["physical_proxy"]["mean_molecular_weight_mu"])
    )
    q_total = gas_convergence * temperature_total * ratio_per_kev
    q_contrast = gas_convergence * temperature_contrast * ratio_per_kev
    features = feature_inventory(axis, q_total, q_contrast, config)
    arrays: dict[str, np.ndarray] = {
        "source_axis_kpc": axis,
        "gas_convergence": gas_convergence.astype(np.float32),
        "temperature_total_keV": temperature_total.astype(np.float32),
        "temperature_contrast_keV": temperature_contrast.astype(np.float32),
        "resolved_temperature_mask": resolved.astype(np.uint8),
        "temperature_bin_id": bin_ids,
        "q_total": q_total.astype(np.float32),
        "q_contrast": q_contrast.astype(np.float32),
        **features,
    }
    product = output / f"{cluster_name}_thermal_stress_features.npz"
    np.savez_compressed(product, **arrays)
    report = {
        "cluster": cluster_name,
        "baryon_map": str(baryon_path),
        "baryon_map_sha256": sha256(baryon_path),
        "temperature_binmap": str(binmap_path),
        "temperature_binmap_sha256": sha256(binmap_path),
        "lens_redshift_from_registered_baryon_map": lens_redshift,
        "center_ra_deg": float(center.ra.deg),
        "center_dec_deg": float(center.dec.deg),
        "source_redshift": float(config["physical_proxy"]["source_redshift"]),
        "critical_surface_density_msun_kpc2": sigma_critical,
        "energy_ratio_per_keV": ratio_per_kev,
        "integrated_temperature_keV": global_temperature,
        "regional_temperatures_keV": {str(key): value for key, value in temperatures.items()},
        "temperature_mapping": mapping,
        "q_total_minimum": float(np.min(q_total)),
        "q_total_maximum": float(np.max(q_total)),
        "q_contrast_minimum": float(np.min(q_contrast)),
        "q_contrast_maximum": float(np.max(q_contrast)),
        "feature_names": sorted(features),
        "feature_count_excluding_axis": len(features) - 1,
        "product": product.relative_to(ROOT).as_posix(),
        "product_bytes": product.stat().st_size,
        "product_sha256": sha256(product),
    }
    render_audit_figure(cluster_name, axis, arrays, output)
    figure = output / f"{cluster_name}_thermal_stress_source_audit.png"
    report["audit_figure"] = figure.relative_to(ROOT).as_posix()
    report["audit_figure_sha256"] = sha256(figure)
    return report


def render_audit_figure(
    cluster_name: str,
    axis: np.ndarray,
    arrays: dict[str, np.ndarray],
    output: Path,
) -> None:
    extent = [float(axis[0]), float(axis[-1]), float(axis[0]), float(axis[-1])]
    panels = (
        (arrays["temperature_total_keV"], "Assigned temperature (keV)", "inferno"),
        (arrays["temperature_contrast_keV"], "Temperature contrast (keV)", "coolwarm"),
        (arrays["q_total"], "Total thermal proxy", "magma"),
        (arrays["q_contrast"], "Contrast thermal proxy", "coolwarm"),
    )
    fig, axes = plt.subplots(2, 2, figsize=(10, 9), constrained_layout=True)
    for axis_plot, (values, title, cmap) in zip(axes.ravel(), panels, strict=True):
        image = axis_plot.imshow(values, origin="lower", extent=extent, cmap=cmap)
        axis_plot.set_xlim(-400, 400)
        axis_plot.set_ylim(-400, 400)
        axis_plot.set_title(title)
        axis_plot.set_xlabel("East (kpc)")
        axis_plot.set_ylabel("North (kpc)")
        fig.colorbar(image, ax=axis_plot, shrink=0.8)
    fig.suptitle(f"{cluster_name}: target-blind v17D thermal source audit")
    fig.savefig(output / f"{cluster_name}_thermal_stress_source_audit.png", dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--spectral-config", type=Path, default=DEFAULT_SPECTRAL_CONFIG)
    parser.add_argument(
        "--regional-temperatures", type=Path, default=DEFAULT_REGIONAL_TEMPERATURES
    )
    parser.add_argument(
        "--integrated-temperatures", type=Path, default=DEFAULT_INTEGRATED_TEMPERATURES
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    config_path = args.config.resolve()
    spectral_path = args.spectral_config.resolve()
    regional_path = args.regional_temperatures.resolve()
    integrated_path = args.integrated_temperatures.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    regional = json.loads(regional_path.read_text(encoding="utf-8"))
    integrated = json.loads(integrated_path.read_text(encoding="utf-8"))
    validate_authorization(
        config_path,
        config,
        spectral_path,
        regional_path,
        regional,
        integrated_path,
        integrated,
    )
    regional_by_cluster = {row["cluster"]: row for row in regional["clusters"]}
    integrated_by_cluster = {row["cluster"]: row for row in integrated["clusters"]}
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    clusters = []
    for cluster_name, cluster_config in config["clusters"].items():
        clusters.append(
            build_cluster(
                cluster_name,
                cluster_config,
                config,
                regional_by_cluster[cluster_name],
                integrated_by_cluster[cluster_name],
                output,
            )
        )
        print(f"{cluster_name}: frozen thermal-stress map constructed", flush=True)
    report = {
        "status": "both_target_blind_thermal_stress_maps_constructed_and_frozen",
        "protocol_version": config["protocol_version"],
        "generated_utc": datetime.now(UTC).isoformat(),
        "config_sha256": sha256(config_path),
        "spectral_config_sha256": sha256(spectral_path),
        "regional_temperatures_report_sha256": sha256(regional_path),
        "integrated_temperatures_report_sha256": sha256(integrated_path),
        "clusters": clusters,
        "source_maps_frozen": True,
        "inverse_coefficients_fit": False,
        "lensing_target_opened": False,
    }
    report_path = output / "report.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(report_path)


if __name__ == "__main__":
    main()
