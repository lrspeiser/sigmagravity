#!/usr/bin/env python3
"""Run the frozen replacement-pair collisionless-stress transfer test."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import gaussian_filter, map_coordinates

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from build_sigma_v18c_collisionless_stress_maps import member_arrays, stress_map

from voidscreen.sigma_covariant_feature_inference import (
    EquivariantDataset,
    MetricFeature,
    convergence_to_shear,
    fit_equivariant_ridge_features,
    predict_residual,
    score_prediction,
)

DEFAULT_CONFIG = ROOT / "configs" / "sigma_v18d_collisionless_stress_transfer.json"
DEFAULT_OUTPUT = ROOT / "results" / "sigma_v18d_collisionless_stress_transfer"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def triplet(convergence: np.ndarray, padding_factor: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    shear_1, shear_2 = convergence_to_shear(convergence, padding_factor=padding_factor)
    return convergence, shear_1, shear_2


def resample_square(source_axis: np.ndarray, values: np.ndarray, target_axis: np.ndarray) -> np.ndarray:
    interpolator = RegularGridInterpolator(
        (source_axis, source_axis), values, bounds_error=False, fill_value=0.0
    )
    east, north = np.meshgrid(target_axis, target_axis, indexing="xy")
    points = np.column_stack([north.ravel(), east.ravel()])
    return interpolator(points).reshape(east.shape)


def load_target(cluster: dict[str, Any], axis: np.ndarray, protocol: dict[str, Any]) -> np.ndarray:
    east, north = np.meshgrid(axis, axis, indexing="xy")
    center_dec = float(cluster["center_dec_deg"])
    ra = float(cluster["center_ra_deg"]) + east / (
        float(cluster["kpc_per_arcsec"])
        * 3600.0
        * math.cos(math.radians(center_dec))
    )
    dec = center_dec + north / (float(cluster["kpc_per_arcsec"]) * 3600.0)
    path = ROOT / cluster["target_kappa_dls_over_ds_1"]
    with fits.open(path, memmap=True) as hdus:
        data = np.asarray(hdus[0].data, dtype=float)
        wcs = WCS(hdus[0].header)
        pixel_x, pixel_y = wcs.world_to_pixel_values(ra, dec)
        sampled = map_coordinates(
            data,
            np.vstack([pixel_y.ravel(), pixel_x.ravel()]),
            order=1,
            mode="constant",
            cval=np.nan,
            prefilter=False,
        ).reshape(east.shape)
    mask = np.hypot(east, north) <= float(protocol["analysis_radius_kpc"])
    if np.count_nonzero(mask & np.isfinite(sampled)) != np.count_nonzero(mask):
        raise RuntimeError("target does not fully cover the frozen analysis aperture")
    scaled = sampled * float(cluster["dls_over_ds_at_z2"])
    finite_values = scaled[mask & np.isfinite(scaled)]
    cap = float(
        np.percentile(
            finite_values,
            float(protocol["target_winsor_percentile_inside_analysis_mask"]),
        )
    )
    scaled = np.minimum(np.nan_to_num(scaled, nan=0.0), cap)
    spacing = float(axis[1] - axis[0])
    return gaussian_filter(
        scaled,
        float(protocol["target_gaussian_smoothing_kpc"]) / spacing,
        mode="constant",
    )


def load_baryon(cluster: dict[str, Any], axis: np.ndarray, protocol: dict[str, Any]) -> np.ndarray:
    spacing = float(axis[1] - axis[0])
    if cluster["baryon_kind"] == "registered_surface_density":
        with np.load(ROOT / cluster["baryon_input"]) as product:
            surface = resample_square(
                np.asarray(product["axis_kpc"], dtype=float),
                np.asarray(product["baryon_surface_density_msun_kpc2"], dtype=float),
                axis,
            )
    elif cluster["baryon_kind"] == "compressed_points":
        with (ROOT / cluster["baryon_input"]).open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        edges = np.r_[axis - 0.5 * spacing, axis[-1] + 0.5 * spacing]
        east = np.asarray(
            [-float(row["x_arcsec"]) * float(cluster["kpc_per_arcsec"]) for row in rows]
        )
        north = np.asarray(
            [float(row["y_arcsec"]) * float(cluster["kpc_per_arcsec"]) for row in rows]
        )
        mass = np.asarray([float(row["mass_msun"]) for row in rows])
        cell_mass, _, _ = np.histogram2d(north, east, bins=(edges, edges), weights=mass)
        surface = cell_mass / spacing**2
    else:
        raise RuntimeError("unknown baryon input representation")
    return gaussian_filter(
        surface,
        float(protocol["baryon_gaussian_smoothing_kpc"]) / spacing,
        mode="constant",
    )


def field_extent(
    channels: tuple[np.ndarray, np.ndarray, np.ndarray], mask: np.ndarray, axis: np.ndarray
) -> dict[str, Any]:
    energy = sum(np.asarray(channel) ** 2 for channel in channels)
    energy = np.where(mask & np.isfinite(energy), energy, 0.0)
    total = float(np.sum(energy))
    if total <= np.finfo(float).tiny:
        return {"valid": False, "radii_kpc": {"R50": None, "R80": None}}
    east, north = np.meshgrid(axis, axis, indexing="xy")
    center_east = float(np.sum(energy * east) / total)
    center_north = float(np.sum(energy * north) / total)
    radius = np.hypot(east - center_east, north - center_north)
    order = np.argsort(radius[mask], kind="stable")
    radii = radius[mask][order]
    cumulative = np.cumsum(energy[mask][order]) / total
    return {
        "valid": True,
        "centroid_east_kpc": center_east,
        "centroid_north_kpc": center_north,
        "radii_kpc": {
            "R50": float(radii[min(int(np.searchsorted(cumulative, 0.5)), radii.size - 1)]),
            "R80": float(radii[min(int(np.searchsorted(cumulative, 0.8)), radii.size - 1)]),
        },
    }


def extent_comparison(
    required: tuple[np.ndarray, np.ndarray, np.ndarray],
    predicted: tuple[np.ndarray, np.ndarray, np.ndarray],
    mask: np.ndarray,
    axis: np.ndarray,
) -> dict[str, Any]:
    required_extent = field_extent(required, mask, axis)
    predicted_extent = field_extent(predicted, mask, axis)
    valid = required_extent["valid"] and predicted_extent["valid"]
    errors = {}
    if valid:
        for key, required_radius in required_extent["radii_kpc"].items():
            errors[key] = abs(predicted_extent["radii_kpc"][key] - required_radius) / required_radius
    return {
        "valid": valid,
        "required": required_extent,
        "predicted": predicted_extent,
        "fractional_radius_errors": errors,
        "maximum_fractional_radius_error": max(errors.values()) if errors else None,
    }


def build_datasets(config: dict[str, Any], grid_points: int) -> dict[str, EquivariantDataset]:
    protocol = config["map_protocol"]
    axis = np.linspace(
        -float(protocol["half_width_kpc"]),
        float(protocol["half_width_kpc"]),
        grid_points,
    )
    east, north = np.meshgrid(axis, axis, indexing="xy")
    mask = np.hypot(east, north) <= float(protocol["analysis_radius_kpc"])
    source_config = json.loads((ROOT / config["parents"]["source_config"]).read_text(encoding="utf-8"))
    readiness_config = json.loads(
        (ROOT / source_config["parents"]["readiness_config"]).read_text(encoding="utf-8")
    )
    datasets = {}
    for name, cluster in config["clusters"].items():
        parent_cluster = readiness_config["clusters"][name]
        members = member_arrays(
            parent_cluster,
            readiness_config["universal_selection"],
            readiness_config["universal_stellar_weight"],
        )
        mapped = stress_map(
            axis,
            members,
            int(source_config["adaptive_kernel"]["primary_neighbor_rank"]),
            float(source_config["lensing_geometry"]["critical_surface_density_msun_kpc2"][name]),
            float(readiness_config["universal_selection"]["speed_of_light_km_s"]),
        )
        target_kappa = load_target(cluster, axis, protocol)
        baryon_surface = load_baryon(cluster, axis, protocol)
        critical = float(source_config["lensing_geometry"]["critical_surface_density_msun_kpc2"][name])
        baryon_kappa = baryon_surface / critical
        base = triplet(baryon_kappa, int(protocol["fourier_padding_factor"]))
        target = triplet(target_kappa, int(protocol["fourier_padding_factor"]))
        feature_triplet = triplet(mapped["q_member"], int(protocol["fourier_padding_factor"]))
        feature = MetricFeature(
            name="member_random_stress",
            family="scalar_scale",
            convergence=feature_triplet[0],
            shear_1=feature_triplet[1],
            shear_2=feature_triplet[2],
        )
        datasets[name] = EquivariantDataset(
            name=name,
            mask=mask,
            base=base,
            target=target,
            features={feature.name: feature},
        )
    return datasets


def symmetric_error(scores: list[dict[str, Any]]) -> float:
    return float(np.sqrt(np.mean([float(score["full_field_NRMSE"]) ** 2 for score in scores])))


def score_transfer(datasets: dict[str, EquivariantDataset]) -> dict[str, Any]:
    directions = []
    for train_name, test_name in (("MACS0416", "PLCKG287"), ("PLCKG287", "MACS0416")):
        fit = fit_equivariant_ridge_features(
            [datasets[train_name]], feature_names=["member_random_stress"], alpha=0.0
        )
        raw_coefficient = float(fit.coefficients["member_random_stress"])
        coefficient = max(raw_coefficient, 0.0)
        coefficients = {"member_random_stress": coefficient}
        test = datasets[test_name]
        score = score_prediction(test, coefficients)
        predicted = predict_residual(test, coefficients)
        required = tuple(
            target - base for target, base in zip(test.target, test.base, strict=True)
        )
        score.update(
            {
                "train_cluster": train_name,
                "test_cluster": test_name,
                "raw_coefficient": raw_coefficient,
                "coefficient": coefficient,
                "coefficient_nonnegative": raw_coefficient >= 0.0,
                "fit_condition_number": fit.standardized_condition_number,
                "halo_scale_diagnostic": extent_comparison(
                    required, predicted, test.mask, np.linspace(-350.0, 350.0, test.mask.shape[0])
                ),
            }
        )
        directions.append(score)
    baseline_scores = [score_prediction(dataset, {"member_random_stress": 0.0}) for dataset in datasets.values()]
    result = {
        "directions": directions,
        "symmetric_cross_cluster_full_field_NRMSE": symmetric_error(directions),
        "symmetric_baryon_baseline_full_field_NRMSE": symmetric_error(baseline_scores),
    }
    result["relative_improvement_over_baryon_baseline"] = 1.0 - (
        result["symmetric_cross_cluster_full_field_NRMSE"]
        / result["symmetric_baryon_baseline_full_field_NRMSE"]
    )
    coefficients = [row["coefficient"] for row in directions]
    result["directional_coefficient_log10_difference"] = (
        abs(math.log10(coefficients[0] / coefficients[1]))
        if min(coefficients) > 0.0
        else math.inf
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if not config["status"].startswith("frozen after both v18C source maps were hashed"):
        raise RuntimeError("v18D transfer protocol is not frozen")
    for key in ("source_config", "source_report"):
        path = ROOT / config["parents"][key]
        if sha256(path) != config["parents"][f"{key}_sha256"]:
            raise RuntimeError(f"frozen {key} changed")
    for name, cluster in config["clusters"].items():
        for key in ("target_kappa_dls_over_ds_1", "baryon_input", "member_stress_product"):
            hash_key = {
                "target_kappa_dls_over_ds_1": "target_kappa_sha256",
                "baryon_input": "baryon_input_sha256",
                "member_stress_product": "member_stress_product_sha256",
            }[key]
            if sha256(ROOT / cluster[key]) != cluster[hash_key]:
                raise RuntimeError(f"frozen {name} {key} changed")

    primary = score_transfer(build_datasets(config, int(config["map_protocol"]["primary_grid_points"])))
    doubled = score_transfer(build_datasets(config, int(config["map_protocol"]["doubled_grid_points"])))
    resolution_components: dict[str, float] = {
        "symmetric_full_field_NRMSE_relative_change": abs(
            doubled["symmetric_cross_cluster_full_field_NRMSE"]
            - primary["symmetric_cross_cluster_full_field_NRMSE"]
        )
        / primary["symmetric_cross_cluster_full_field_NRMSE"]
    }
    for first, second in zip(primary["directions"], doubled["directions"], strict=True):
        tag = f"{first['train_cluster']}_to_{first['test_cluster']}"
        resolution_components[f"{tag}_full_field_NRMSE_relative_change"] = abs(
            second["full_field_NRMSE"] - first["full_field_NRMSE"]
        ) / first["full_field_NRMSE"]
        resolution_components[f"{tag}_shear_alignment_absolute_change"] = abs(
            second["residual_shear_alignment_cosine"] - first["residual_shear_alignment_cosine"]
        )
        resolution_components[f"{tag}_power_closed_absolute_change"] = abs(
            second["residual_power_closed"] - first["residual_power_closed"]
        )
        for radius in ("R50", "R80"):
            old = first["halo_scale_diagnostic"]["predicted"]["radii_kpc"][radius]
            new = second["halo_scale_diagnostic"]["predicted"]["radii_kpc"][radius]
            resolution_components[f"{tag}_{radius}_relative_change"] = abs(new - old) / old
    resolution = {
        "components": resolution_components,
        "maximum_change": max(resolution_components.values()),
    }
    gates = config["gates"]
    gate_results = {
        "absolute_source_sufficiency": primary["symmetric_cross_cluster_full_field_NRMSE"]
        <= float(gates["maximum_symmetric_cross_cluster_full_field_NRMSE"]),
        "material_improvement": primary["relative_improvement_over_baryon_baseline"]
        >= float(gates["minimum_relative_improvement_over_baryon_baseline"]),
        "residual_shear_alignment_both_directions": all(
            row["residual_shear_alignment_cosine"]
            >= float(gates["minimum_residual_shear_alignment_cosine_each_direction"])
            for row in primary["directions"]
        ),
        "residual_power_both_directions": all(
            row["residual_power_closed"]
            >= float(gates["minimum_residual_power_closed_each_direction"])
            for row in primary["directions"]
        ),
        "halo_scale_both_directions": all(
            row["halo_scale_diagnostic"]["valid"]
            and row["halo_scale_diagnostic"]["maximum_fractional_radius_error"]
            <= float(gates["maximum_fractional_R50_or_R80_error_each_direction"])
            for row in primary["directions"]
        ),
        "positive_consistent_coefficient": all(
            row["coefficient_nonnegative"] for row in primary["directions"]
        )
        and primary["directional_coefficient_log10_difference"]
        <= float(gates["maximum_directional_coefficient_log10_difference"]),
        "doubled_resolution_stable": resolution["maximum_change"]
        <= float(gates["maximum_map_resolution_change"]),
    }
    gate_results["advance"] = all(gate_results.values())

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(2, 3, figsize=(12, 8), constrained_layout=True)
    datasets = build_datasets(config, int(config["map_protocol"]["primary_grid_points"]))
    for row_index, direction in enumerate(primary["directions"]):
        dataset = datasets[direction["test_cluster"]]
        predicted = predict_residual(dataset, {"member_random_stress": direction["coefficient"]})
        required = tuple(target - base for target, base in zip(dataset.target, dataset.base, strict=True))
        panels = (required[0], predicted[0], required[0] - predicted[0])
        for axis_plot, values, title in zip(axes[row_index], panels, ("required residual", "predicted residual", "difference"), strict=True):
            image = axis_plot.imshow(values, origin="lower", extent=(-350, 350, -350, 350), cmap="coolwarm")
            axis_plot.set(title=f"{direction['test_cluster']} {title}", xlim=(-200, 200), ylim=(-200, 200))
            figure.colorbar(image, ax=axis_plot)
    figure_path = output / "collisionless_stress_transfer.png"
    figure.savefig(figure_path, dpi=180)
    plt.close(figure)

    report = {
        "status": "completed Sigma v18D collisionless-stress transfer diagnostic",
        "protocol_version": config["protocol_version"],
        "generated_utc": datetime.now(UTC).isoformat(),
        "input_hashes": {
            "config": sha256(config_path),
            "source_config": sha256(ROOT / config["parents"]["source_config"]),
            "source_report": sha256(ROOT / config["parents"]["source_report"]),
        },
        "primary_result": primary,
        "doubled_resolution_result": doubled,
        "resolution_stability": resolution,
        "gate_results": gate_results,
        "decision": (
            "advance to covariant causal-state derivation"
            if gate_results["advance"]
            else "retire the tested instantaneous projected collisionless-stress source"
        ),
        "figure": str(figure_path.relative_to(ROOT)).replace("\\", "/"),
        "figure_sha256": sha256(figure_path),
        "sample_is_spent": True,
        "inverse_coefficient_is_physical_constant": False,
        "per_cluster_amplitude_scale_shear_or_orientation": False,
        "lensing_target_opened": True,
        "holdout_opened": False,
    }
    (output / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
