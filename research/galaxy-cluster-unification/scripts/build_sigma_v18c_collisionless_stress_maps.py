#!/usr/bin/env python3
"""Build target-blind adaptive collisionless-member-stress maps."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "sigma_v18c_collisionless_stress_maps.json"
DEFAULT_OUTPUT = ROOT / "results" / "sigma_v18c_collisionless_stress_maps"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def member_arrays(
    cluster: dict[str, Any], selection: dict[str, Any], weight: dict[str, Any]
) -> dict[str, np.ndarray]:
    rows = read_csv(ROOT / cluster["member_catalog"])
    quality = []
    for row in rows:
        if cluster["quality_column"] is not None and float(
            row[cluster["quality_column"]]
        ) < float(cluster["minimum_quality"]):
            continue
        quality.append(row)
    median_redshift = float(
        statistics.median(float(row[cluster["redshift_column"]]) for row in quality)
    )
    values: list[tuple[float, float, float, float]] = []
    for row in quality:
        flux_or_magnitude = float(row[cluster["photometry_column"]])
        if cluster["photometry_kind"] == "cgs_fnu":
            if flux_or_magnitude <= 0.0:
                continue
            magnitude = -2.5 * math.log10(flux_or_magnitude) - 48.6
        else:
            magnitude = flux_or_magnitude
        absolute = (
            magnitude
            - float(cluster["planck18_distance_modulus"])
            + 2.5 * math.log10(1.0 + float(cluster["redshift"]))
        )
        mass = float(weight["mass_to_light_solar"]) * 10.0 ** (
            -0.4 * (absolute - float(weight["solar_absolute_ab_magnitude"]))
        )
        dec = float(row[cluster["dec_column"]])
        east_arcsec = (
            (float(row[cluster["ra_column"]]) - float(cluster["center_ra_deg"]))
            * math.cos(math.radians(0.5 * (dec + float(cluster["center_dec_deg"]))))
            * 3600.0
        )
        north_arcsec = (dec - float(cluster["center_dec_deg"])) * 3600.0
        x_kpc = east_arcsec * float(cluster["planck18_kpc_per_arcsec"])
        y_kpc = north_arcsec * float(cluster["planck18_kpc_per_arcsec"])
        velocity = float(selection["speed_of_light_km_s"]) * (
            float(row[cluster["redshift_column"]]) - median_redshift
        ) / (1.0 + median_redshift)
        if math.hypot(x_kpc, y_kpc) > float(selection["projected_aperture_kpc"]):
            continue
        if abs(velocity) > float(
            selection["maximum_absolute_rest_frame_velocity_km_s"]
        ):
            continue
        values.append((x_kpc, y_kpc, velocity, mass))
    array = np.asarray(values, dtype=float)
    return {
        "x_kpc": array[:, 0],
        "y_kpc": array[:, 1],
        "velocity_km_s": array[:, 2],
        "stellar_mass_msun": array[:, 3],
        "median_redshift": np.asarray(median_redshift),
    }


def adaptive_bandwidths(x: np.ndarray, y: np.ndarray, neighbor_rank: int) -> np.ndarray:
    separation = np.hypot(x[:, None] - x[None, :], y[:, None] - y[None, :])
    ordered = np.sort(separation, axis=1)
    if neighbor_rank >= ordered.shape[1]:
        raise RuntimeError("neighbor rank exceeds selected member count")
    sigma = 0.5 * ordered[:, neighbor_rank]
    if not np.all(np.isfinite(sigma) & (sigma > 0.0)):
        raise RuntimeError("adaptive kernel has nonpositive bandwidth")
    return sigma


def stress_map(
    axis: np.ndarray,
    members: dict[str, np.ndarray],
    neighbor_rank: int,
    critical_surface_density: float,
    speed_of_light: float,
) -> dict[str, np.ndarray]:
    grid_x, grid_y = np.meshgrid(axis, axis, indexing="xy")
    x = members["x_kpc"]
    y = members["y_kpc"]
    velocity = members["velocity_km_s"]
    mass = members["stellar_mass_msun"]
    sigma = adaptive_bandwidths(x, y, neighbor_rank)
    density = np.zeros_like(grid_x)
    first = np.zeros_like(grid_x)
    second = np.zeros_like(grid_x)
    for px, py, pv, pm, ps in zip(x, y, velocity, mass, sigma, strict=True):
        kernel = np.exp(-0.5 * ((grid_x - px) ** 2 + (grid_y - py) ** 2) / ps**2)
        kernel /= 2.0 * math.pi * ps**2
        weighted = pm * kernel
        density += weighted
        first += weighted * pv
        second += weighted * pv**2
    local_mean = np.divide(first, density, out=np.zeros_like(first), where=density > 0.0)
    variance = np.divide(second, density, out=np.zeros_like(second), where=density > 0.0)
    variance = np.maximum(variance - local_mean**2, 0.0)
    q_member = density * variance / speed_of_light**2 / critical_surface_density
    return {
        "q_member": q_member,
        "member_kappa": density / critical_surface_density,
        "local_mean_velocity_km_s": local_mean,
        "local_velocity_dispersion_km_s": np.sqrt(variance),
        "bandwidth_kpc": sigma,
    }


def enclosed_radii(axis: np.ndarray, field: np.ndarray) -> dict[str, float]:
    grid_x, grid_y = np.meshgrid(axis, axis, indexing="xy")
    total = float(np.sum(field))
    if total <= 0.0 or not np.isfinite(total):
        raise RuntimeError("stress source has no finite positive weight")
    center_x = float(np.sum(grid_x * field) / total)
    center_y = float(np.sum(grid_y * field) / total)
    radius = np.hypot(grid_x - center_x, grid_y - center_y).ravel()
    weight = field.ravel()
    order = np.argsort(radius)
    cumulative = np.cumsum(weight[order]) / total
    result: dict[str, float] = {}
    for quantile in (0.5, 0.8):
        index = int(np.searchsorted(cumulative, quantile, side="left"))
        result[f"R{int(quantile * 100)}_kpc"] = float(radius[order[index]])
    result["centroid_east_kpc"] = center_x
    result["centroid_north_kpc"] = center_y
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if not config["status"].startswith("frozen after v18B readiness passed"):
        raise RuntimeError("v18C source protocol is not frozen")
    if config["authorization"]["lensing_target_opened"]:
        raise RuntimeError("v18C source construction cannot open a lensing target")
    readiness_config_path = ROOT / config["parents"]["readiness_config"]
    readiness_report_path = ROOT / config["parents"]["readiness_report"]
    if sha256(readiness_config_path) != config["parents"]["readiness_config_sha256"]:
        raise RuntimeError("v18B readiness config changed")
    if sha256(readiness_report_path) != config["parents"]["readiness_report_sha256"]:
        raise RuntimeError("v18B readiness report changed")
    readiness = json.loads(readiness_report_path.read_text(encoding="utf-8"))
    if readiness["status"] != config["authorization"]["required_readiness_status"]:
        raise RuntimeError("v18B readiness did not complete")
    if not readiness["source_construction_authorized"]:
        raise RuntimeError("v18B did not authorize source construction")
    parent = json.loads(readiness_config_path.read_text(encoding="utf-8"))

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    grid = config["grid"]
    axis = np.linspace(
        -float(grid["half_width_kpc"]),
        float(grid["half_width_kpc"]),
        int(grid["points_per_axis"]),
    )
    ranks = [int(value) for value in config["adaptive_kernel"]["declared_sensitivity_neighbor_ranks"]]
    primary_rank = int(config["adaptive_kernel"]["primary_neighbor_rank"])
    if primary_rank not in ranks:
        raise RuntimeError("primary neighbor rank is absent from the frozen sensitivity set")

    cluster_reports = []
    figure, axes = plt.subplots(1, len(parent["clusters"]), figsize=(11, 4.5), constrained_layout=True)
    for plot_axis, (name, cluster) in zip(axes, parent["clusters"].items(), strict=True):
        members = member_arrays(cluster, parent["universal_selection"], parent["universal_stellar_weight"])
        products: dict[str, np.ndarray] = {
            "axis_kpc": axis,
            "member_x_kpc": members["x_kpc"],
            "member_y_kpc": members["y_kpc"],
            "member_velocity_km_s": members["velocity_km_s"],
            "member_stellar_mass_msun": members["stellar_mass_msun"],
        }
        rank_reports = []
        primary = None
        for rank in ranks:
            mapped = stress_map(
                axis,
                members,
                rank,
                float(config["lensing_geometry"]["critical_surface_density_msun_kpc2"][name]),
                float(parent["universal_selection"]["speed_of_light_km_s"]),
            )
            for key, value in mapped.items():
                products[f"{key}_k{rank}"] = value
            rank_reports.append(
                {
                    "neighbor_rank": rank,
                    "bandwidth_kpc": {
                        "minimum": float(np.min(mapped["bandwidth_kpc"])),
                        "median": float(np.median(mapped["bandwidth_kpc"])),
                        "maximum": float(np.max(mapped["bandwidth_kpc"])),
                    },
                    "source_extent": enclosed_radii(axis, mapped["q_member"]),
                    "q_member_maximum": float(np.max(mapped["q_member"])),
                }
            )
            if rank == primary_rank:
                primary = mapped
        if primary is None:
            raise RuntimeError("primary member-stress map was not constructed")
        product_path = output / f"{name}_collisionless_stress_features.npz"
        np.savez_compressed(product_path, **products)
        image = plot_axis.imshow(
            primary["q_member"],
            origin="lower",
            extent=(axis[0], axis[-1], axis[0], axis[-1]),
            cmap="magma",
        )
        plot_axis.scatter(members["x_kpc"], members["y_kpc"], s=2, c="cyan", alpha=0.35)
        plot_axis.set(title=f"{name}: member random stress", xlabel="east [kpc]", ylabel="north [kpc]")
        plot_axis.set_xlim(axis[0], axis[-1])
        plot_axis.set_ylim(axis[0], axis[-1])
        figure.colorbar(image, ax=plot_axis, label=r"$q_{member}$")
        cluster_reports.append(
            {
                "cluster": name,
                "selected_members": int(members["x_kpc"].size),
                "product": str(product_path.relative_to(ROOT)).replace("\\", "/"),
                "product_bytes": product_path.stat().st_size,
                "product_sha256": sha256(product_path),
                "rank_diagnostics": rank_reports,
            }
        )

    figure_path = output / "collisionless_stress_maps.png"
    figure.savefig(figure_path, dpi=180)
    plt.close(figure)
    report = {
        "status": "both_target_blind_collisionless_stress_maps_constructed_and_frozen",
        "protocol_version": config["protocol_version"],
        "generated_utc": datetime.now(UTC).isoformat(),
        "input_hashes": {
            "config": sha256(config_path),
            "readiness_config": sha256(readiness_config_path),
            "readiness_report": sha256(readiness_report_path),
        },
        "clusters": cluster_reports,
        "figure": str(figure_path.relative_to(ROOT)).replace("\\", "/"),
        "figure_sha256": sha256(figure_path),
        "primary_neighbor_rank": primary_rank,
        "fixed_physical_length": False,
        "per_cluster_bandwidth_amplitude_or_orientation": False,
        "inverse_coefficient_fit": False,
        "lensing_target_opened": False,
        "holdout_opened": False,
        "source_maps_frozen": True,
    }
    (output / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
