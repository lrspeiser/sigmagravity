#!/usr/bin/env python3
"""Run the frozen V19I continuous cluster-member source-field gate."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from datetime import UTC, datetime
from pathlib import Path

for thread_variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[thread_variable] = "1"

import matplotlib.pyplot as plt
import numpy as np
import sigma_v19f_chandra_common as common
from numba import njit
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score

ROOT = common.ROOT
DEFAULT_CONFIG = ROOT / "configs" / "sigma_v19i_continuous_member_field.json"
DEFAULT_OUTPUT = ROOT / "results" / "sigma_v19i_continuous_member_field"


def stable_seed(base: int, cluster: str) -> int:
    payload = f"{base}:{cluster}:continuous-member-field".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big")


def load_catalog(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def physical_catalog(
    rows: list[dict[str, str]], redshift: float, kpc_per_arcsec: float
) -> dict[str, np.ndarray | float]:
    ra = np.asarray([float(row["ra_deg"]) for row in rows])
    dec = np.asarray([float(row["dec_deg"]) for row in rows])
    cz = np.asarray([float(row["heliocentric_cz_km_s"]) for row in rows])
    cz_error = np.asarray([float(row["cz_uncertainty_km_s"]) for row in rows])
    ra0 = float(np.median(ra))
    dec0 = float(np.median(dec))
    x = (ra - ra0) * math.cos(math.radians(dec0)) * 3600.0 * kpc_per_arcsec
    y = (dec - dec0) * 3600.0 * kpc_per_arcsec
    velocity = (cz - np.median(cz)) / (1.0 + redshift)
    velocity_error = cz_error / (1.0 + redshift)
    return {
        "x": x,
        "y": y,
        "cz": cz,
        "velocity": velocity,
        "velocity_error": velocity_error,
        "ra_origin_deg": ra0,
        "dec_origin_deg": dec0,
    }


def loo_bandwidth_scores(
    x: np.ndarray, y: np.ndarray, bandwidths: list[float]
) -> np.ndarray:
    count = len(x)
    if count < 2:
        return np.full(len(bandwidths), -np.inf)
    distance2 = (x[:, None] - x[None, :]) ** 2 + (y[:, None] - y[None, :]) ** 2
    np.fill_diagonal(distance2, np.inf)
    scores = []
    for bandwidth in bandwidths:
        density = np.exp(-distance2 / (2.0 * bandwidth**2)).sum(axis=1)
        density /= (count - 1) * 2.0 * math.pi * bandwidth**2
        scores.append(float(np.log(density).sum()) if np.all(density > 0.0) else -np.inf)
    return np.asarray(scores)


def select_bandwidth(
    x: np.ndarray, y: np.ndarray, bandwidths: list[float]
) -> tuple[float, np.ndarray]:
    scores = loo_bandwidth_scores(x, y, bandwidths)
    if not np.all(np.isfinite(scores)):
        raise RuntimeError("a leave-one-out bandwidth score is not finite")
    maximum = float(np.max(scores))
    tied = [
        index for index, score in enumerate(scores) if abs(float(score) - maximum) <= 1e-12
    ]
    selected = max(tied, key=lambda index: bandwidths[index])
    return float(bandwidths[selected]), scores


def evaluation_grid(
    x: np.ndarray, y: np.ndarray, bandwidth: float, spacing: float, padding: float
) -> tuple[np.ndarray, np.ndarray]:
    extra = padding * bandwidth
    x0 = math.floor((float(np.min(x)) - extra) / spacing) * spacing
    x1 = math.ceil((float(np.max(x)) + extra) / spacing) * spacing
    y0 = math.floor((float(np.min(y)) - extra) / spacing) * spacing
    y1 = math.ceil((float(np.max(y)) + extra) / spacing) * spacing
    return (
        np.arange(x0, x1 + 0.5 * spacing, spacing),
        np.arange(y0, y1 + 0.5 * spacing, spacing),
    )


def separable_kernel_fields(
    x: np.ndarray,
    y: np.ndarray,
    velocity: np.ndarray,
    velocity_error: np.ndarray,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    bandwidth: float,
) -> dict[str, np.ndarray]:
    exponent_x = np.exp(-((x_grid[:, None] - x[None, :]) ** 2) / (2.0 * bandwidth**2))
    exponent_y = np.exp(-((y_grid[:, None] - y[None, :]) ** 2) / (2.0 * bandwidth**2))
    normalization = 1.0 / (2.0 * math.pi * bandwidth**2)

    def projected(coefficient: np.ndarray) -> np.ndarray:
        return normalization * ((exponent_y * coefficient[None, :]) @ exponent_x.T)

    number_density = projected(np.ones(len(x)))
    velocity_sum = projected(velocity)
    corrected_second_sum = projected(velocity**2 - velocity_error**2)
    mean_velocity = velocity_sum / number_density
    variance = np.maximum(0.0, corrected_second_sum / number_density - mean_velocity**2)
    return {
        "number_density": number_density,
        "mean_velocity": mean_velocity,
        "intrinsic_variance": variance,
        "number_current": velocity_sum,
        "random_stress": number_density * variance,
    }


def density_grid(
    x: np.ndarray,
    y: np.ndarray,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    bandwidth: float,
) -> np.ndarray:
    exponent_x = np.exp(-((x_grid[:, None] - x[None, :]) ** 2) / (2.0 * bandwidth**2))
    exponent_y = np.exp(-((y_grid[:, None] - y[None, :]) ** 2) / (2.0 * bandwidth**2))
    return (exponent_y @ exponent_x.T) / (2.0 * math.pi * bandwidth**2)


@njit
def _root(parent: np.ndarray, value: int) -> int:
    while parent[value] != value:
        parent[value] = parent[parent[value]]
        value = parent[value]
    return value


@njit
def _merge_tree(
    density: np.ndarray, order: np.ndarray, width: int
) -> tuple[np.ndarray, np.ndarray, int]:
    size = len(density)
    parent = np.full(size, -1, dtype=np.int64)
    peak = np.full(size, -1, dtype=np.int64)
    dead_peak = np.empty(size, dtype=np.int64)
    saddle = np.empty(size, dtype=np.float64)
    deaths = 0
    neighbor_y = (-1, -1, -1, 0, 0, 1, 1, 1)
    neighbor_x = (-1, 0, 1, -1, 1, -1, 0, 1)
    height = size // width
    for value in order:
        parent[value] = value
        peak[value] = value
        row = value // width
        column = value - row * width
        for neighbor in range(8):
            yy = row + neighbor_y[neighbor]
            xx = column + neighbor_x[neighbor]
            if yy < 0 or yy >= height or xx < 0 or xx >= width:
                continue
            other = yy * width + xx
            if parent[other] < 0:
                continue
            first = _root(parent, value)
            second = _root(parent, other)
            if first == second:
                continue
            first_peak = peak[first]
            second_peak = peak[second]
            first_density = density[first_peak]
            second_density = density[second_peak]
            if first_density > second_density or (
                first_density == second_density and first_peak < second_peak
            ):
                survivor = first
                loser = second
            else:
                survivor = second
                loser = first
            dead_peak[deaths] = peak[loser]
            saddle[deaths] = density[value]
            deaths += 1
            parent[loser] = survivor
            parent[value] = survivor
            peak[survivor] = peak[survivor]
    root = _root(parent, int(order[0]))
    dead_peak[deaths] = peak[root]
    saddle[deaths] = np.min(density[density > 0.0])
    deaths += 1
    return dead_peak, saddle, deaths


def persistent_modes(
    density: np.ndarray,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    bandwidth: float,
    config: dict,
) -> list[dict[str, float | int]]:
    flat = np.ascontiguousarray(density.ravel(), dtype=np.float64)
    # Stable descending density; flattened-grid index resolves exact ties.
    order = np.argsort(-flat, kind="stable").astype(np.int64)
    peak_indices, saddles, count = _merge_tree(flat, order, density.shape[1])
    maximum = float(np.max(flat))
    minimum_ratio = float(config["minimum_peak_to_saddle_density_ratio"])
    minimum_fraction = float(config["minimum_peak_density_fraction_of_global"])
    peaks = peak_indices[:count]
    saddle_values = saddles[:count]
    peak_density_values = flat[peaks]
    ratio_values = peak_density_values / np.maximum(saddle_values, np.finfo(float).tiny)
    rows = peaks // density.shape[1]
    columns = peaks - rows * density.shape[1]
    x_values = x_grid[columns]
    y_values = y_grid[rows]
    boundary_distances = np.minimum.reduce(
        (
            x_values - x_grid[0],
            x_grid[-1] - x_values,
            y_values - y_grid[0],
            y_grid[-1] - y_values,
        )
    )
    accepted = np.flatnonzero(
        (ratio_values >= minimum_ratio)
        & (peak_density_values >= minimum_fraction * maximum)
        & (boundary_distances >= bandwidth)
    )
    modes = []
    for index in accepted:
        ratio = float(ratio_values[index])
        modes.append(
            {
                "flat_index": int(peaks[index]),
                "x_kpc": float(x_values[index]),
                "y_kpc": float(y_values[index]),
                "peak_density_per_kpc2": float(peak_density_values[index]),
                "saddle_density_per_kpc2": float(saddle_values[index]),
                "peak_to_saddle_ratio": ratio,
                "log_persistence": math.log(ratio),
                "boundary_distance_kpc": float(boundary_distances[index]),
            }
        )
    modes.sort(key=lambda row: (-float(row["peak_density_per_kpc2"]), row["x_kpc"], row["y_kpc"]))
    return modes


def select_primary_pair(modes: list[dict], config: dict) -> list[dict]:
    minimum = float(config["minimum_pair_separation_kpc"])
    maximum = float(config["maximum_pair_separation_kpc"])
    candidates = []
    for first in range(len(modes)):
        for second in range(first + 1, len(modes)):
            separation = math.hypot(
                modes[first]["x_kpc"] - modes[second]["x_kpc"],
                modes[first]["y_kpc"] - modes[second]["y_kpc"],
            )
            if minimum <= separation <= maximum:
                product = (
                    modes[first]["peak_density_per_kpc2"]
                    * modes[second]["peak_density_per_kpc2"]
                )
                tie_coordinates = tuple(
                    sorted(
                        (
                            (modes[first]["x_kpc"], modes[first]["y_kpc"]),
                            (modes[second]["x_kpc"], modes[second]["y_kpc"]),
                        )
                    )
                )
                candidates.append((product, separation, tie_coordinates, first, second))
    if not candidates:
        return []
    candidates.sort(key=lambda row: (-row[0], -row[1], row[2]))
    chosen = candidates[0]
    return [dict(modes[chosen[3]]), dict(modes[chosen[4]])]


def evaluate_points(
    x: np.ndarray,
    y: np.ndarray,
    velocity: np.ndarray,
    velocity_error: np.ndarray,
    points: np.ndarray,
    bandwidth: float,
) -> list[dict[str, float]]:
    result = []
    normalization = 1.0 / (2.0 * math.pi * bandwidth**2)
    for point_x, point_y in points:
        weights = normalization * np.exp(
            -((x - point_x) ** 2 + (y - point_y) ** 2) / (2.0 * bandwidth**2)
        )
        density = float(np.sum(weights))
        mean = float(np.sum(weights * velocity) / density)
        variance = max(
            0.0,
            float(np.sum(weights * ((velocity - mean) ** 2 - velocity_error**2)) / density),
        )
        result.append(
            {
                "number_density_per_kpc2": density,
                "mean_velocity_km_s": mean,
                "intrinsic_sigma_km_s": math.sqrt(variance),
                "number_current_per_kpc2_km_s": density * mean,
                "random_stress_per_kpc2_km2_s2": density * variance,
            }
        )
    return result


def bootstrap_modes(
    source: dict,
    redshift: float,
    primary_pair: list[dict],
    bandwidths: list[float],
    field_config: dict,
    mode_config: dict,
    uncertainty_config: dict,
    draws: int,
    seed: int,
) -> tuple[list[dict], list[dict]]:
    generator = np.random.default_rng(seed)
    records = []
    failures = []
    primary_points = np.asarray([[row["x_kpc"], row["y_kpc"]] for row in primary_pair])
    count = len(source["x"])
    for draw in range(draws):
        try:
            indices = generator.integers(0, count, size=count)
            x = source["x"][indices]
            y = source["y"][indices]
            cz_error = source["velocity_error"][indices] * (1.0 + redshift)
            cz = source["cz"][indices] + generator.normal(0.0, cz_error)
            velocity = (cz - np.median(cz)) / (1.0 + redshift)
            velocity_error = cz_error / (1.0 + redshift)
            bandwidth, _ = select_bandwidth(x, y, bandwidths)
            x_grid, y_grid = evaluation_grid(
                x,
                y,
                bandwidth,
                float(field_config["evaluation_grid_spacing_kpc"]),
                float(field_config["grid_padding_bandwidths"]),
            )
            density = density_grid(x, y, x_grid, y_grid, bandwidth)
            modes = persistent_modes(density, x_grid, y_grid, bandwidth, mode_config)
            if len(primary_pair) and len(modes):
                mode_points = np.asarray([[row["x_kpc"], row["y_kpc"]] for row in modes])
                distances = np.linalg.norm(
                    primary_points[:, None, :] - mode_points[None, :, :], axis=2
                )
                primary_index, mode_index = linear_sum_assignment(distances)
                match_by_primary = {int(p): int(m) for p, m in zip(primary_index, mode_index)}
            else:
                distances = np.empty((len(primary_pair), 0))
                match_by_primary = {}
            matches = []
            for primary_index in range(len(primary_pair)):
                if primary_index not in match_by_primary:
                    matches.append({"recovered": False})
                    continue
                mode_index = match_by_primary[primary_index]
                distance = float(distances[primary_index, mode_index])
                recovered = distance <= float(uncertainty_config["mode_match_radius_kpc"])
                mode = modes[mode_index]
                match = {
                    "recovered": bool(recovered),
                    "distance_kpc": distance,
                    "x_kpc": mode["x_kpc"],
                    "y_kpc": mode["y_kpc"],
                }
                if recovered:
                    match.update(
                        evaluate_points(
                            x,
                            y,
                            velocity,
                            velocity_error,
                            np.asarray([[mode["x_kpc"], mode["y_kpc"]]]),
                            bandwidth,
                        )[0]
                    )
                matches.append(match)
            records.append(
                {
                    "draw": draw,
                    "selected_bandwidth_kpc": bandwidth,
                    "persistent_mode_count": len(modes),
                    "matches": matches,
                }
            )
        except Exception as error:  # noqa: BLE001 - fail closed and preserve the draw
            failures.append({"draw": draw, "error": f"{type(error).__name__}: {error}"})
        if (draw + 1) % 100 == 0:
            print(f"bootstrap {draw + 1}/{draws}: {len(failures)} failures", flush=True)
    return records, failures


def interval_summary(
    records: list[dict], primary_count: int, requested_draws: int
) -> list[dict]:
    fields = (
        "x_kpc",
        "y_kpc",
        "distance_kpc",
        "number_density_per_kpc2",
        "mean_velocity_km_s",
        "intrinsic_sigma_km_s",
        "number_current_per_kpc2_km_s",
        "random_stress_per_kpc2_km2_s2",
    )
    summaries = []
    for primary_index in range(primary_count):
        recovered = [
            row["matches"][primary_index]
            for row in records
            if len(row["matches"]) > primary_index
            and row["matches"][primary_index].get("recovered", False)
        ]
        summary = {
            "primary_mode_index": primary_index,
            "recovered_draws": len(recovered),
            "recovery_fraction_of_requested": len(recovered) / requested_draws,
            "quantiles_2p5_16_50_84_97p5": {},
        }
        for field in fields:
            values = np.asarray([row[field] for row in recovered], dtype=float)
            summary["quantiles_2p5_16_50_84_97p5"][field] = (
                np.percentile(values, [2.5, 16.0, 50.0, 84.0, 97.5]).tolist()
                if len(values)
                else []
            )
        summaries.append(summary)
    return summaries


def published_label_validation(
    rows: list[dict[str, str]], pair: list[dict], x_kpc: np.ndarray, y_kpc: np.ndarray
) -> dict:
    declared = [row["subcluster_label"].strip() for row in rows]
    usable = np.asarray([value not in {"", "unassigned"} for value in declared])
    if not np.any(usable) or len(pair) < 2:
        return {"available": bool(np.any(usable)), "used_for_selection": False}
    target_values = sorted({value for value, keep in zip(declared, usable) if keep})
    target_map = {value: index for index, value in enumerate(target_values)}
    target = np.asarray([target_map[value] for value, keep in zip(declared, usable) if keep])
    member_xy = np.column_stack([x_kpc, y_kpc])[usable]
    pair_xy = np.asarray([[row["x_kpc"], row["y_kpc"]] for row in pair])
    predicted = np.argmin(np.linalg.norm(member_xy[:, None, :] - pair_xy[None, :, :], axis=2), axis=1)
    return {
        "available": True,
        "rows": int(np.sum(usable)),
        "adjusted_rand_index": float(adjusted_rand_score(target, predicted)),
        "used_for_selection": False,
    }


def render_fields(
    path: Path,
    fields: dict[str, np.ndarray],
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    modes: list[dict],
    pair: list[dict],
    cluster: str,
) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.6), constrained_layout=True)
    panels = (
        ("number_density", "Member number density"),
        ("mean_velocity", "Mean LOS velocity (km/s)"),
        ("number_current", "LOS number current proxy"),
    )
    extent = [x_grid[0], x_grid[-1], y_grid[0], y_grid[-1]]
    pair_indices = {int(row["flat_index"]) for row in pair}
    for axis, (field, title) in zip(axes, panels):
        image = axis.imshow(fields[field], origin="lower", extent=extent, aspect="equal")
        for mode in modes:
            selected = int(mode["flat_index"]) in pair_indices
            axis.scatter(
                mode["x_kpc"], mode["y_kpc"], marker="*" if selected else "+",
                s=90 if selected else 35, color="white", edgecolors="black" if selected else None,
                linewidths=0.8,
            )
        axis.set_title(title)
        axis.set_xlabel("x (kpc)")
        figure.colorbar(image, ax=axis, fraction=0.046)
    axes[0].set_ylabel("y (kpc)")
    figure.suptitle(f"{cluster}: frozen source-only continuous member fields")
    figure.savefig(path, dpi=160)
    plt.close(figure)


def run_cluster(config: dict, cluster: str, output: Path) -> dict:
    cluster_config = config["inputs"][cluster]
    catalog_path = ROOT / cluster_config["catalog"]
    if common.sha256(catalog_path) != cluster_config["catalog_sha256"]:
        raise RuntimeError(f"{cluster} catalog hash mismatch")
    rows = load_catalog(catalog_path)
    if len(rows) != int(cluster_config["rows"]):
        raise RuntimeError(f"{cluster} catalog row count mismatch")
    source = physical_catalog(
        rows,
        float(cluster_config["redshift"]),
        float(cluster_config["kpc_per_arcsec_Planck18"]),
    )
    bandwidths = [float(value) for value in config["spatial_kernel"]["candidate_bandwidths_kpc"]]
    bandwidth, scores = select_bandwidth(source["x"], source["y"], bandwidths)
    field_config = config["continuous_fields"]
    mode_config = config["topological_modes"]
    x_grid, y_grid = evaluation_grid(
        source["x"], source["y"], bandwidth,
        float(field_config["evaluation_grid_spacing_kpc"]),
        float(field_config["grid_padding_bandwidths"]),
    )
    fields = separable_kernel_fields(
        source["x"], source["y"], source["velocity"], source["velocity_error"],
        x_grid, y_grid, bandwidth,
    )
    modes = persistent_modes(fields["number_density"], x_grid, y_grid, bandwidth, mode_config)
    pair = select_primary_pair(modes, mode_config)
    for mode, values in zip(
        pair,
        evaluate_points(
            source["x"], source["y"], source["velocity"], source["velocity_error"],
            np.asarray([[row["x_kpc"], row["y_kpc"]] for row in pair]), bandwidth,
        ),
    ):
        mode.update(values)
    print(
        f"{cluster}: h={bandwidth:.1f} kpc, {len(modes)} persistent modes, "
        f"primary pair={'yes' if len(pair) == 2 else 'no'}",
        flush=True,
    )
    draws = int(config["uncertainty"]["catalog_bootstraps"])
    bootstrap, failures = bootstrap_modes(
        source, float(cluster_config["redshift"]), pair, bandwidths, field_config,
        mode_config, config["uncertainty"], draws,
        stable_seed(int(config["uncertainty"]["bootstrap_seed"]), cluster),
    )
    summaries = interval_summary(bootstrap, len(pair), draws)
    recovery = [row["recovery_fraction_of_requested"] for row in summaries]
    separation = (
        math.hypot(pair[0]["x_kpc"] - pair[1]["x_kpc"], pair[0]["y_kpc"] - pair[1]["y_kpc"])
        if len(pair) == 2 else None
    )
    all_arrays_finite = all(np.all(np.isfinite(value)) for value in fields.values())
    gates = {
        "all_bandwidth_scores_finite": bool(np.all(np.isfinite(scores))),
        "minimum_persistent_primary_modes": len(pair) == int(mode_config["minimum_primary_modes"]),
        "primary_pair_separation_inside_frozen_bounds": bool(
            separation is not None
            and float(mode_config["minimum_pair_separation_kpc"]) <= separation
            <= float(mode_config["maximum_pair_separation_kpc"])
        ),
        "each_primary_mode_recovered_in_at_least_68_percent_of_bootstraps": bool(
            len(recovery) == 2
            and all(value >= float(config["uncertainty"]["minimum_mode_recovery_fraction"]) for value in recovery)
        ),
        "at_least_99_percent_of_bootstraps_finite": len(failures) / draws
        <= float(config["uncertainty"]["maximum_failed_bootstrap_fraction"]),
        "all_primary_field_arrays_finite": bool(all_arrays_finite),
    }
    stem = cluster.lower()
    arrays_path = output / f"{stem}_fields.npz"
    np.savez_compressed(arrays_path, x_kpc=x_grid, y_kpc=y_grid, **fields)
    figure_path = output / f"{stem}_fields.png"
    render_fields(figure_path, fields, x_grid, y_grid, modes, pair, cluster)
    draws_path = output / f"{stem}_bootstrap.json"
    draws_path.write_text(json.dumps(bootstrap, separators=(",", ":")) + "\n", encoding="utf-8")
    return {
        "cluster": cluster,
        "rows": len(rows),
        "coordinate_origin": {
            "ra_deg": source["ra_origin_deg"],
            "dec_deg": source["dec_origin_deg"],
        },
        "bandwidth_scores": [
            {"bandwidth_kpc": value, "loo_log_density": float(score)}
            for value, score in zip(bandwidths, scores)
        ],
        "selected_bandwidth_kpc": bandwidth,
        "grid": {
            "spacing_kpc": field_config["evaluation_grid_spacing_kpc"],
            "shape_y_x": list(fields["number_density"].shape),
            "x_bounds_kpc": [float(x_grid[0]), float(x_grid[-1])],
            "y_bounds_kpc": [float(y_grid[0]), float(y_grid[-1])],
        },
        "persistent_mode_count": len(modes),
        "persistent_modes": modes,
        "primary_pair": pair,
        "primary_pair_separation_kpc": separation,
        "published_label_validation": published_label_validation(
            rows, pair, source["x"], source["y"]
        ),
        "bootstrap": {
            "requested_draws": draws,
            "accepted_draws": len(bootstrap),
            "failed_draws": len(failures),
            "failure_fraction": len(failures) / draws,
            "selected_bandwidth_counts": {
                str(value): sum(row["selected_bandwidth_kpc"] == value for row in bootstrap)
                for value in bandwidths
            },
            "primary_mode_summaries": summaries,
            "failures": failures,
            "draws_file": draws_path.relative_to(ROOT).as_posix(),
            "draws_sha256": common.sha256(draws_path),
        },
        "products": [
            {
                "kind": "continuous_field_arrays",
                "path": arrays_path.relative_to(ROOT).as_posix(),
                "sha256": common.sha256(arrays_path),
                "bytes": arrays_path.stat().st_size,
            },
            {
                "kind": "source_only_diagnostic_figure",
                "path": figure_path.relative_to(ROOT).as_posix(),
                "sha256": common.sha256(figure_path),
                "bytes": figure_path.stat().st_size,
            },
        ],
        "gates": gates,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = common.load_json(config_path)
    common.validate_parent_hashes(config)
    source_report = common.load_json(ROOT / config["parents"]["source_map_report"])
    failed_report = common.load_json(ROOT / config["parents"]["failed_member_phase_report"])
    if source_report["status"] != "both_clusters_passed_frozen_v19h_source_map_gate":
        raise RuntimeError("the inherited source-map gate did not pass")
    if failed_report["status"] != "frozen_v19h_member_phase_gate_failed":
        raise RuntimeError("V19I is only authorized after the frozen discrete gate failure")
    if source_report["lensing_target_opened"] is not False or failed_report["lensing_target_opened"] is not False:
        raise RuntimeError("a parent opened a sealed lensing target")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    clusters = [run_cluster(config, cluster, output) for cluster in config["sample"]["clusters"]]
    failed = [row["cluster"] for row in clusters if not all(row["gates"].values())]
    report = {
        "status": (
            "both_clusters_passed_frozen_v19i_continuous_member_field_gate"
            if not failed else "frozen_v19i_continuous_member_field_gate_failed"
        ),
        "protocol_version": config["protocol_version"],
        "generated_utc": datetime.now(UTC).isoformat(),
        "config_sha256": common.sha256(config_path),
        "runner_sha256": common.sha256(Path(__file__).resolve()),
        "failed_clusters": failed,
        "clusters": clusters,
        "published_subcluster_labels_used_for_selection": False,
        "mass_current_claimed": False,
        "number_current_is_geometry_proxy_only": True,
        "registered_science_images_visually_inspected": False,
        "lensing_target_opened": False,
        "gravity_formula_selected": False,
        "gravity_parameter_changed": False,
    }
    report_path = output / "report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"status: {report['status']}")
    print(f"report: {report_path}")
    print(f"sha256: {common.sha256(report_path)}")


if __name__ == "__main__":
    main()
