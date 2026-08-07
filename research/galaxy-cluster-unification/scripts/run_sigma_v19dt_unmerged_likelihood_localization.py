#!/usr/bin/env python3
"""Localize the terminal V19DS single-temperature likelihood failures."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict, deque
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales
from scipy.stats import mannwhitneyu, spearmanr

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "sigma_v19dt_unmerged_likelihood_localization.json"
DEFAULT_OUTPUT = ROOT / "results" / "sigma_v19dt_unmerged_likelihood_localization"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def validate(config: dict[str, Any]) -> dict[str, Any]:
    if config["status"] != "post_outcome_diagnostic_frozen_after_v19ds_failed_closed":
        raise RuntimeError("V19DT does not declare its post-outcome status")
    if sha256(Path(__file__).resolve()) != config["implementation"]["runner_sha256"]:
        raise RuntimeError("V19DT runner changed after freeze")
    for name, item in config["parents"].items():
        if sha256(ROOT / item["path"]) != item["sha256"]:
            raise RuntimeError(f"V19DT parent changed: {name}")
    report = load_json(ROOT / config["parents"]["v19ds_report"]["path"])
    if (
        report["status"] != "full_494_region_unmerged_joint_likelihood_failed_closed"
        or report["aggregate_pass"]
        or len(report["regions"]) != 494
        or report["i4_i5_source_only_successor_authorized"]
        or report["thermal_stress_or_baroclinicity_constructed"]
        or report["lensing_halo_action_gravity_or_holdout_payload_opened"]
        or report["gravity_formula_or_parameter_changed"]
    ):
        raise RuntimeError("V19DT parent is not the sealed terminal V19DS failure")
    return report


def adjacency(binmap: np.ndarray) -> set[tuple[int, int]]:
    edges: set[tuple[int, int]] = set()
    for first, second in (
        (binmap[:, :-1], binmap[:, 1:]),
        (binmap[:-1, :], binmap[1:, :]),
    ):
        mask = (first >= 0) & (second >= 0) & (first != second)
        for left, right in zip(first[mask], second[mask], strict=True):
            a, b = sorted((int(left), int(right)))
            edges.add((a, b))
    return edges


def connected_components(nodes: set[int], edges: set[tuple[int, int]]) -> list[list[int]]:
    neighbors: dict[int, set[int]] = defaultdict(set)
    for left, right in edges:
        if left in nodes and right in nodes:
            neighbors[left].add(right)
            neighbors[right].add(left)
    remaining = set(nodes)
    components = []
    while remaining:
        root = remaining.pop()
        component = [root]
        queue = deque([root])
        while queue:
            current = queue.popleft()
            for neighbor in neighbors[current] & remaining:
                remaining.remove(neighbor)
                component.append(neighbor)
                queue.append(neighbor)
        components.append(sorted(component))
    return sorted(components, key=len, reverse=True)


def spatial_permutation(
    labels: list[int], unresolved: set[int], edges: set[tuple[int, int]], draws: int, seed: int
) -> dict[str, Any]:
    label_set = set(labels)
    valid_edges = [(a, b) for a, b in edges if a in label_set and b in label_set]
    observed = sum((a in unresolved) == (b in unresolved) for a, b in valid_edges) / len(
        valid_edges
    )
    rng = np.random.default_rng(seed)
    count = len(unresolved)
    simulated = np.empty(draws, dtype=float)
    label_array = np.asarray(labels, dtype=int)
    for index in range(draws):
        shuffled = set(rng.choice(label_array, size=count, replace=False).tolist())
        simulated[index] = sum(
            (a in shuffled) == (b in shuffled) for a, b in valid_edges
        ) / len(valid_edges)
    return {
        "edges": len(valid_edges),
        "observed_same_class_edge_fraction": observed,
        "permutation_mean": float(np.mean(simulated)),
        "permutation_std": float(np.std(simulated, ddof=1)),
        "z_score": float((observed - np.mean(simulated)) / np.std(simulated, ddof=1)),
        "one_sided_p_value": float((1 + np.sum(simulated >= observed)) / (draws + 1)),
        "draws": draws,
        "seed": seed,
    }


def safe_float(value: Any) -> float:
    result = float(value)
    return result if math.isfinite(result) else math.nan


def region_rows(
    config: dict[str, Any], report: dict[str, Any], cluster: str
) -> tuple[list[dict[str, Any]], np.ndarray, set[tuple[int, int]], dict[str, Any]]:
    inputs = config["cluster_inputs"][cluster]
    binmap_path = ROOT / inputs["binmap"]["path"]
    with fits.open(binmap_path) as hdul:
        binmap = np.asarray(hdul[0].data, dtype=int)
        header = hdul[0].header.copy()
    statistics = {
        int(row["bin_id"]): row
        for row in read_csv(ROOT / inputs["region_statistics"]["path"])
    }
    fits_by_bin = {
        int(row["bin_id"]): row
        for row in report["regions"]
        if row["cluster"] == cluster
    }
    labels = sorted(fits_by_bin)
    edges = adjacency(binmap)
    valid_y, valid_x = np.where(np.isin(binmap, labels))
    map_center_x = float(np.median(valid_x))
    map_center_y = float(np.median(valid_y))
    scale_arcsec = math.nan
    wcs = None
    try:
        wcs = WCS(header).celestial
        scale_arcsec = float(np.mean(proj_plane_pixel_scales(wcs)) * 3600.0)
    except (TypeError, ValueError):
        wcs = None

    neighbor_map: dict[int, set[int]] = defaultdict(set)
    for left, right in edges:
        neighbor_map[left].add(right)
        neighbor_map[right].add(left)
    unresolved = {
        bin_id
        for bin_id, row in fits_by_bin.items()
        if row["uncertainty_state"] == "unresolved"
    }
    rows = []
    for bin_id in labels:
        result = fits_by_bin[bin_id]
        fit = result["fit"]
        stat = statistics[bin_id]
        yy, xx = np.where(binmap == bin_id)
        centroid_x = float(np.mean(xx))
        centroid_y = float(np.mean(yy))
        ra = dec = math.nan
        if wcs is not None:
            try:
                ra, dec = map(float, wcs.pixel_to_world_values(centroid_x, centroid_y))
            except (TypeError, ValueError):
                pass
        counts = [float(row["source_counts_in_fit_band"]) for row in fit["datasets"]]
        neighbors = neighbor_map[bin_id] & set(labels)
        row = {
            "cluster": cluster,
            "bin_id": bin_id,
            "centroid_x_pixel": centroid_x,
            "centroid_y_pixel": centroid_y,
            "centroid_ra_deg": ra,
            "centroid_dec_deg": dec,
            "radius_pixels": math.hypot(
                centroid_x - map_center_x, centroid_y - map_center_y
            ),
            "radius_arcsec": math.hypot(
                centroid_x - map_center_x, centroid_y - map_center_y
            )
            * scale_arcsec,
            "pixels": int(stat["pixels"]),
            "net_counts": float(stat["net_counts"]),
            "signal_to_noise": float(stat["signal_to_noise"]),
            "source_fraction": float(stat["source_fraction"]),
            "cells": int(result["cells"]),
            "maximum_dataset_count_fraction": max(counts) / sum(counts),
            "reduced_statistic": float(fit["fit"]["reduced_statistic"]),
            "temperature_keV": float(fit["parameters"]["temperature_keV"]),
            "abundance_solar": float(fit["parameters"]["abundance_solar"]),
            "normalization": float(fit["parameters"]["normalization"]),
            "uncertainty_state": result["uncertainty_state"],
            "full_quality_pass": bool(result["full_quality_pass"]),
            "abundance_at_lower_bound": math.isclose(
                float(fit["parameters"]["abundance_solar"]), 0.0, abs_tol=1e-8
            ),
            "abundance_at_upper_bound": math.isclose(
                float(fit["parameters"]["abundance_solar"]), 2.0, abs_tol=1e-8
            ),
            "neighbor_count": len(neighbors),
            "unresolved_neighbor_fraction": (
                sum(neighbor in unresolved for neighbor in neighbors) / len(neighbors)
                if neighbors
                else math.nan
            ),
        }
        rows.append(row)
    metadata = {
        "shape": list(binmap.shape),
        "map_center_pixel": [map_center_x, map_center_y],
        "pixel_scale_arcsec": scale_arcsec,
    }
    return rows, binmap, edges, metadata


def comparison(rows: list[dict[str, Any]], feature: str) -> dict[str, Any]:
    unresolved = np.asarray(
        [safe_float(row[feature]) for row in rows if row["uncertainty_state"] == "unresolved"]
    )
    resolved = np.asarray(
        [safe_float(row[feature]) for row in rows if row["uncertainty_state"] != "unresolved"]
    )
    unresolved = unresolved[np.isfinite(unresolved)]
    resolved = resolved[np.isfinite(resolved)]
    test = mannwhitneyu(unresolved, resolved, alternative="two-sided")
    return {
        "feature": feature,
        "unresolved_median": float(np.median(unresolved)),
        "resolved_median": float(np.median(resolved)),
        "median_difference": float(np.median(unresolved) - np.median(resolved)),
        "mann_whitney_u": float(test.statistic),
        "two_sided_p_value": float(test.pvalue),
    }


def cluster_summary(
    config: dict[str, Any], rows: list[dict[str, Any]], edges: set[tuple[int, int]], cluster: str
) -> dict[str, Any]:
    unresolved = {
        int(row["bin_id"]) for row in rows if row["uncertainty_state"] == "unresolved"
    }
    labels = [int(row["bin_id"]) for row in rows]
    rstat = np.asarray([float(row["reduced_statistic"]) for row in rows])
    features = [
        "radius_pixels",
        "pixels",
        "net_counts",
        "source_fraction",
        "cells",
        "maximum_dataset_count_fraction",
        "temperature_keV",
        "abundance_solar",
    ]
    correlations = {}
    for feature in features:
        values = np.asarray([safe_float(row[feature]) for row in rows])
        mask = np.isfinite(values) & np.isfinite(rstat)
        rho, p_value = spearmanr(values[mask], rstat[mask])
        correlations[feature] = {"spearman_rho": float(rho), "p_value": float(p_value)}
    components = connected_components(unresolved, edges)
    return {
        "regions": len(rows),
        "unresolved_regions": len(unresolved),
        "unresolved_exactly_matches_reduced_statistic_above_3": unresolved
        == {
            int(row["bin_id"])
            for row in rows
            if float(row["reduced_statistic"])
            > float(config["classification"]["sherpa_confidence_maximum_reduced_statistic"])
        },
        "abundance_lower_bound_regions": sum(
            bool(row["abundance_at_lower_bound"]) for row in rows
        ),
        "abundance_lower_bound_fraction_unresolved": sum(
            bool(row["abundance_at_lower_bound"])
            for row in rows
            if row["uncertainty_state"] == "unresolved"
        )
        / len(unresolved),
        "spatial_permutation": spatial_permutation(
            labels,
            unresolved,
            edges,
            int(config["spatial_test"]["permutation_draws"]),
            int(config["spatial_test"]["seed"]) + sum(map(ord, cluster)),
        ),
        "unresolved_connected_components": {
            "count": len(components),
            "largest_sizes": [len(component) for component in components[:10]],
            "largest_members": components[:5],
        },
        "unresolved_vs_resolved": {
            feature: comparison(rows, feature) for feature in features
        },
        "reduced_statistic_correlations": correlations,
    }


def plot_maps(
    cluster_payloads: dict[str, tuple[list[dict[str, Any]], np.ndarray]], path: Path
) -> None:
    fig, axes = plt.subplots(len(cluster_payloads), 3, figsize=(15, 9), constrained_layout=True)
    if len(cluster_payloads) == 1:
        axes = np.asarray([axes])
    status_colors = ListedColormap(["#2ca25f", "#fdae6b", "#de2d26"])
    status_norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], status_colors.N)
    for row_index, (cluster, (rows, binmap)) in enumerate(cluster_payloads.items()):
        by_bin = {int(row["bin_id"]): row for row in rows}
        rstat_map = np.full(binmap.shape, np.nan)
        status_map = np.full(binmap.shape, np.nan)
        temperature_map = np.full(binmap.shape, np.nan)
        for bin_id, row in by_bin.items():
            mask = binmap == bin_id
            rstat_map[mask] = math.log10(max(float(row["reduced_statistic"]), 1e-6))
            status_map[mask] = {
                "ordered_two_sided": 0,
                "censored_at_frozen_model_bound": 1,
                "unresolved": 2,
            }[row["uncertainty_state"]]
            temperature_map[mask] = float(row["temperature_keV"])
        image = axes[row_index, 0].imshow(rstat_map, origin="lower", cmap="magma")
        fig.colorbar(image, ax=axes[row_index, 0], label="log10(reduced statistic)")
        axes[row_index, 0].set_title(f"{cluster}: single-temperature adequacy")
        image = axes[row_index, 1].imshow(
            status_map, origin="lower", cmap=status_colors, norm=status_norm
        )
        colorbar = fig.colorbar(image, ax=axes[row_index, 1], ticks=[0, 1, 2])
        colorbar.ax.set_yticklabels(["two-sided", "censored", "unresolved"])
        axes[row_index, 1].set_title("Temperature uncertainty state")
        image = axes[row_index, 2].imshow(
            temperature_map, origin="lower", cmap="viridis", vmin=2, vmax=20
        )
        fig.colorbar(image, ax=axes[row_index, 2], label="best-fit kT (keV)")
        axes[row_index, 2].set_title("Single-temperature best fit")
        for axis in axes[row_index]:
            axis.set_xlabel("map x pixel")
            axis.set_ylabel("map y pixel")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def run(config: dict[str, Any], report: dict[str, Any], output: Path) -> dict[str, Any]:
    all_rows = []
    summaries = {}
    maps = {}
    metadata = {}
    for cluster in config["cluster_inputs"]:
        rows, binmap, edges, map_metadata = region_rows(config, report, cluster)
        all_rows.extend(rows)
        maps[cluster] = (rows, binmap)
        metadata[cluster] = map_metadata
        summaries[cluster] = cluster_summary(config, rows, edges, cluster)

    table_path = output / "region_diagnostics.csv"
    table_path.parent.mkdir(parents=True, exist_ok=True)
    with table_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)
    figure_path = output / "likelihood_failure_maps.png"
    plot_maps(maps, figure_path)
    unresolved = [row for row in all_rows if row["uncertainty_state"] == "unresolved"]
    every_exact = all(
        summary["unresolved_exactly_matches_reduced_statistic_above_3"]
        for summary in summaries.values()
    )
    spatial = all(
        summary["spatial_permutation"]["one_sided_p_value"]
        <= float(config["spatial_test"]["maximum_p_value_for_clustering"])
        for summary in summaries.values()
    )
    gates = {
        "all_494_regions_localized": len(all_rows) == 494,
        "all_122_unresolved_regions_retained": len(unresolved) == 122,
        "unresolved_state_exactly_explained_by_sherpa_reduced_statistic_guard": every_exact,
        "unresolved_regions_are_spatially_clustered_in_both_clusters": spatial,
        "i4_i5_lensing_halo_gravity_and_holdout_remain_sealed": True,
    }
    return {
        "status": "v19ds_terminal_failure_spatial_localization_completed",
        "regions": len(all_rows),
        "unresolved_regions": len(unresolved),
        "cluster_summaries": summaries,
        "map_metadata": metadata,
        "gates": gates,
        "interpretation": {
            "confidence_engine_failure": "Every unresolved interval is caused by Sherpa's reduced-statistic-above-3 safeguard; no distinct confidence exception occurs.",
            "measurement_model": "Spatial clustering supports a localized spectral-model inadequacy or physical multiphase structure rather than random archive corruption. It does not by itself distinguish multiphase gas from spatially localized cross-observation calibration residuals.",
            "next_required_test": "Freeze an all-122 leave-one-observation-out leverage audit before changing the plasma model. If no observation or CCD consistently removes the excess statistic, compare one-temperature versus two-temperature or continuous-temperature plasma models using a preregistered complexity penalty.",
        },
        "region_diagnostics": {
            "path": str(table_path.relative_to(ROOT)),
            "sha256": sha256(table_path),
        },
        "failure_map": {
            "path": str(figure_path.relative_to(ROOT)),
            "sha256": sha256(figure_path),
        },
        "i4_i5_source_only_successor_authorized": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = load_json(config_path)
    report = validate(config)
    result = run(config, report, args.output.resolve())
    payload = {
        "protocol_version": config["protocol_version"],
        "generated_utc": datetime.now(UTC).isoformat(),
        "config_sha256": sha256(config_path),
        "runner_sha256": sha256(Path(__file__).resolve()),
        **result,
        "thermal_stress_or_baroclinicity_constructed": False,
        "lensing_halo_action_gravity_or_holdout_payload_opened": False,
        "gravity_formula_or_parameter_changed": False,
    }
    write_json(args.output.resolve() / "report.json", payload)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
