#!/usr/bin/env python3
"""Run the frozen V19K smooth-null versus discontinuity front test."""

from __future__ import annotations

import argparse
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
from scipy import ndimage
from scipy.optimize import minimize
from scipy.spatial import cKDTree

ROOT = common.ROOT
DEFAULT_CONFIG = ROOT / "configs" / "sigma_v19k_smooth_null_front_likelihood.json"
DEFAULT_OUTPUT = ROOT / "results" / "sigma_v19k_smooth_null_fronts"
V19J_RUNNER = ROOT / "scripts" / "run_sigma_v19j_automated_fronts.py"


def stable_seed(base: int, label: str) -> int:
    payload = f"{base}:{label}:v19k-smooth-null".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big")


def v19j_module():
    return common.load_module(V19J_RUNNER, "sigma_v19j_for_v19k")


def circular_difference(first: float, second: float) -> float:
    return abs(math.atan2(math.sin(first - second), math.cos(first - second)))


def footprint(radius_pixels: float) -> np.ndarray:
    extent = math.ceil(radius_pixels)
    coordinate = np.arange(-extent, extent + 1)
    yy, xx = np.meshgrid(coordinate, coordinate, indexing="ij")
    return xx**2 + yy**2 <= radius_pixels**2 + 1e-12


def local_maximum_seeds(
    candidate: np.ndarray,
    score: np.ndarray,
    angle: np.ndarray,
    scale_index: np.ndarray,
    pixel_kpc: float,
    radius_kpc: float,
    center_logical_x: float,
    center_logical_y: float,
) -> list[dict]:
    safe = np.where(candidate & np.isfinite(score), score, -np.inf)
    maximum = ndimage.maximum_filter(
        safe, footprint=footprint(radius_kpc / pixel_kpc), mode="constant", cval=-np.inf
    )
    maxima = candidate & np.isfinite(score) & (safe == maximum)
    # Resolve an exact connected score plateau by the lexicographically first pixel.
    labels, count = ndimage.label(maxima, structure=np.ones((3, 3), dtype=int))
    keep = []
    for label_index in range(1, count + 1):
        rows, columns = np.nonzero(labels == label_index)
        values = score[rows, columns]
        best = float(np.max(values))
        tied = np.flatnonzero(values == best)
        chosen = min(tied, key=lambda index: (int(rows[index]), int(columns[index])))
        row = int(rows[chosen])
        column = int(columns[chosen])
        keep.append(
            {
                "seed_id": 0,
                "row": row,
                "column": column,
                "x_kpc": float((column + 1.0 - center_logical_x) * pixel_kpc),
                "y_kpc": float((row + 1.0 - center_logical_y) * pixel_kpc),
                "normal_rad": float(angle[row, column] % (2.0 * math.pi)),
                "v19j_score_sigma": float(score[row, column]),
                "scale_index": int(scale_index[row, column]),
                "scale_kpc": float((8.0, 16.0, 32.0, 64.0)[scale_index[row, column]]),
            }
        )
    keep.sort(key=lambda row: (row["row"], row["column"]))
    for index, seed in enumerate(keep, start=1):
        seed["seed_id"] = index
    return keep


def extract_profile(
    seed: dict,
    counts: np.ndarray,
    background: np.ndarray,
    background_variance: np.ndarray,
    exposure: np.ndarray,
    mask: np.ndarray,
    pixel_kpc: float,
    config: dict,
) -> dict:
    normal_extent = float(config["normal_extent_each_side_kpc"])
    tangent_extent = float(config["tangent_half_width_kpc"])
    bin_width = float(config["normal_bin_width_kpc"])
    bins = int(config["normal_bins"])
    radius_pixels = math.ceil(math.hypot(normal_extent, tangent_extent) / pixel_kpc)
    row0 = int(seed["row"])
    column0 = int(seed["column"])
    row_min = max(0, row0 - radius_pixels)
    row_max = min(counts.shape[0], row0 + radius_pixels + 1)
    column_min = max(0, column0 - radius_pixels)
    column_max = min(counts.shape[1], column0 + radius_pixels + 1)
    yy, xx = np.indices((row_max - row_min, column_max - column_min))
    dx = (xx + column_min - column0) * pixel_kpc
    dy = (yy + row_min - row0) * pixel_kpc
    cosine = math.cos(float(seed["normal_rad"]))
    sine = math.sin(float(seed["normal_rad"]))
    normal = dx * cosine + dy * sine
    tangent = -dx * sine + dy * cosine
    geometric = (
        (normal >= -normal_extent)
        & (normal < normal_extent)
        & (np.abs(tangent) <= tangent_extent)
    )
    local_mask = mask[row_min:row_max, column_min:column_max]
    valid = geometric & local_mask
    geometric_count = int(np.sum(geometric))
    valid_count = int(np.sum(valid))
    valid_fraction = valid_count / geometric_count if geometric_count else 0.0
    if valid_fraction < float(config["minimum_overall_valid_pixel_fraction"]):
        return {"valid": False, "failure": "profile_valid_fraction", "valid_fraction": valid_fraction}
    bin_index = np.floor((normal + normal_extent) / bin_width).astype(int)
    selected = valid & (bin_index >= 0) & (bin_index < bins)
    indices = bin_index[selected]

    def summed(array: np.ndarray) -> np.ndarray:
        local = array[row_min:row_max, column_min:column_max]
        return np.bincount(indices, weights=local[selected], minlength=bins).astype(float)

    count_sum = summed(counts)
    background_sum = summed(background)
    background_variance_sum = summed(background_variance)
    exposure_sum = summed(exposure)
    valid_pixels = np.bincount(indices, minlength=bins).astype(int)
    centers = -normal_extent + (np.arange(bins) + 0.5) * bin_width
    positive = exposure_sum > 0.0
    if int(np.sum(positive)) < int(config["minimum_bins_with_positive_exposure"]):
        return {
            "valid": False,
            "failure": "positive_exposure_bins",
            "valid_fraction": valid_fraction,
        }
    return {
        "valid": True,
        "valid_fraction": valid_fraction,
        "centers_kpc": centers,
        "counts": count_sum,
        "background": background_sum,
        "background_variance": background_variance_sum,
        "exposure": exposure_sum,
        "valid_pixels": valid_pixels,
        "positive": positive,
    }


def log_likelihood_and_gradient(
    parameters: np.ndarray,
    design: np.ndarray,
    counts: np.ndarray,
    background: np.ndarray,
    exposure: np.ndarray,
) -> tuple[float, np.ndarray]:
    eta = np.clip(design @ parameters, -100.0, 100.0)
    source = exposure * np.exp(eta)
    expected = background + source
    if np.any(expected <= 0.0) or np.any(~np.isfinite(expected)):
        return -np.inf, np.full(len(parameters), np.nan)
    log_likelihood = float(np.sum(counts * np.log(expected) - expected))
    gradient = design.T @ (source * (counts / expected - 1.0))
    return log_likelihood, np.asarray(gradient, dtype=float)


def fit_design(
    initial: np.ndarray,
    bounds: list[tuple[float, float]],
    design: np.ndarray,
    counts: np.ndarray,
    background: np.ndarray,
    exposure: np.ndarray,
) -> dict:
    def objective(parameters: np.ndarray) -> tuple[float, np.ndarray]:
        value, gradient = log_likelihood_and_gradient(
            parameters, design, counts, background, exposure
        )
        if not np.isfinite(value) or np.any(~np.isfinite(gradient)):
            return 1e100, np.zeros_like(parameters)
        return -value, -gradient

    result = minimize(
        objective,
        initial,
        method="L-BFGS-B",
        jac=True,
        bounds=bounds,
        options={"maxiter": 500, "ftol": 1e-12, "maxls": 30},
    )
    if not result.success or not np.isfinite(result.fun):
        fallback = minimize(
            lambda parameters: objective(parameters)[0],
            initial,
            method="Powell",
            bounds=bounds,
            options={"maxiter": 1000, "ftol": 1e-12, "xtol": 1e-10},
        )
        result = minimize(
            objective,
            fallback.x,
            method="L-BFGS-B",
            jac=True,
            bounds=bounds,
            options={"maxiter": 500, "ftol": 1e-12, "maxls": 30},
        )
    value, _ = log_likelihood_and_gradient(
        result.x, design, counts, background, exposure
    )
    return {
        "success": bool(result.success and np.isfinite(value)),
        "message": str(result.message),
        "parameters": np.asarray(result.x),
        "log_likelihood": float(value),
        "iterations": int(result.nit),
    }


def fit_profile(profile: dict, config: dict) -> dict:
    positive = profile["positive"]
    centers = profile["centers_kpc"][positive]
    tau = centers / 200.0
    counts = profile["counts"][positive]
    background = profile["background"][positive]
    exposure = profile["exposure"][positive]
    bounds_config = config["parameter_bounds"]
    null_bounds = [tuple(bounds_config[key]) for key in ("a", "b", "c")]
    alternative_bounds = null_bounds + [tuple(bounds_config["d"])]
    net_rate = max(float(np.sum(counts - background) / np.sum(exposure)), math.exp(-40.0))
    initial_a = float(np.clip(math.log(net_rate), *null_bounds[0]))
    null_design = np.column_stack([np.ones(len(tau)), tau, tau**2])
    null = fit_design(
        np.asarray([initial_a, 0.0, 0.0]),
        null_bounds,
        null_design,
        counts,
        background,
        exposure,
    )
    alternatives = []
    if null["success"]:
        for delta in config["step_location_grid_kpc"]:
            step = (centers >= float(delta)).astype(float)
            design = np.column_stack([null_design, step])
            initial = np.concatenate([null["parameters"], [math.log(4.0)]])
            fit = fit_design(
                initial,
                alternative_bounds,
                design,
                counts,
                background,
                exposure,
            )
            fit["delta_kpc"] = float(delta)
            alternatives.append(fit)
    finite = [row for row in alternatives if row["success"]]
    if not null["success"] or not finite:
        return {
            "success": False,
            "null": serialize_fit(null),
            "alternatives": [serialize_fit(row) for row in alternatives],
        }
    finite.sort(
        key=lambda row: (-row["log_likelihood"], abs(row["delta_kpc"]), row["delta_kpc"])
    )
    best = finite[0]
    delta_cash = max(0.0, 2.0 * (best["log_likelihood"] - null["log_likelihood"]))
    score = math.sqrt(delta_cash)
    return {
        "success": True,
        "null": serialize_fit(null),
        "best_alternative": serialize_fit(best),
        "delta_cash": float(delta_cash),
        "step_score_sigma": float(score),
        "density_compression": float(math.exp(best["parameters"][3] / 2.0)),
        "tested_step_locations_kpc": [float(row["delta_kpc"]) for row in alternatives],
        "finite_alternative_count": len(finite),
    }


def serialize_fit(fit: dict) -> dict:
    result = {
        "success": bool(fit.get("success", False)),
        "message": str(fit.get("message", "")),
        "log_likelihood": float(fit.get("log_likelihood", -np.inf)),
        "iterations": int(fit.get("iterations", 0)),
        "parameters": np.asarray(fit.get("parameters", [])).tolist(),
    }
    if "delta_kpc" in fit:
        result["delta_kpc"] = float(fit["delta_kpc"])
    return result


def fit_seeds(
    seeds: list[dict],
    counts: np.ndarray,
    background: np.ndarray,
    background_variance: np.ndarray,
    exposure: np.ndarray,
    mask: np.ndarray,
    pixel_kpc: float,
    config: dict,
) -> list[dict]:
    records = []
    for index, seed in enumerate(seeds, start=1):
        profile = extract_profile(
            seed,
            counts,
            background,
            background_variance,
            exposure,
            mask,
            pixel_kpc,
            config["profile_extraction"],
        )
        record = dict(seed)
        record["profile_valid"] = bool(profile["valid"])
        record["profile_valid_fraction"] = float(profile.get("valid_fraction", 0.0))
        if profile["valid"]:
            fit = fit_profile(profile, config["poisson_models"])
            record["fit"] = fit
            record["passes_step_score"] = bool(
                fit.get("success", False)
                and fit.get("step_score_sigma", 0.0)
                >= float(config["poisson_models"]["minimum_discontinuity_score_sigma"])
            )
        else:
            record["profile_failure"] = profile["failure"]
            record["fit"] = {"success": False}
            record["passes_step_score"] = False
        records.append(record)
        if index % 100 == 0:
            print(f"seed fits {index}/{len(seeds)}", flush=True)
    return records


class UnionFind:
    def __init__(self, size: int):
        self.parent = list(range(size))

    def find(self, value: int) -> int:
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, first: int, second: int) -> None:
        first_root = self.find(first)
        second_root = self.find(second)
        if first_root != second_root:
            self.parent[max(first_root, second_root)] = min(first_root, second_root)


def tangent_alignment(node: dict, other: dict) -> float:
    vector = np.asarray(
        [other["x_kpc"] - node["x_kpc"], other["y_kpc"] - node["y_kpc"]]
    )
    vector /= np.linalg.norm(vector)
    tangent = np.asarray([-math.sin(node["normal_rad"]), math.cos(node["normal_rad"])])
    return math.acos(float(np.clip(abs(np.dot(vector, tangent)), 0.0, 1.0)))


def arc_span(angle: np.ndarray) -> float:
    ordered = np.sort(angle % (2.0 * math.pi))
    gaps = np.diff(np.concatenate([ordered, [ordered[0] + 2.0 * math.pi]]))
    return float(2.0 * math.pi - np.max(gaps))


def link_arcs(nodes: list[dict], config: dict, v19j) -> list[dict]:
    passed = [node for node in nodes if node["passes_step_score"]]
    if not passed:
        return []
    coordinates = np.asarray([[node["x_kpc"], node["y_kpc"]] for node in passed])
    pairs = cKDTree(coordinates).query_pairs(float(config["maximum_node_separation_kpc"]))
    union = UnionFind(len(passed))
    max_normal = math.radians(float(config["maximum_normal_difference_deg"]))
    max_tangent = math.radians(30.0)
    for first, second in sorted(pairs):
        if circular_difference(passed[first]["normal_rad"], passed[second]["normal_rad"]) > max_normal:
            continue
        if tangent_alignment(passed[first], passed[second]) > max_tangent:
            continue
        if tangent_alignment(passed[second], passed[first]) > max_tangent:
            continue
        union.union(first, second)
    components: dict[int, list[dict]] = {}
    for index, node in enumerate(passed):
        components.setdefault(union.find(index), []).append(node)
    arcs = []
    for component in components.values():
        if len(component) < int(config["minimum_component_nodes"]):
            continue
        x = np.asarray([node["x_kpc"] for node in component])
        y = np.asarray([node["y_kpc"] for node in component])
        circle = v19j.circle_fit(x, y)
        if circle is None:
            continue
        center_x, center_y, radius = circle
        if not (
            float(config["curvature_radius_kpc"][0])
            <= radius
            <= float(config["curvature_radius_kpc"][1])
        ):
            continue
        radial = np.hypot(x - center_x, y - center_y)
        rms_residual = float(np.sqrt(np.mean((radial - radius) ** 2)))
        if rms_residual > float(config["maximum_rms_radial_residual_kpc"]):
            continue
        angles = np.arctan2(y - center_y, x - center_x)
        span = arc_span(angles)
        length = radius * span
        if length < float(config["minimum_projected_length_kpc"]):
            continue
        weights = np.asarray([node["fit"]["delta_cash"] for node in component])
        total_weight = float(np.sum(weights))
        normal_value = np.sum(weights * np.exp(1j * np.asarray([node["normal_rad"] for node in component])))
        if total_weight <= 0.0 or abs(normal_value) <= np.finfo(float).eps:
            continue
        arcs.append(
            {
                "arc_id": 0,
                "node_count": len(component),
                "node_seed_ids": [int(node["seed_id"]) for node in component],
                "circle_center_x_kpc": float(center_x),
                "circle_center_y_kpc": float(center_y),
                "curvature_radius_kpc": float(radius),
                "rms_radial_residual_kpc": rms_residual,
                "angular_span_deg": math.degrees(span),
                "projected_length_kpc": float(length),
                "representative_x_kpc": float(np.sum(weights * x) / total_weight),
                "representative_y_kpc": float(np.sum(weights * y) / total_weight),
                "normal_faint_to_bright_deg": float(math.degrees(np.angle(normal_value) % (2.0 * math.pi))),
                "component_delta_cash": total_weight,
                "median_step_score_sigma": float(
                    np.median([node["fit"]["step_score_sigma"] for node in component])
                ),
                "median_density_compression": float(
                    np.median([node["fit"]["density_compression"] for node in component])
                ),
            }
        )
    arcs.sort(
        key=lambda row: (
            -row["component_delta_cash"],
            -row["projected_length_kpc"],
            row["representative_x_kpc"],
            row["representative_y_kpc"],
        )
    )
    for index, arc in enumerate(arcs, start=1):
        arc["arc_id"] = index
    return arcs


def analyze_arrays(
    counts: np.ndarray,
    background: np.ndarray,
    background_variance: np.ndarray,
    exposure: np.ndarray,
    mask: np.ndarray,
    v19j_result: dict,
    pixel_kpc: float,
    center_logical_x: float,
    center_logical_y: float,
    config: dict,
    v19j,
) -> tuple[list[dict], list[dict]]:
    seeds = local_maximum_seeds(
        v19j_result["candidate"],
        v19j_result["best_score"],
        v19j_result["best_angle"],
        v19j_result["best_scale"],
        pixel_kpc,
        10.0,
        center_logical_x,
        center_logical_y,
    )
    seed_fits = fit_seeds(
        seeds,
        counts,
        background,
        background_variance,
        exposure,
        mask,
        pixel_kpc,
        config,
    )
    arcs = link_arcs(seed_fits, config["arc_linking"], v19j)
    return seed_fits, arcs


def fixture_maps(kind: str, shape: tuple[int, int] = (384, 384)) -> tuple[np.ndarray, ...]:
    yy, xx = np.indices(shape, dtype=float)
    center_x = (shape[1] + 1.0) / 2.0
    center_y = (shape[0] + 1.0) / 2.0
    x = (xx + 1.0 - center_x) * 4.0
    y = (yy + 1.0 - center_y) * 4.0
    radius = np.hypot(x, y)
    if kind == "uniform":
        expected = np.full(shape, 25.0)
    elif kind == "linear":
        expected = 25.0 + 15.0 * x / np.max(np.abs(x))
    elif kind == "radial":
        expected = 5.0 + 40.0 / (1.0 + (radius / 180.0) ** 2)
    elif kind in {"step", "masked_step"}:
        expected = np.where(radius <= 300.0, 40.0, 10.0)
    else:
        raise ValueError(kind)
    counts = np.rint(expected)
    background = np.zeros(shape)
    background_variance = np.zeros(shape)
    exposure = np.full(shape, 1e8)
    mask = np.ones(shape, dtype=bool)
    if kind == "masked_step":
        mask[:, shape[1] // 2 :] = False
        counts = np.where(mask, counts, 0.0)
        exposure = np.where(mask, exposure, 0.0)
    return counts, background, background_variance, exposure, mask


def mandatory_fixtures(config: dict, v19j) -> dict:
    v19j_config = common.load_json(ROOT / config["parents"]["v19j_config"])
    results = {}
    for kind in ("uniform", "linear", "radial", "step", "masked_step"):
        counts, background, background_variance, exposure, mask = fixture_maps(kind)
        front = v19j.detect_fronts(
            counts,
            background,
            background_variance,
            exposure,
            mask,
            4.0,
            v19j_config,
            192.5,
            192.5,
        )
        seeds, arcs = analyze_arrays(
            counts,
            background,
            background_variance,
            exposure,
            mask,
            front,
            4.0,
            192.5,
            192.5,
            config,
            v19j,
        )
        record = {
            "v19j_candidate_pixels": int(np.sum(front["candidate"])),
            "v19k_seed_count": len(seeds),
            "passing_seed_count": sum(seed["passes_step_score"] for seed in seeds),
            "retained_arc_count": len(arcs),
        }
        if kind == "step" and arcs:
            distances = []
            compressions = []
            for seed in seeds:
                if seed["passes_step_score"]:
                    distances.append(abs(math.hypot(seed["x_kpc"], seed["y_kpc"]) - 300.0))
                    compressions.append(seed["fit"]["density_compression"])
            record["median_passing_seed_distance_to_circle_kpc"] = float(np.median(distances))
            record["median_passing_seed_density_compression"] = float(np.median(compressions))
            record["passed"] = bool(
                record["median_passing_seed_distance_to_circle_kpc"] <= 16.0
                and 1.5 <= record["median_passing_seed_density_compression"] <= 2.5
            )
        else:
            record["passed"] = len(arcs) == (1 if kind == "step" else 0)
        results[kind] = record
        print(f"fixture {kind}: {record}", flush=True)
    return results


def load_science_inputs(config: dict, v19j_report: dict, cluster: str, v19j) -> tuple:
    map_report = common.load_json(ROOT / config["parents"]["source_map_report"])
    map_record = next(row for row in map_report["clusters"] if row["cluster"] == cluster)
    paths = v19j.product_paths(map_record)
    counts = v19j.load_array(paths["soft_counts"])
    background = v19j.load_array(paths["soft_scaled_background"])
    background_variance = v19j.load_array(paths["soft_background_variance"])
    exposure = v19j.load_array(paths["soft_exposure"])
    mask = v19j.load_array(paths["analysis_mask"]) > 0.5
    v19j_record = next(row for row in v19j_report["clusters"] if row["cluster"] == cluster)
    product = next(row for row in v19j_record["products"] if row["kind"] == "ridge_arrays")
    product_path = ROOT / product["path"]
    if common.sha256(product_path) != product["sha256"]:
        raise RuntimeError(f"{cluster} V19J ridge product hash mismatch")
    with np.load(product_path) as payload:
        front = {
            "best_scale": np.asarray(payload["best_scale_index"]),
            "best_score": np.asarray(payload["best_score_sigma"]),
            "best_angle": np.asarray(payload["best_normal_angle_rad"]),
            "candidate": np.asarray(payload["nonmaximum_candidate"], dtype=bool),
        }
    pixel_kpc = (
        float(common.load_json(ROOT / config["parents"]["v19j_config"])["physical_grid"]["pixel_scale_arcsec"])
        * float(
            common.load_json(ROOT / config["parents"]["v19j_config"])["physical_grid"][
                "cluster_kpc_per_arcsec"
            ][cluster]
        )
    )
    return (
        counts,
        background,
        background_variance,
        exposure,
        mask,
        front,
        pixel_kpc,
        float(map_record["final_center"]["logicalx"]),
        float(map_record["final_center"]["logicaly"]),
    )


def render_diagnostic(
    path: Path,
    counts: np.ndarray,
    background: np.ndarray,
    exposure: np.ndarray,
    mask: np.ndarray,
    seeds: list[dict],
    arcs: list[dict],
    cluster: str,
) -> None:
    rate = np.divide(
        counts - background,
        exposure,
        out=np.zeros_like(counts),
        where=exposure > 0.0,
    )
    rate[~mask] = np.nan
    score_map = np.full(counts.shape, np.nan)
    for seed in seeds:
        if seed["fit"].get("success", False):
            score_map[seed["row"], seed["column"]] = seed["fit"]["step_score_sigma"]
    figure, axes = plt.subplots(1, 2, figsize=(11, 5), constrained_layout=True)
    image = axes[0].imshow(rate, origin="lower", cmap="viridis")
    figure.colorbar(image, ax=axes[0], fraction=0.046)
    axes[0].set_title("Background-subtracted soft rate")
    image = axes[1].imshow(score_map, origin="lower", cmap="magma", vmin=0.0, vmax=10.0)
    figure.colorbar(image, ax=axes[1], fraction=0.046)
    axes[1].set_title("V19K seed discontinuity score")
    arc_seed_ids = {seed_id for arc in arcs for seed_id in arc["node_seed_ids"]}
    for axis in axes:
        for seed in seeds:
            if seed["seed_id"] in arc_seed_ids:
                axis.scatter(seed["column"], seed["row"], s=9, facecolors="none", edgecolors="cyan")
    figure.suptitle(f"{cluster}: smooth-null front likelihood")
    figure.savefig(path, dpi=160)
    plt.close(figure)


def run_cluster(config: dict, v19j_report: dict, cluster: str, output: Path, v19j) -> dict:
    values = load_science_inputs(config, v19j_report, cluster, v19j)
    counts, background, background_variance, exposure, mask, front, pixel_kpc, center_x, center_y = values
    seeds, arcs = analyze_arrays(
        counts,
        background,
        background_variance,
        exposure,
        mask,
        front,
        pixel_kpc,
        center_x,
        center_y,
        config,
        v19j,
    )
    stem = cluster.lower()
    seed_path = output / f"{stem}_seed_fits.json"
    seed_path.write_text(json.dumps(seeds, separators=(",", ":")) + "\n", encoding="utf-8")
    arc_path = output / f"{stem}_arc_catalog.json"
    arc_path.write_text(json.dumps(arcs, indent=2) + "\n", encoding="utf-8")
    figure_path = output / f"{stem}_diagnostic.png"
    render_diagnostic(figure_path, counts, background, exposure, mask, seeds, arcs, cluster)
    products = []
    for kind, path in (
        ("seed_profile_likelihoods", seed_path),
        ("retained_arc_catalog", arc_path),
        ("post_likelihood_diagnostic", figure_path),
    ):
        products.append(
            {
                "kind": kind,
                "path": path.relative_to(ROOT).as_posix(),
                "sha256": common.sha256(path),
                "bytes": path.stat().st_size,
            }
        )
    attempted = [seed for seed in seeds if seed["profile_valid"]]
    failed = [seed for seed in attempted if not seed["fit"].get("success", False)]
    failed_fraction = len(failed) / len(attempted) if attempted else 1.0
    gates = {
        "all_seed_fit_likelihoods_finite_or_explicitly_failed": all(
            (not seed["profile_valid"])
            or (seed["fit"].get("success", False) and np.isfinite(seed["fit"]["delta_cash"]))
            or (not seed["fit"].get("success", False))
            for seed in seeds
        ),
        "maximum_failed_seed_fraction": failed_fraction
        <= float(config["science_gates"]["maximum_failed_seed_fraction"]),
        "minimum_retained_arcs": len(arcs)
        >= int(config["science_gates"]["minimum_retained_arcs_per_cluster"]),
        "every_retained_arc_passes_fixed_geometry": all(
            arc["node_count"] >= int(config["arc_linking"]["minimum_component_nodes"])
            and arc["projected_length_kpc"]
            >= float(config["arc_linking"]["minimum_projected_length_kpc"])
            and arc["rms_radial_residual_kpc"]
            <= float(config["arc_linking"]["maximum_rms_radial_residual_kpc"])
            for arc in arcs
        ),
    }
    print(
        f"{cluster}: {len(seeds)} seeds, {len(attempted)} valid profiles, "
        f"{sum(seed['passes_step_score'] for seed in seeds)} passing seeds, {len(arcs)} arcs",
        flush=True,
    )
    return {
        "cluster": cluster,
        "pixel_scale_kpc": pixel_kpc,
        "seed_count": len(seeds),
        "profile_valid_seed_count": len(attempted),
        "successful_seed_fit_count": len(attempted) - len(failed),
        "failed_seed_fit_count": len(failed),
        "failed_seed_fit_fraction": failed_fraction,
        "passing_step_seed_count": sum(seed["passes_step_score"] for seed in seeds),
        "retained_arc_count": len(arcs),
        "arcs": arcs,
        "products": products,
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
    v19j_report = common.load_json(ROOT / config["parents"]["v19j_report"])
    visual_audit = common.load_json(ROOT / config["parents"]["v19j_visual_audit"])
    if visual_audit["failure_class"] != "statistic and topology implementation failure":
        raise RuntimeError("the inherited V19J failure class changed")
    if visual_audit["scientific_threshold_changed"] is not False:
        raise RuntimeError("the inherited V19J audit changed a scientific threshold")
    v19j = v19j_module()
    fixtures = mandatory_fixtures(config, v19j)
    if not all(row["passed"] for row in fixtures.values()):
        raise RuntimeError(f"a mandatory pre-science V19K fixture failed: {fixtures}")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    clusters = [
        run_cluster(config, v19j_report, cluster, output, v19j)
        for cluster in config["sample"]["clusters"]
    ]
    failed = [row["cluster"] for row in clusters if not all(row["gates"].values())]
    report = {
        "status": (
            "both_clusters_passed_frozen_v19k_smooth_null_front_map_gate"
            if not failed
            else "frozen_v19k_smooth_null_front_map_gate_failed"
        ),
        "protocol_version": config["protocol_version"],
        "generated_utc": datetime.now(UTC).isoformat(),
        "config_sha256": common.sha256(config_path),
        "runner_sha256": common.sha256(Path(__file__).resolve()),
        "mandatory_pre_science_fixtures": fixtures,
        "failed_clusters": failed,
        "clusters": clusters,
        "final_broken_power_law_profile_fit_run": False,
        "parametric_bootstrap_run": False,
        "spectrum_or_response_constructed": False,
        "shock_classification_run": False,
        "post_hash_visual_audit_run": False,
        "published_front_coordinate_used": False,
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
