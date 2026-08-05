#!/usr/bin/env python3
"""Run the frozen V19J target-blind multiscale X-ray front search."""

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
from astropy.io import fits
from scipy import ndimage, signal
from skimage.morphology import closing, disk, skeletonize

ROOT = common.ROOT
DEFAULT_CONFIG = ROOT / "configs" / "sigma_v19j_automated_front_implementation.json"
DEFAULT_OUTPUT = ROOT / "results" / "sigma_v19j_automated_fronts"


def stable_seed(base: int, label: str) -> int:
    payload = f"{base}:{label}:v19j-front".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big")


def load_array(path: Path) -> np.ndarray:
    with fits.open(path, memmap=False) as handle:
        value = np.asarray(handle[0].data, dtype=float)
    if value.ndim != 2 or not np.all(np.isfinite(value)):
        raise RuntimeError(f"invalid two-dimensional FITS array: {path}")
    return value


def product_paths(cluster_report: dict) -> dict[str, Path]:
    products = {}
    for product in cluster_report["frozen_snapshot"]["products"]:
        path = ROOT / product["relative_path"]
        if path.stat().st_size != int(product["bytes"]):
            raise RuntimeError(f"map product size changed: {path}")
        if common.sha256(path) != product["sha256"]:
            raise RuntimeError(f"map product hash changed: {path}")
        products[product["role"]] = path
    required = {
        "soft_counts",
        "soft_scaled_background",
        "soft_background_variance",
        "soft_exposure",
        "analysis_mask",
    }
    if not required.issubset(products):
        raise RuntimeError("frozen source-map snapshot is incomplete")
    return products


def half_gaussian_kernel(
    sigma_pixels: float, angle: float, truncation_sigma: float
) -> tuple[np.ndarray, np.ndarray]:
    radius = math.ceil(truncation_sigma * sigma_pixels)
    coordinate = np.arange(-radius, radius + 1, dtype=float)
    yy, xx = np.meshgrid(coordinate, coordinate, indexing="ij")
    radial2 = xx**2 + yy**2
    gaussian = np.exp(-radial2 / (2.0 * sigma_pixels**2))
    gaussian[radial2 > (truncation_sigma * sigma_pixels) ** 2] = 0.0
    projection = xx * math.cos(angle) + yy * math.sin(angle)
    positive = np.where(projection > 1e-12, gaussian, 0.0)
    negative = np.where(projection < -1e-12, gaussian, 0.0)
    if np.sum(positive) <= 0.0 or np.sum(negative) <= 0.0:
        raise RuntimeError("a half-Gaussian kernel is empty")
    return positive, negative


def convolve(array: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    return signal.fftconvolve(array, kernel, mode="same")


def wrap_angle(value: np.ndarray) -> np.ndarray:
    return np.mod(value, 2.0 * math.pi)


def orientation_indices(angle: np.ndarray, count: int) -> np.ndarray:
    step = 2.0 * math.pi / count
    return np.floor(wrap_angle(angle) / step + 0.5).astype(int) % count


def scale_score(
    counts: np.ndarray,
    background: np.ndarray,
    background_variance: np.ndarray,
    exposure: np.ndarray,
    mask: np.ndarray,
    sigma_pixels: float,
    orientation_count: int,
    truncation_sigma: float,
    minimum_valid_fraction: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    net = np.where(mask, counts - background, 0.0)
    exposure_masked = np.where(mask, exposure, 0.0)
    variance_counts = np.where(mask, np.maximum(0.0, counts + background_variance), 0.0)
    smooth_net = ndimage.gaussian_filter(net, sigma_pixels, mode="constant", cval=0.0)
    smooth_exposure = ndimage.gaussian_filter(
        exposure_masked, sigma_pixels, mode="constant", cval=0.0
    )
    smooth_rate = np.divide(
        smooth_net,
        smooth_exposure,
        out=np.zeros_like(smooth_net),
        where=smooth_exposure > 0.0,
    )
    gradient_y, gradient_x = np.gradient(smooth_rate)
    gradient_angle = np.arctan2(gradient_y, gradient_x)
    assigned = orientation_indices(gradient_angle, orientation_count)
    score = np.full(counts.shape, np.nan, dtype=float)
    selected_valid = np.zeros(counts.shape, dtype=bool)
    selected_rate_difference = np.full(counts.shape, np.nan, dtype=float)
    for orientation in range(orientation_count):
        choose = assigned == orientation
        if not np.any(choose):
            continue
        angle = 2.0 * math.pi * orientation / orientation_count
        positive, negative = half_gaussian_kernel(sigma_pixels, angle, truncation_sigma)
        results = []
        for kernel in (positive, negative):
            net_sum = convolve(net, kernel)
            exposure_sum = convolve(exposure_masked, kernel)
            variance_sum = convolve(variance_counts, kernel**2)
            valid_fraction = convolve(mask.astype(float), kernel) / np.sum(kernel)
            rate = np.divide(
                net_sum,
                exposure_sum,
                out=np.full_like(net_sum, np.nan),
                where=exposure_sum > 0.0,
            )
            variance = np.divide(
                variance_sum,
                exposure_sum**2,
                out=np.full_like(variance_sum, np.nan),
                where=exposure_sum > 0.0,
            )
            results.append((rate, variance, valid_fraction))
        positive_result, negative_result = results
        difference = positive_result[0] - negative_result[0]
        variance = positive_result[1] + negative_result[1]
        valid = (
            choose
            & mask
            & (positive_result[2] >= minimum_valid_fraction)
            & (negative_result[2] >= minimum_valid_fraction)
            & np.isfinite(difference)
            & np.isfinite(variance)
            & (variance > 0.0)
        )
        score[valid] = np.abs(difference[valid]) / np.sqrt(variance[valid])
        selected_valid[valid] = True
        selected_rate_difference[valid] = difference[valid]
    return score, gradient_angle, selected_valid, selected_rate_difference


def normal_nonmaximum(score: np.ndarray, angle: np.ndarray) -> np.ndarray:
    yy, xx = np.indices(score.shape, dtype=float)
    cosine = np.cos(angle)
    sine = np.sin(angle)
    safe_score = np.where(np.isfinite(score), score, -np.inf)
    positive = ndimage.map_coordinates(
        safe_score, [yy + sine, xx + cosine], order=1, mode="constant", cval=-np.inf
    )
    negative = ndimage.map_coordinates(
        safe_score, [yy - sine, xx - cosine], order=1, mode="constant", cval=-np.inf
    )
    return np.isfinite(score) & (score >= positive) & (score >= negative)


def skeleton_length_pixels(component: np.ndarray) -> float:
    length = 0.0
    for dy, dx, weight in (
        (0, 1, 1.0),
        (1, 0, 1.0),
        (1, 1, math.sqrt(2.0)),
        (1, -1, math.sqrt(2.0)),
    ):
        source_y = slice(max(0, -dy), component.shape[0] - max(0, dy))
        target_y = slice(max(0, dy), component.shape[0] - max(0, -dy))
        source_x = slice(max(0, -dx), component.shape[1] - max(0, dx))
        target_x = slice(max(0, dx), component.shape[1] - max(0, -dx))
        length += weight * float(
            np.sum(component[source_y, source_x] & component[target_y, target_x])
        )
    return length


def circle_fit(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float] | None:
    design = np.column_stack([2.0 * x, 2.0 * y, np.ones(len(x))])
    target = x**2 + y**2
    solution, _, rank, _ = np.linalg.lstsq(design, target, rcond=None)
    if rank < 3:
        return None
    center_x, center_y, constant = solution
    radius2 = constant + center_x**2 + center_y**2
    if not np.isfinite(radius2) or radius2 <= 0.0:
        return None
    return float(center_x), float(center_y), math.sqrt(float(radius2))


def circular_mean(angle: np.ndarray, weight: np.ndarray) -> float:
    value = np.sum(weight * np.exp(1j * angle))
    if abs(value) <= np.finfo(float).eps:
        return float("nan")
    return float(np.angle(value) % (2.0 * math.pi))


def link_ridges(
    candidate: np.ndarray,
    score: np.ndarray,
    angle: np.ndarray,
    scale_index: np.ndarray,
    mask: np.ndarray,
    pixel_kpc: float,
    config: dict,
    center_logical_x: float,
    center_logical_y: float,
) -> tuple[np.ndarray, list[dict]]:
    gap = float(config["maximum_link_gap_kpc"])
    closing_radius = math.floor((gap / 2.0) / pixel_kpc)
    closed = closing(candidate, disk(max(0, closing_radius)))
    skeleton = skeletonize(closed) & mask
    labels, count = ndimage.label(skeleton, structure=np.ones((3, 3), dtype=int))
    ridges = []
    for label_index in range(1, count + 1):
        component = labels == label_index
        row, column = np.nonzero(component)
        if len(row) < 3:
            continue
        length = skeleton_length_pixels(component) * pixel_kpc
        if length < float(config["minimum_projected_length_kpc"]):
            continue
        valid_fraction = float(np.mean(mask[component]))
        if valid_fraction < float(config["minimum_component_valid_fraction"]):
            continue
        x = (column + 1.0 - center_logical_x) * pixel_kpc
        y = (row + 1.0 - center_logical_y) * pixel_kpc
        circle = circle_fit(x, y)
        if circle is None:
            continue
        circle_x, circle_y, radius = circle
        if not (
            float(config["minimum_curvature_radius_kpc"])
            <= radius
            <= float(config["maximum_curvature_radius_kpc"])
        ):
            continue
        component_score = np.where(np.isfinite(score[component]), score[component], 0.0)
        weights = component_score**2
        if np.sum(weights) <= 0.0:
            continue
        representative_x = float(np.sum(weights * x) / np.sum(weights))
        representative_y = float(np.sum(weights * y) / np.sum(weights))
        representative_angle = circular_mean(angle[component], weights)
        if not np.isfinite(representative_angle):
            continue
        ridges.append(
            {
                "ridge_id": 0,
                "skeleton_pixels": len(row),
                "projected_length_kpc": float(length),
                "valid_fraction": valid_fraction,
                "curvature_center_x_kpc": circle_x,
                "curvature_center_y_kpc": circle_y,
                "curvature_radius_kpc": float(radius),
                "representative_x_kpc": representative_x,
                "representative_y_kpc": representative_y,
                "normal_faint_to_bright_deg": math.degrees(representative_angle),
                "maximum_score_sigma": float(np.nanmax(score[component])),
                "median_score_sigma": float(np.nanmedian(score[component])),
                "integrated_half_z2": float(0.5 * np.sum(weights)),
                "scale_pixel_counts_8_16_32_64_kpc": [
                    int(np.sum(scale_index[component] == index)) for index in range(4)
                ],
                "row_min_max": [int(np.min(row)), int(np.max(row))],
                "column_min_max": [int(np.min(column)), int(np.max(column))],
            }
        )
    ridges.sort(key=lambda row: (-row["integrated_half_z2"], -row["projected_length_kpc"]))
    for index, ridge in enumerate(ridges, start=1):
        ridge["ridge_id"] = index
    return skeleton, ridges


def detect_fronts(
    counts: np.ndarray,
    background: np.ndarray,
    background_variance: np.ndarray,
    exposure: np.ndarray,
    mask: np.ndarray,
    pixel_kpc: float,
    config: dict,
    center_logical_x: float,
    center_logical_y: float,
) -> dict:
    physical = config["physical_grid"]
    score_config = config["poisson_step_score"]
    scores = []
    angles = []
    valids = []
    differences = []
    nonmaximum = []
    for scale_kpc in physical["candidate_gaussian_sigma_kpc"]:
        score, angle, valid, difference = scale_score(
            counts,
            background,
            background_variance,
            exposure,
            mask,
            float(scale_kpc) / pixel_kpc,
            int(config["local_normal"]["directed_orientation_bins"]),
            float(physical["kernel_truncation_sigma"]),
            float(score_config["minimum_side_valid_fraction"]),
        )
        scores.append(score)
        angles.append(angle)
        valids.append(valid)
        differences.append(difference)
        nonmaximum.append(normal_nonmaximum(score, angle))
    score_cube = np.asarray(scores)
    angle_cube = np.asarray(angles)
    valid_cube = np.asarray(valids)
    difference_cube = np.asarray(differences)
    safe_score = np.where(np.isfinite(score_cube), score_cube, -np.inf)
    best_scale = np.argmax(safe_score, axis=0)
    best_score = np.take_along_axis(score_cube, best_scale[None, ...], axis=0)[0]
    best_angle = np.take_along_axis(angle_cube, best_scale[None, ...], axis=0)[0]
    best_nonmaximum = np.take_along_axis(
        np.asarray(nonmaximum), best_scale[None, ...], axis=0
    )[0]
    candidate = (
        mask
        & np.isfinite(best_score)
        & (best_score >= float(score_config["minimum_single_scale_score_sigma"]))
        & best_nonmaximum
    )
    skeleton, ridges = link_ridges(
        candidate,
        best_score,
        best_angle,
        best_scale,
        mask,
        pixel_kpc,
        config["ridge_linking"],
        center_logical_x,
        center_logical_y,
    )
    return {
        "score_cube": score_cube,
        "angle_cube": angle_cube,
        "valid_cube": valid_cube,
        "difference_cube": difference_cube,
        "best_scale": best_scale,
        "best_score": best_score,
        "best_angle": best_angle,
        "candidate": candidate,
        "skeleton": skeleton,
        "ridges": ridges,
    }


def synthetic_fixtures(config: dict) -> dict:
    shape = (256, 256)
    yy, xx = np.indices(shape)
    exposure = np.ones(shape)
    mask = np.ones(shape, dtype=bool)
    background = np.zeros(shape)
    background_variance = np.zeros(shape)
    pixel_kpc = 1.0
    center = 128.5

    uniform_counts = np.full(shape, 40.0)
    uniform = detect_fronts(
        uniform_counts,
        background,
        background_variance,
        exposure,
        mask,
        pixel_kpc,
        config,
        center,
        center,
    )

    radius = np.hypot(xx + 1.0 - center, yy + 1.0 - center)
    curved_counts = np.where(radius <= 70.0, 400.0, 25.0)
    curved = detect_fronts(
        curved_counts,
        background,
        background_variance,
        exposure,
        mask,
        pixel_kpc,
        config,
        center,
        center,
    )
    nearest = (
        min(
            abs(ridge["curvature_radius_kpc"] - 70.0)
            for ridge in curved["ridges"]
        )
        if curved["ridges"]
        else float("inf")
    )

    masked = mask.copy()
    masked[:, 128:] = False
    masked_counts = np.where(masked, np.where(xx < 128, 400.0, 25.0), 0.0)
    masked_exposure = np.where(masked, 1.0, 0.0)
    masked_result = detect_fronts(
        masked_counts,
        background,
        background_variance,
        masked_exposure,
        masked,
        pixel_kpc,
        config,
        center,
        center,
    )
    return {
        "uniform_field_retained_ridges": len(uniform["ridges"]),
        "uniform_field_passed": len(uniform["ridges"]) == 0,
        "curved_step_retained_ridges": len(curved["ridges"]),
        "curved_step_nearest_radius_error_kpc": float(nearest),
        "curved_step_passed": bool(nearest <= 8.0),
        "masked_edge_retained_ridges": len(masked_result["ridges"]),
        "masked_edge_passed": len(masked_result["ridges"]) == 0,
    }


def render_diagnostic(
    path: Path,
    counts: np.ndarray,
    background: np.ndarray,
    exposure: np.ndarray,
    mask: np.ndarray,
    result: dict,
    cluster: str,
) -> None:
    rate = np.divide(
        counts - background,
        exposure,
        out=np.zeros_like(counts),
        where=exposure > 0.0,
    )
    rate[~mask] = np.nan
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.8), constrained_layout=True)
    panels = (
        (rate, "Background-subtracted soft rate"),
        (result["best_score"], "Maximum Poisson step score"),
        (result["candidate"].astype(float), "Nonmaximum ridge candidates"),
    )
    for axis, (image, title) in zip(axes, panels):
        shown = axis.imshow(image, origin="lower", cmap="viridis")
        row, column = np.nonzero(result["skeleton"])
        axis.scatter(column, row, s=1.0, color="white", alpha=0.8)
        axis.set_title(title)
        figure.colorbar(shown, ax=axis, fraction=0.046)
    figure.suptitle(f"{cluster}: frozen target-blind V19J front search")
    figure.savefig(path, dpi=150)
    plt.close(figure)


def run_cluster(config: dict, map_report: dict, cluster: str, output: Path) -> dict:
    record = next(row for row in map_report["clusters"] if row["cluster"] == cluster)
    paths = product_paths(record)
    counts = load_array(paths["soft_counts"])
    background = load_array(paths["soft_scaled_background"])
    background_variance = load_array(paths["soft_background_variance"])
    exposure = load_array(paths["soft_exposure"])
    mask = load_array(paths["analysis_mask"]) > 0.5
    shapes = {value.shape for value in (counts, background, background_variance, exposure, mask)}
    if len(shapes) != 1:
        raise RuntimeError(f"{cluster} frozen map shapes differ")
    pixel_kpc = (
        float(config["physical_grid"]["pixel_scale_arcsec"])
        * float(config["physical_grid"]["cluster_kpc_per_arcsec"][cluster])
    )
    result = detect_fronts(
        counts,
        background,
        background_variance,
        exposure,
        mask,
        pixel_kpc,
        config,
        float(record["final_center"]["logicalx"]),
        float(record["final_center"]["logicaly"]),
    )
    stem = cluster.lower()
    score_path = output / f"{stem}_score_cube.npz"
    np.savez_compressed(
        score_path,
        score_sigma=result["score_cube"],
        normal_angle_rad=result["angle_cube"],
        valid=result["valid_cube"],
        signed_rate_difference=result["difference_cube"],
    )
    ridge_path = output / f"{stem}_ridge_products.npz"
    np.savez_compressed(
        ridge_path,
        best_scale_index=result["best_scale"],
        best_score_sigma=result["best_score"],
        best_normal_angle_rad=result["best_angle"],
        nonmaximum_candidate=result["candidate"],
        linked_skeleton=result["skeleton"],
    )
    catalog_path = output / f"{stem}_ridge_catalog.json"
    catalog_path.write_text(json.dumps(result["ridges"], indent=2) + "\n", encoding="utf-8")
    figure_path = output / f"{stem}_diagnostic.png"
    render_diagnostic(figure_path, counts, background, exposure, mask, result, cluster)
    products = []
    for kind, path in (
        ("scale_score_cube", score_path),
        ("ridge_arrays", ridge_path),
        ("retained_ridge_catalog", catalog_path),
        ("automated_diagnostic", figure_path),
    ):
        products.append(
            {
                "kind": kind,
                "path": path.relative_to(ROOT).as_posix(),
                "sha256": common.sha256(path),
                "bytes": path.stat().st_size,
            }
        )
    finite_or_invalid = bool(
        np.all(np.isfinite(result["score_cube"]) | ~result["valid_cube"])
    )
    gates = {
        "all_four_scale_score_arrays_finite_or_explicitly_invalid": finite_or_invalid,
        "minimum_retained_ridges": len(result["ridges"])
        >= int(config["gates"]["minimum_retained_ridges_per_cluster"]),
        "every_retained_ridge_length_and_curvature_inside_frozen_bounds": all(
            float(config["ridge_linking"]["minimum_projected_length_kpc"])
            <= ridge["projected_length_kpc"]
            and float(config["ridge_linking"]["minimum_curvature_radius_kpc"])
            <= ridge["curvature_radius_kpc"]
            <= float(config["ridge_linking"]["maximum_curvature_radius_kpc"])
            for ridge in result["ridges"]
        ),
    }
    print(
        f"{cluster}: {int(np.sum(result['candidate']))} candidate pixels, "
        f"{int(np.sum(result['skeleton']))} skeleton pixels, "
        f"{len(result['ridges'])} retained ridges",
        flush=True,
    )
    return {
        "cluster": cluster,
        "shape_y_x": list(counts.shape),
        "pixel_scale_kpc": pixel_kpc,
        "candidate_pixels": int(np.sum(result["candidate"])),
        "linked_skeleton_pixels": int(np.sum(result["skeleton"])),
        "retained_ridge_count": len(result["ridges"]),
        "ridges": result["ridges"],
        "valid_score_pixels_by_scale": [
            int(np.sum(result["valid_cube"][index])) for index in range(4)
        ],
        "pixels_at_or_above_5sigma_by_scale": [
            int(np.sum(result["score_cube"][index] >= 5.0)) for index in range(4)
        ],
        "maximum_score_sigma_by_scale": [
            float(np.nanmax(result["score_cube"][index])) for index in range(4)
        ],
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
    map_report = common.load_json(ROOT / config["parents"]["source_map_report"])
    member_report = common.load_json(ROOT / config["parents"]["v19i_member_failure_report"])
    if map_report["status"] != "both_clusters_passed_frozen_v19h_source_map_gate":
        raise RuntimeError("the inherited source-map gate did not pass")
    if map_report["lensing_target_opened"] is not False:
        raise RuntimeError("the inherited source-map stage opened a lensing target")
    if member_report["status"] != "frozen_v19i_continuous_member_field_gate_failed":
        raise RuntimeError("the declared independent member-field outcome changed")
    fixtures = synthetic_fixtures(config)
    if not all(value for key, value in fixtures.items() if key.endswith("_passed")):
        raise RuntimeError(f"a frozen synthetic front fixture failed: {fixtures}")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    clusters = [run_cluster(config, map_report, cluster, output) for cluster in config["sample"]["clusters"]]
    failed = [row["cluster"] for row in clusters if not all(row["gates"].values())]
    report = {
        "status": (
            "both_clusters_passed_frozen_v19j_automated_front_map_gate"
            if not failed
            else "frozen_v19j_automated_front_map_gate_failed"
        ),
        "protocol_version": config["protocol_version"],
        "generated_utc": datetime.now(UTC).isoformat(),
        "config_sha256": common.sha256(config_path),
        "runner_sha256": common.sha256(Path(__file__).resolve()),
        "synthetic_fixtures": fixtures,
        "failed_clusters": failed,
        "clusters": clusters,
        "profile_fit_run": False,
        "parametric_bootstrap_run": False,
        "shock_classification_run": False,
        "spectrum_or_response_constructed": False,
        "registered_science_image_visually_inspected": False,
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
