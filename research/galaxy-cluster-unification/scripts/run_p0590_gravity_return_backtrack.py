#!/usr/bin/env python3
"""Fit a frozen conservative return kernel and backtrack apparent-dark peaks."""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.wcs import WCS
from scipy.ndimage import gaussian_filter, map_coordinates, maximum_filter
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from voidscreen.gravity_return import (  # noqa: E402
    jensen_shannon_divergence,
    normalized_ring_kernel,
    routed_arrival_map,
    semicircle_arc_geometry,
    source_origin_probabilities,
    transition_radius_arcsec,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def coordinate_grid(protocol: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    grid = protocol["grid"]
    axis = np.linspace(-grid["half_width_arcsec"], grid["half_width_arcsec"], grid["pixels_per_axis"])
    xx, yy = np.meshgrid(axis, axis, indexing="xy")
    score_mask = np.hypot(xx, yy) <= float(grid["score_radius_arcsec"])
    return axis, xx, yy, score_mask


def source_surface(sources: pd.DataFrame, axis: np.ndarray, smoothing_arcsec: float) -> np.ndarray:
    spacing = float(axis[1] - axis[0])
    edges = np.r_[axis - 0.5 * spacing, axis[-1] + 0.5 * spacing]
    surface, _, _ = np.histogram2d(
        sources.y_arcsec,
        sources.x_arcsec,
        bins=(edges, edges),
        weights=sources.mass_msun,
    )
    surface = gaussian_filter(surface, float(smoothing_arcsec) / spacing, mode="constant")
    return np.maximum(surface, 0.0)


def map_path(model: dict) -> Path:
    return ROOT / "data/raw/p0590_macs0416_kappa" / Path(model["url"]).name


def load_reprojected_map(model: dict, protocol: dict, xx: np.ndarray, yy: np.ndarray, mask: np.ndarray) -> np.ndarray:
    path = map_path(model)
    reference_ra = 64.0381416667
    reference_dec = -24.0674722222
    cosine = math.cos(math.radians(reference_dec))
    ra = reference_ra - xx / (3600.0 * cosine)
    dec = reference_dec + yy / 3600.0
    with fits.open(path, memmap=True) as hdul:
        data = np.asarray(hdul[0].data, dtype=float)
        wcs = WCS(hdul[0].header)
        pixel_x, pixel_y = wcs.world_to_pixel_values(ra, dec)
        sampled = map_coordinates(
            data,
            np.vstack([pixel_y.ravel(), pixel_x.ravel()]),
            order=1,
            mode="constant",
            cval=np.nan,
            prefilter=False,
        ).reshape(xx.shape)
    preprocessing = protocol["target_preprocessing"]
    finite_score = mask & np.isfinite(sampled)
    if np.count_nonzero(finite_score) < 0.8 * np.count_nonzero(mask):
        raise RuntimeError(f"{model['model_id']} does not cover enough of the score aperture")
    annulus = (
        (np.hypot(xx, yy) >= preprocessing["background_annulus_arcsec"][0])
        & (np.hypot(xx, yy) <= preprocessing["background_annulus_arcsec"][1])
        & np.isfinite(sampled)
    )
    background = float(np.nanmedian(sampled[annulus]))
    values = np.maximum(np.nan_to_num(sampled - background, nan=0.0), 0.0)
    cap = float(np.percentile(values[finite_score], preprocessing["winsorize_inside_score_percentile"]))
    values = np.minimum(values, cap)
    spacing = float(protocol["grid"]["half_width_arcsec"] * 2.0 / (protocol["grid"]["pixels_per_axis"] - 1))
    values = gaussian_filter(values, preprocessing["smoothing_arcsec"] / spacing, mode="constant")
    values[~mask] = 0.0
    total = float(np.sum(values[mask]))
    if total <= 0.0:
        raise RuntimeError(f"{model['model_id']} target is empty after preprocessing")
    return values / total


def pearson_map(first: np.ndarray, second: np.ndarray, mask: np.ndarray) -> float:
    return float(np.corrcoef(first[mask], second[mask])[0, 1])


def positive_dark_residual(target: np.ndarray, local: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, float]:
    denominator = float(np.sum(local[mask] ** 2))
    scale = max(float(np.sum(target[mask] * local[mask]) / max(denominator, np.finfo(float).tiny)), 0.0)
    residual = np.maximum(target - scale * local, 0.0)
    residual[~mask] = 0.0
    if residual.sum() > 0.0:
        residual /= residual.sum()
    return residual, scale


def find_peaks(surface: np.ndarray, axis: np.ndarray, mask: np.ndarray, count: int, separation_arcsec: float) -> list[tuple[float, float, float]]:
    spacing = float(axis[1] - axis[0])
    size = max(3, int(round(separation_arcsec / spacing)))
    if size % 2 == 0:
        size += 1
    maxima = (surface == maximum_filter(surface, size=size, mode="constant")) & mask & (surface > 0.0)
    indices = np.argwhere(maxima)
    order = np.argsort(surface[maxima])[::-1]
    result = []
    for position in indices[order]:
        iy, ix = map(int, position)
        x, y = float(axis[ix]), float(axis[iy])
        if all(math.hypot(x - px, y - py) >= separation_arcsec for px, py, _ in result):
            result.append((x, y, float(surface[iy, ix])))
        if len(result) == count:
            break
    return result


def origin_group(row: pd.Series, north: tuple[float, float], south: tuple[float, float]) -> str:
    if row.component == "member_star":
        return f"member_{row.source_id}"
    if row.component == "bcg_stars":
        dn = math.hypot(row.x_arcsec - north[0], row.y_arcsec - north[1])
        ds = math.hypot(row.x_arcsec - south[0], row.y_arcsec - south[1])
        return "BCG_N" if dn <= ds else "BCG_S"
    return str(row.component)


def main() -> None:
    protocol_path = ROOT / "configs/p0590_gravity_return_backtrack_protocol.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    sources_path = ROOT / protocol["source"]["nominal_baryons"]
    sources = pd.read_csv(sources_path)
    axis, xx, yy, score_mask = coordinate_grid(protocol)
    local_surface = source_surface(sources, axis, protocol["source"]["source_smoothing_arcsec"])
    local_surface[~score_mask] = 0.0
    local_surface /= local_surface.sum()
    total_mass = float(json.loads((ROOT / protocol["source"]["p0589_report"]).read_text())["nominal"]["total_mass_msun"])
    transition_radius = transition_radius_arcsec(
        total_mass,
        a0_m_s2=protocol["source"]["a0_m_s2"],
        scale_kpc_per_arcsec=protocol["source"]["scale_kpc_per_arcsec"],
    )

    development_models = [model for model in protocol["target_maps"] if model["role"] == "development"]
    development_targets = {
        model["model_id"]: load_reprojected_map(model, protocol, xx, yy, score_mask)
        for model in development_models
    }
    candidate_rows = []
    predictions: dict[tuple[float, float, float], tuple[np.ndarray, np.ndarray]] = {}
    for lambda_return, eta_width, routed_fraction in itertools.product(
        protocol["frozen_factorial"]["lambda_return_radius"],
        protocol["frozen_factorial"]["eta_width_fraction"],
        protocol["frozen_factorial"]["routed_fraction"],
    ):
        return_radius = float(lambda_return) * transition_radius
        width = max(float(eta_width) * return_radius, float(axis[1] - axis[0]))
        kernel = normalized_ring_kernel(axis, return_radius_arcsec=return_radius, width_arcsec=width)
        prediction, arrival = routed_arrival_map(local_surface, kernel, routed_fraction=routed_fraction)
        prediction[~score_mask] = 0.0
        prediction /= prediction.sum()
        arrival[~score_mask] = 0.0
        arrival /= arrival.sum()
        key = (float(lambda_return), float(eta_width), float(routed_fraction))
        predictions[key] = (prediction, arrival)
        scores = [jensen_shannon_divergence(prediction, target, score_mask) for target in development_targets.values()]
        candidate_rows.append(
            {
                "lambda_return_radius": lambda_return,
                "eta_width_fraction": eta_width,
                "routed_fraction": routed_fraction,
                "return_radius_arcsec": return_radius,
                "width_arcsec": width,
                "development_mean_jsd": float(np.mean(scores)),
                **{f"{model_id}_jsd": score for model_id, score in zip(development_targets, scores, strict=True)},
            }
        )
    candidates = pd.DataFrame(candidate_rows).sort_values(
        ["development_mean_jsd", "lambda_return_radius", "eta_width_fraction", "routed_fraction"]
    )
    best = candidates.iloc[0]
    best_key = (float(best.lambda_return_radius), float(best.eta_width_fraction), float(best.routed_fraction))
    best_prediction, best_arrival = predictions[best_key]

    # The method-holdout files are first opened only after the candidate is locked.
    holdout_models = [model for model in protocol["target_maps"] if model["role"] == "method_holdout"]
    holdout_targets = {
        model["model_id"]: load_reprojected_map(model, protocol, xx, yy, score_mask)
        for model in holdout_models
    }
    targets = {**development_targets, **holdout_targets}
    models = {model["model_id"]: model for model in protocol["target_maps"]}
    uniform = score_mask.astype(float)
    uniform /= uniform.sum()
    rotated = np.rot90(local_surface)
    rotated[~score_mask] = 0.0
    rotated /= rotated.sum()
    map_rows = []
    residuals = {}
    peaks = {}
    for model_id, target in targets.items():
        residual, baryon_scale = positive_dark_residual(target, local_surface, score_mask)
        residuals[model_id] = residual
        peaks[model_id] = find_peaks(
            residual,
            axis,
            score_mask,
            protocol["selection"]["peaks_per_map"],
            protocol["selection"]["peak_minimum_separation_arcsec"],
        )
        local_jsd = jensen_shannon_divergence(local_surface, target, score_mask)
        route_jsd = jensen_shannon_divergence(best_prediction, target, score_mask)
        map_rows.append(
            {
                "model_id": model_id,
                "role": models[model_id]["role"],
                "method": models[model_id]["method"],
                "local_baryon_jsd": local_jsd,
                "locked_route_jsd": route_jsd,
                "jsd_improvement_fraction": (local_jsd - route_jsd) / local_jsd,
                "uniform_jsd": jensen_shannon_divergence(uniform, target, score_mask),
                "rotated_baryon_jsd": jensen_shannon_divergence(rotated, target, score_mask),
                "local_baryon_pearson": pearson_map(local_surface, target, score_mask),
                "locked_route_pearson": pearson_map(best_prediction, target, score_mask),
                "arrival_to_apparent_dark_residual_jsd": jensen_shannon_divergence(best_arrival, residual, score_mask),
                "fitted_local_baryon_projection_scale": baryon_scale,
            }
        )
    map_scores = pd.DataFrame(map_rows)

    north = (-0.053688162174961813, 0.04399991999548547)
    south = (20.161589672378902, -35.43040008000702)
    source_groups = sources.apply(origin_group, axis=1, args=(north, south))
    backtrack_rows = []
    for model_id, model_peaks in peaks.items():
        for rank, (peak_x, peak_y, peak_value) in enumerate(model_peaks, start=1):
            probability = source_origin_probabilities(
                sources.x_arcsec,
                sources.y_arcsec,
                sources.mass_msun,
                destination_x_arcsec=peak_x,
                destination_y_arcsec=peak_y,
                return_radius_arcsec=float(best.return_radius_arcsec),
                width_arcsec=float(best.width_arcsec),
            )
            grouped = pd.Series(probability).groupby(source_groups.reset_index(drop=True)).sum().sort_values(ascending=False)
            top_source = int(np.argmax(probability))
            origin_x = float(sources.iloc[top_source].x_arcsec)
            origin_y = float(sources.iloc[top_source].y_arcsec)
            distance = math.hypot(origin_x - peak_x, origin_y - peak_y)
            geometry = semicircle_arc_geometry(distance, protocol["source"]["scale_kpc_per_arcsec"])
            backtrack_rows.append(
                {
                    "model_id": model_id,
                    "role": models[model_id]["role"],
                    "peak_rank": rank,
                    "peak_x_arcsec": peak_x,
                    "peak_y_arcsec": peak_y,
                    "peak_residual_density": peak_value,
                    "top_origin_group": str(grouped.index[0]),
                    "top_origin_group_probability": float(grouped.iloc[0]),
                    "second_origin_group": str(grouped.index[1]),
                    "second_origin_group_probability": float(grouped.iloc[1]),
                    "top_source_component": str(sources.iloc[top_source].component),
                    "top_source_id": str(sources.iloc[top_source].source_id),
                    "top_source_probability": float(probability[top_source]),
                    "origin_x_arcsec": origin_x,
                    "origin_y_arcsec": origin_y,
                    **geometry,
                }
            )
    backtracks = pd.DataFrame(backtrack_rows)

    holdout_ids = list(holdout_targets)
    first = np.asarray([(x, y) for x, y, _ in peaks[holdout_ids[0]]])
    second = np.asarray([(x, y) for x, y, _ in peaks[holdout_ids[1]]])
    distance_matrix = np.hypot(first[:, None, 0] - second[None, :, 0], first[:, None, 1] - second[None, :, 1])
    row_indices, column_indices = linear_sum_assignment(distance_matrix)
    peak_agreement = float(np.median(distance_matrix[row_indices, column_indices]))

    output_dir = ROOT / protocol["outputs"]["directory"]
    output_dir.mkdir(parents=True, exist_ok=True)
    candidates.to_csv(output_dir / protocol["outputs"]["candidate_scores"], index=False)
    map_scores.to_csv(output_dir / protocol["outputs"]["map_scores"], index=False)
    backtracks.to_csv(output_dir / protocol["outputs"]["backtracked_peaks"], index=False)

    figure, axes = plt.subplots(2, 3, figsize=(14, 9), constrained_layout=True)
    extent = [axis[0], axis[-1], axis[0], axis[-1]]
    for panel, (model_id, target) in zip(axes.flat[:4], targets.items(), strict=True):
        panel.imshow(target, origin="lower", extent=extent, cmap="magma")
        panel.contour(xx, yy, best_prediction, levels=np.quantile(best_prediction[score_mask], [0.7, 0.9, 0.98]), colors="cyan", linewidths=0.8)
        panel.set_title(f"{model_id}: target + return contours")
    axes.flat[4].imshow(best_prediction, origin="lower", extent=extent, cmap="viridis")
    axes.flat[4].set_title("locked conservative return prediction")
    mean_holdout = np.mean(np.stack(list(holdout_targets.values())), axis=0)
    axes.flat[5].imshow(mean_holdout, origin="lower", extent=extent, cmap="gray_r")
    for row in backtracks[backtracks.role == "method_holdout"].itertuples():
        axes.flat[5].plot([row.origin_x_arcsec, row.peak_x_arcsec], [row.origin_y_arcsec, row.peak_y_arcsec], color="tab:red", alpha=0.5, lw=0.8)
        axes.flat[5].scatter([row.peak_x_arcsec], [row.peak_y_arcsec], color="gold", s=12)
    axes.flat[5].set_title("holdout peaks backtracked to top sources")
    for panel in axes.flat:
        panel.set(xlim=(-115, 115), ylim=(-115, 115), xlabel="west offset (arcsec)", ylabel="north offset (arcsec)")
        panel.set_aspect("equal")
    figure.savefig(output_dir / protocol["outputs"]["figure"], dpi=180)
    plt.close(figure)

    holdout_scores = map_scores[map_scores.role == "method_holdout"]
    gate_values = {
        "both_method_holdouts_improve_over_local_baryon_jsd": bool(np.all(holdout_scores.jsd_improvement_fraction > 0.0)),
        "mean_holdout_jsd_improvement_fraction": float(holdout_scores.jsd_improvement_fraction.mean()),
        "mean_holdout_jsd_improvement_pass": bool(holdout_scores.jsd_improvement_fraction.mean() >= protocol["advance_gates"]["mean_holdout_jsd_improvement_fraction_min"]),
        "minimum_holdout_top_origin_group_probability": float(backtracks[backtracks.role == "method_holdout"].top_origin_group_probability.min()),
        "backtrack_origin_probability_pass": bool(backtracks[backtracks.role == "method_holdout"].top_origin_group_probability.min() >= protocol["advance_gates"]["holdout_residual_peak_backtrack_top_origin_probability_min"]),
        "method_holdout_peak_agreement_median_arcsec": peak_agreement,
        "method_peak_agreement_pass": bool(peak_agreement <= protocol["advance_gates"]["map_method_peak_agreement_median_arcsec_max"]),
    }
    report = {
        "report_version": "P0590-GRAVITY-RETURN-BACKTRACK-RESULTS-0.1.0",
        "status": "complete_descriptive_single_cluster_method_transfer",
        "protocol": {"path": rel(protocol_path), "sha256": sha256(protocol_path)},
        "input_hashes": {
            "baryon_sources": sha256(sources_path),
            **{model["model_id"]: sha256(map_path(model)) for model in protocol["target_maps"]},
        },
        "natural_scale": {"total_baryonic_mass_msun": total_mass, "transition_radius_arcsec": transition_radius, "transition_radius_kpc": transition_radius * protocol["source"]["scale_kpc_per_arcsec"]},
        "locked_candidate": best.to_dict(),
        "method_holdout": {
            "mean_local_baryon_jsd": float(holdout_scores.local_baryon_jsd.mean()),
            "mean_locked_route_jsd": float(holdout_scores.locked_route_jsd.mean()),
            "mean_improvement_fraction": float(holdout_scores.jsd_improvement_fraction.mean()),
            "systems_improved": int(np.count_nonzero(holdout_scores.jsd_improvement_fraction > 0.0)),
            "maps": int(len(holdout_scores)),
        },
        "backtracking": {
            "holdout_peaks": int(np.count_nonzero(backtracks.role == "method_holdout")),
            "median_top_origin_group_probability": float(backtracks[backtracks.role == "method_holdout"].top_origin_group_probability.median()),
            "dominant_origin_groups": backtracks[backtracks.role == "method_holdout"].top_origin_group.value_counts().to_dict(),
            "median_projected_route_kpc": float(backtracks[backtracks.role == "method_holdout"].projected_distance_kpc.median()),
            "median_illustrative_hidden_height_kpc": float(backtracks[backtracks.role == "method_holdout"].maximum_hidden_height_kpc.median()),
        },
        "gates": gate_values,
        "interpretation": "The return law is only a viable next-stage seed if it improves both method holdouts and the independently reconstructed residual peaks agree. Backtracking confidence alone cannot validate the path physics.",
        "claim_limits": protocol["claim_limits"],
    }
    (output_dir / protocol["outputs"]["report"]).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (output_dir / protocol["outputs"]["summary"]).write_text(
        "# P0590 conservative gravity-return backtracking\n\n"
        f"The frozen source-only scale is r_a={transition_radius:.2f} arcsec. Development selected lambda={best.lambda_return_radius:g}, eta={best.eta_width_fraction:g}, f={best.routed_fraction:g}, giving a return radius of {best.return_radius_arcsec:.2f} arcsec.\n\n"
        f"On the two reconstruction-method holdouts, mean JSD changed from {holdout_scores.local_baryon_jsd.mean():.4f} to {holdout_scores.locked_route_jsd.mean():.4f} ({holdout_scores.jsd_improvement_fraction.mean():+.2%}). The two holdout residual-peak sets agree to a median {peak_agreement:.2f} arcsec after optimal matching.\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
