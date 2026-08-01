#!/usr/bin/env python3
"""Infer and cross-validate nonlocal baryonic gravity-return paths."""

from __future__ import annotations

import hashlib
import json
import math
import sys
from dataclasses import dataclass, replace
from itertools import product
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.cosmology import Planck18
from astropy.io import fits
from scipy.ndimage import gaussian_filter, map_coordinates, maximum_filter


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from voidscreen.gravity_arc_tomography import sinkhorn_transport


@dataclass
class ClusterContext:
    label: str
    redshift: float
    kpc_per_arcsec: float
    axis_kpc: np.ndarray
    x_grid: np.ndarray
    y_grid: np.ndarray
    radius_grid: np.ndarray
    aperture: np.ndarray
    positions: np.ndarray
    soft_weights: np.ndarray
    hard_weights: np.ndarray
    target_mean: np.ndarray
    target_samples: np.ndarray
    target_raw_mean: np.ndarray
    target_raw_samples: np.ndarray
    mean_kappa_unprocessed: np.ndarray


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_safe(value):
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        number = float(value)
        return number if np.isfinite(number) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def normalized_in_aperture(image: np.ndarray, aperture: np.ndarray) -> np.ndarray:
    values = np.maximum(np.asarray(image, dtype=float), 0.0)
    total = float(np.sum(values[aperture]))
    if not np.isfinite(total) or total <= 0.0:
        raise ValueError("map has no positive support in scoring aperture")
    return values / total


def regrid_kappa(path: Path, context_geometry: dict) -> np.ndarray:
    data = np.asarray(fits.getdata(path), dtype=float)
    header = fits.getheader(path)
    pixel_scale_arcsec = abs(float(header["CDELT1"])) * 3600.0
    pixel_scale_kpc = pixel_scale_arcsec * context_geometry["kpc_per_arcsec"]
    center_x = float(header["CRPIX1"]) - 1.0
    center_y = float(header["CRPIX2"]) - 1.0
    pixel_x = center_x + context_geometry["x_grid"] / pixel_scale_kpc
    pixel_y = center_y + context_geometry["y_grid"] / pixel_scale_kpc
    coordinates = np.vstack([pixel_y.ravel(), pixel_x.ravel()])
    return map_coordinates(
        data,
        coordinates,
        order=1,
        mode="constant",
        cval=np.nan,
        prefilter=False,
    ).reshape(context_geometry["x_grid"].shape)


def preprocess_target(
    image: np.ndarray,
    radius: np.ndarray,
    aperture: np.ndarray,
    *,
    spacing_kpc: float,
    smoothing_kpc: float,
    subtract_background: bool,
) -> np.ndarray:
    finite = np.isfinite(image)
    filled = np.where(finite, image, 0.0)
    if subtract_background:
        annulus = finite & (radius >= 250.0) & (radius <= 300.0)
        if not np.any(annulus):
            raise ValueError("target lacks the frozen 250-300 kpc background annulus")
        filled = filled - float(np.median(image[annulus]))
    filled = np.maximum(filled, 0.0)
    smoothed = gaussian_filter(
        filled,
        sigma=float(smoothing_kpc) / float(spacing_kpc),
        mode="constant",
        cval=0.0,
    )
    smoothed[~aperture] = 0.0
    return normalized_in_aperture(smoothed, aperture)


def build_contexts(protocol: dict) -> list[ClusterContext]:
    acquisition = json.loads(
        (ROOT / protocol["inputs"]["acquisition_protocol"]).read_text(encoding="utf-8")
    )
    audit = json.loads((ROOT / protocol["inputs"]["input_audit"]).read_text(encoding="utf-8"))
    if audit["status"] != "completed_without_inspecting_catalog_to_kappa_spatial_correlation":
        raise RuntimeError("input audit is not in the frozen pre-correlation state")
    sources = pd.read_csv(ROOT / protocol["inputs"]["sources"])
    settings = protocol["spatial_preprocessing"]
    size = int(settings["pixels_per_axis"])
    spacing = float(settings["grid_spacing_kpc"])
    axis = (np.arange(size) - (size - 1) / 2.0) * spacing
    x_grid, y_grid = np.meshgrid(axis, axis, indexing="xy")
    radius_grid = np.hypot(x_grid, y_grid)
    aperture = radius_grid <= float(settings["common_radius_kpc"])
    contexts = []
    for system in acquisition["systems"]:
        label = system["label"]
        redshift = float(system["cluster_redshift"])
        kpc_per_arcsec = float(Planck18.kpc_proper_per_arcmin(redshift).value / 60.0)
        range_paths = sorted((ROOT / system["lensing_directory"] / "range").glob("*_kappa.fits"))
        header = fits.getheader(range_paths[0])
        pixel_scale_kpc = abs(float(header["CDELT1"])) * 3600.0 * kpc_per_arcsec
        center_x = float(header["CRPIX1"]) - 1.0
        center_y = float(header["CRPIX2"]) - 1.0
        local_sources = sources[sources.system.eq(label)].copy()
        local_sources["x_kpc"] = (local_sources.map_x_pixel - center_x) * pixel_scale_kpc
        local_sources["y_kpc"] = (local_sources.map_y_pixel - center_y) * pixel_scale_kpc
        local_sources["radius_kpc"] = np.hypot(local_sources.x_kpc, local_sources.y_kpc)
        local_sources = local_sources[local_sources.radius_kpc <= float(settings["common_radius_kpc"])]
        positions = local_sources[["x_kpc", "y_kpc"]].to_numpy(float)
        soft = np.maximum(local_sources.soft_f160w_weight.to_numpy(float), 0.0)
        hard = np.where(
            local_sources.hard_member.to_numpy(bool),
            np.maximum(local_sources.f160w_flux_nJy.to_numpy(float), 0.0),
            0.0,
        )
        soft_positive = soft > 0.0
        positions = positions[soft_positive]
        hard = hard[soft_positive]
        soft = soft[soft_positive]
        soft /= np.sum(soft)
        if np.sum(hard) > 0.0:
            hard /= np.sum(hard)
        geometry = {
            "kpc_per_arcsec": kpc_per_arcsec,
            "x_grid": x_grid,
            "y_grid": y_grid,
        }
        regridded = np.asarray([regrid_kappa(path, geometry) for path in range_paths])
        finite_count = np.sum(np.isfinite(regridded), axis=0)
        mean_unprocessed = np.divide(
            np.nansum(regridded, axis=0),
            finite_count,
            out=np.zeros_like(regridded[0]),
            where=finite_count > 0,
        )
        primary_samples = np.asarray(
            [
                preprocess_target(
                    item,
                    radius_grid,
                    aperture,
                    spacing_kpc=spacing,
                    smoothing_kpc=float(settings["target_smoothing_kpc"]),
                    subtract_background=True,
                )[aperture]
                for item in regridded
            ]
        )
        raw_samples = np.asarray(
            [
                preprocess_target(
                    item,
                    radius_grid,
                    aperture,
                    spacing_kpc=spacing,
                    smoothing_kpc=float(settings["target_smoothing_kpc"]),
                    subtract_background=False,
                )[aperture]
                for item in regridded
            ]
        )
        target_mean = np.zeros_like(x_grid)
        target_mean[aperture] = np.mean(primary_samples, axis=0)
        target_mean = normalized_in_aperture(target_mean, aperture)
        target_raw_mean = np.zeros_like(x_grid)
        target_raw_mean[aperture] = np.mean(raw_samples, axis=0)
        target_raw_mean = normalized_in_aperture(target_raw_mean, aperture)
        contexts.append(
            ClusterContext(
                label,
                redshift,
                kpc_per_arcsec,
                axis,
                x_grid,
                y_grid,
                radius_grid,
                aperture,
                positions,
                soft,
                hard,
                target_mean,
                primary_samples,
                target_raw_mean,
                raw_samples,
                mean_unprocessed,
            )
        )
    return contexts


def deposit_points(
    context: ClusterContext,
    positions: np.ndarray,
    weights: np.ndarray,
    width_kpc: float,
) -> np.ndarray:
    spacing = float(context.axis_kpc[1] - context.axis_kpc[0])
    edges = np.concatenate(
        [context.axis_kpc - 0.5 * spacing, [context.axis_kpc[-1] + 0.5 * spacing]]
    )
    histogram, _, _ = np.histogram2d(
        positions[:, 1], positions[:, 0], bins=[edges, edges], weights=weights
    )
    smoothed = gaussian_filter(
        histogram,
        sigma=float(width_kpc) / spacing,
        mode="constant",
        cval=0.0,
    )
    total = float(np.sum(smoothed))
    if total <= 0.0:
        raise ValueError("deposition lost all routed weight")
    return smoothed / total


def baryonic_directions(context: ClusterContext, softening_kpc: float) -> dict:
    positions = context.positions
    weights = context.soft_weights
    center = np.sum(positions * weights[:, None], axis=0)
    inward = center[None, :] - positions
    inward_norm = np.linalg.norm(inward, axis=1)
    inward /= np.maximum(inward_norm[:, None], np.finfo(float).tiny)
    delta = positions[None, :, :] - positions[:, None, :]
    distance2 = np.sum(np.square(delta), axis=2)
    np.fill_diagonal(distance2, np.inf)
    external = np.sum(
        weights[None, :, None]
        * delta
        / np.power(distance2[:, :, None] + float(softening_kpc) ** 2, 1.5),
        axis=1,
    )
    external_strength = np.linalg.norm(external, axis=1)
    external /= np.maximum(external_strength[:, None], np.finfo(float).tiny)
    partner_score = weights[None, :] / (distance2 + float(softening_kpc) ** 2)
    partner_index = np.argmax(partner_score, axis=1)
    partner_vector = positions[partner_index] - positions
    partner_distance = np.linalg.norm(partner_vector, axis=1)
    partner = partner_vector / np.maximum(partner_distance[:, None], np.finfo(float).tiny)
    return {
        "center": center,
        "inward": inward,
        "radius": inward_norm,
        "external": external,
        "external_strength": external_strength,
        "partner": partner,
        "partner_distance": partner_distance,
        "partner_index": partner_index,
    }


def scaled_length(base: float, ratio: np.ndarray, exponent: float) -> np.ndarray:
    safe = np.maximum(np.asarray(ratio, dtype=float), 1.0e-6)
    return np.clip(float(base) * np.power(safe, float(exponent)), 0.2 * base, 3.0 * base)


def routed_component(
    context: ClusterContext,
    law: str,
    *,
    return_scale_kpc: float,
    exponent: float,
    width_kpc: float,
    landing_mode: str,
    softening_kpc: float,
    tube_samples: int,
) -> np.ndarray:
    positions = context.positions
    weights = context.soft_weights
    directions = baryonic_directions(context, softening_kpc)
    if law == "isotropic_return":
        angles = np.linspace(0.0, 2.0 * np.pi, 32, endpoint=False)
        unit = np.column_stack([np.cos(angles), np.sin(angles)])
        endpoint = positions[:, None, :] + float(return_scale_kpc) * unit[None, :, :]
        endpoint = endpoint.reshape(-1, 2)
        routed_weight = np.repeat(weights / len(angles), len(angles))
        if landing_mode == "tube":
            fractions = np.linspace(0.0, 1.0, int(tube_samples))
            starts = np.repeat(positions, len(angles), axis=0)
            path = starts[:, None, :] + fractions[None, :, None] * (
                endpoint[:, None, :] - starts[:, None, :]
            )
            endpoint = path.reshape(-1, 2)
            routed_weight = np.repeat(routed_weight / len(fractions), len(fractions))
        return deposit_points(context, endpoint, routed_weight, width_kpc)
    if law == "center_return":
        direction = directions["inward"]
        length = scaled_length(return_scale_kpc, directions["radius"] / 100.0, exponent)
    elif law == "external_field_return":
        direction = directions["external"]
        positive = directions["external_strength"] > 0.0
        median = float(np.median(directions["external_strength"][positive]))
        length = scaled_length(
            return_scale_kpc, directions["external_strength"] / median, exponent
        )
    elif law == "strongest_neighbor_return":
        direction = directions["partner"]
        length = scaled_length(return_scale_kpc, directions["partner_distance"] / 100.0, exponent)
        length = np.minimum(length, directions["partner_distance"])
    else:
        raise ValueError(law)
    endpoint = positions + length[:, None] * direction
    if landing_mode == "endpoint":
        return deposit_points(context, endpoint, weights, width_kpc)
    fractions = np.linspace(0.0, 1.0, int(tube_samples))
    path = positions[:, None, :] + fractions[None, :, None] * (
        endpoint[:, None, :] - positions[:, None, :]
    )
    return deposit_points(
        context,
        path.reshape(-1, 2),
        np.repeat(weights / len(fractions), len(fractions)),
        width_kpc,
    )


def candidate_specs(protocol: dict) -> list[dict]:
    grid = protocol["forward_laws"]["grids"]
    specs = []
    for width in grid["local_width_kpc"]:
        specs.append({"family": "local_gaussian", "width_kpc": float(width)})
    for width in grid["central_halo_width_kpc"]:
        specs.append({"family": "central_halo_null", "width_kpc": float(width)})
    for fraction, length, width, landing in product(
        grid["redistributed_fraction"],
        grid["return_scale_kpc"],
        grid["landing_width_kpc"],
        protocol["forward_laws"]["landing_modes"],
    ):
        specs.append(
            {
                "family": "isotropic_return",
                "fraction": float(fraction),
                "return_scale_kpc": float(length),
                "exponent": 0.0,
                "width_kpc": float(width),
                "landing_mode": landing,
            }
        )
    for family, fraction, length, exponent, width, landing in product(
        ["center_return", "external_field_return", "strongest_neighbor_return"],
        grid["redistributed_fraction"],
        grid["return_scale_kpc"],
        grid["environment_exponent"],
        grid["landing_width_kpc"],
        protocol["forward_laws"]["landing_modes"],
    ):
        specs.append(
            {
                "family": family,
                "fraction": float(fraction),
                "return_scale_kpc": float(length),
                "exponent": float(exponent),
                "width_kpc": float(width),
                "landing_mode": landing,
            }
        )
    for index, spec in enumerate(specs):
        spec["candidate_id"] = f"C{index:04d}"
    return specs


def prediction_for_spec(context: ClusterContext, spec: dict, protocol: dict) -> np.ndarray:
    family = spec["family"]
    width = float(spec["width_kpc"])
    if family == "local_gaussian":
        prediction = deposit_points(context, context.positions, context.soft_weights, width)
    elif family == "central_halo_null":
        center = np.sum(context.positions * context.soft_weights[:, None], axis=0)
        prediction = deposit_points(context, center[None, :], np.ones(1), width)
    else:
        local = deposit_points(context, context.positions, context.soft_weights, width)
        grid = protocol["forward_laws"]["grids"]
        routed = routed_component(
            context,
            family,
            return_scale_kpc=float(spec["return_scale_kpc"]),
            exponent=float(spec["exponent"]),
            width_kpc=width,
            landing_mode=spec["landing_mode"],
            softening_kpc=float(grid["external_field_softening_kpc"]),
            tube_samples=int(grid["tube_samples"]),
        )
        fraction = float(spec["fraction"])
        prediction = (1.0 - fraction) * local + fraction * routed
    return normalized_in_aperture(prediction, context.aperture)


def shape_metrics(prediction: np.ndarray, target: np.ndarray, aperture: np.ndarray) -> dict:
    p = np.maximum(prediction[aperture], 0.0)
    q = np.maximum(target[aperture] if target.shape == prediction.shape else target, 0.0)
    p /= np.sum(p)
    q /= np.sum(q)
    middle = 0.5 * (p + q)
    positive_p = p > 0.0
    positive_q = q > 0.0
    js = 0.5 * np.sum(p[positive_p] * np.log(p[positive_p] / middle[positive_p]))
    js += 0.5 * np.sum(q[positive_q] * np.log(q[positive_q] / middle[positive_q]))
    if np.std(p) > 0.0 and np.std(q) > 0.0:
        correlation = float(np.corrcoef(p, q)[0, 1])
    else:
        correlation = 0.0
    nrmse = float(np.sqrt(np.sum(np.square(p - q))) / np.sqrt(np.sum(np.square(q))))
    x = np.asarray(np.broadcast_to(np.arange(prediction.shape[1]), prediction.shape))[aperture]
    y = np.asarray(np.broadcast_to(np.arange(prediction.shape[0])[:, None], prediction.shape))[aperture]
    spacing = 10.0
    centroid_p = np.array([np.sum(p * x), np.sum(p * y)])
    centroid_q = np.array([np.sum(q * x), np.sum(q * y)])
    return {
        "jensen_shannon": float(js),
        "pearson": correlation,
        "normalized_RMSE": nrmse,
        "centroid_offset_kpc": float(np.linalg.norm(centroid_p - centroid_q) * spacing),
    }


def weighted_quantile(values: np.ndarray, weights: np.ndarray, quantile: float) -> float:
    order = np.argsort(values)
    cumulative = np.cumsum(weights[order])
    return float(np.interp(float(quantile), cumulative, values[order]))


def inverse_tomography(protocol: dict, contexts: list[ClusterContext], output: Path):
    settings = protocol["inverse_tomography"]
    spacing = float(protocol["spatial_preprocessing"]["grid_spacing_kpc"])
    factor = int(round(float(settings["target_grid_spacing_kpc"]) / spacing))
    records = []
    peak_records = []
    primary_plans = {}
    for context in contexts:
        size = context.target_mean.shape[0]
        usable = (size // factor) * factor
        coarse = context.target_mean[:usable, :usable].reshape(
            usable // factor, factor, usable // factor, factor
        ).sum(axis=(1, 3))
        coarse_x = context.x_grid[:usable, :usable].reshape(
            usable // factor, factor, usable // factor, factor
        ).mean(axis=(1, 3))
        coarse_y = context.y_grid[:usable, :usable].reshape(
            usable // factor, factor, usable // factor, factor
        ).mean(axis=(1, 3))
        target_mask = (np.hypot(coarse_x, coarse_y) <= 300.0) & (coarse > 0.0)
        target_positions = np.column_stack([coarse_x[target_mask], coarse_y[target_mask]])
        target_weights = coarse[target_mask]
        target_weights /= np.sum(target_weights)
        cost = np.sum(
            np.square(context.positions[:, None, :] - target_positions[None, :, :]), axis=2
        )
        directions = baryonic_directions(
            context,
            float(protocol["forward_laws"]["grids"]["external_field_softening_kpc"]),
        )
        for entropy_length in settings["entropy_length_kpc"]:
            plan = sinkhorn_transport(
                context.soft_weights,
                target_weights,
                cost,
                entropy=float(entropy_length) ** 2,
                iterations=500,
                tolerance=1.0e-8,
            )
            displacement = target_positions[None, :, :] - context.positions[:, None, :]
            distance = np.linalg.norm(displacement, axis=2)
            unit = displacement / np.maximum(distance[:, :, None], np.finfo(float).tiny)
            moving = distance > 5.0
            moving_weight = np.where(moving, plan, 0.0)
            moving_weight /= max(np.sum(moving_weight), np.finfo(float).tiny)
            record = {
                "system": context.label,
                "entropy_length_kpc": float(entropy_length),
                "mean_path_kpc": float(np.sum(plan * distance)),
                "median_path_kpc": weighted_quantile(distance.ravel(), plan.ravel(), 0.5),
                "p90_path_kpc": weighted_quantile(distance.ravel(), plan.ravel(), 0.9),
                "fraction_le_25_kpc": float(np.sum(plan[distance <= 25.0])),
                "fraction_le_50_kpc": float(np.sum(plan[distance <= 50.0])),
                "fraction_le_100_kpc": float(np.sum(plan[distance <= 100.0])),
                "fraction_gt_150_kpc": float(np.sum(plan[distance > 150.0])),
                "mean_cos_inward": float(
                    np.sum(moving_weight * np.sum(unit * directions["inward"][:, None, :], axis=2))
                ),
                "mean_cos_external": float(
                    np.sum(moving_weight * np.sum(unit * directions["external"][:, None, :], axis=2))
                ),
                "mean_cos_neighbor": float(
                    np.sum(moving_weight * np.sum(unit * directions["partner"][:, None, :], axis=2))
                ),
                "source_marginal_max_error": float(
                    np.max(np.abs(np.sum(plan, axis=1) - context.soft_weights))
                ),
                "target_marginal_max_error": float(
                    np.max(np.abs(np.sum(plan, axis=0) - target_weights))
                ),
            }
            records.append(record)
            if float(entropy_length) == float(settings["primary_entropy_length_kpc"]):
                primary_plans[context.label] = (plan, target_positions, target_weights)
        plan, target_positions, target_weights = primary_plans[context.label]
        target_image = np.zeros_like(context.target_mean)
        target_image[context.aperture] = context.target_mean[context.aperture]
        peaks = target_image == maximum_filter(target_image, size=11, mode="constant")
        peaks &= context.radius_grid <= 250.0
        peak_indices = np.argwhere(peaks & (target_image > 0.0))
        peak_indices = sorted(
            peak_indices, key=lambda item: target_image[tuple(item)], reverse=True
        )[:5]
        for peak_rank, (iy, ix) in enumerate(peak_indices, start=1):
            peak_position = np.array([context.x_grid[iy, ix], context.y_grid[iy, ix]])
            near = np.linalg.norm(target_positions - peak_position[None, :], axis=1) <= 25.0
            contribution = np.sum(plan[:, near], axis=1)
            total = float(np.sum(contribution))
            for origin_rank in np.argsort(contribution)[::-1][:5]:
                peak_records.append(
                    {
                        "system": context.label,
                        "peak_rank": peak_rank,
                        "peak_x_kpc": float(peak_position[0]),
                        "peak_y_kpc": float(peak_position[1]),
                        "peak_target_weight_within_25kpc": total,
                        "origin_rank": int(np.where(np.argsort(contribution)[::-1] == origin_rank)[0][0] + 1),
                        "source_index": int(origin_rank),
                        "source_x_kpc": float(context.positions[origin_rank, 0]),
                        "source_y_kpc": float(context.positions[origin_rank, 1]),
                        "source_weight": float(context.soft_weights[origin_rank]),
                        "fraction_of_peak_inflow": float(
                            contribution[origin_rank] / max(total, np.finfo(float).tiny)
                        ),
                        "source_to_peak_kpc": float(
                            np.linalg.norm(context.positions[origin_rank] - peak_position)
                        ),
                    }
                )
    frame = pd.DataFrame(records)
    peaks = pd.DataFrame(peak_records)
    frame.to_csv(output / "inverse_path_statistics.csv", index=False)
    peaks.to_csv(output / "peak_origins.csv", index=False)
    return frame, peaks, primary_plans


def run_forward_grid(
    protocol: dict,
    contexts: list[ClusterContext],
    output: Path,
    *,
    filename: str = "forward_grid.csv",
    label: str = "primary",
):
    specs = candidate_specs(protocol)
    rows = []
    for context in contexts:
        print(f"forward grid {label} {context.label}: {len(specs)} candidates", flush=True)
        local_cache = {}
        route_cache = {}
        center_cache = {}
        for spec in specs:
            family = spec["family"]
            width = float(spec["width_kpc"])
            if family == "local_gaussian":
                if width not in local_cache:
                    local_cache[width] = deposit_points(
                        context, context.positions, context.soft_weights, width
                    )
                prediction = normalized_in_aperture(local_cache[width], context.aperture)
            elif family == "central_halo_null":
                if width not in center_cache:
                    center_cache[width] = prediction_for_spec(context, spec, protocol)
                prediction = center_cache[width]
            else:
                if width not in local_cache:
                    local_cache[width] = deposit_points(
                        context, context.positions, context.soft_weights, width
                    )
                route_key = (
                    family,
                    float(spec["return_scale_kpc"]),
                    float(spec["exponent"]),
                    width,
                    spec["landing_mode"],
                )
                if route_key not in route_cache:
                    route_cache[route_key] = routed_component(
                        context,
                        family,
                        return_scale_kpc=route_key[1],
                        exponent=route_key[2],
                        width_kpc=width,
                        landing_mode=route_key[4],
                        softening_kpc=float(
                            protocol["forward_laws"]["grids"]["external_field_softening_kpc"]
                        ),
                        tube_samples=int(protocol["forward_laws"]["grids"]["tube_samples"]),
                    )
                fraction = float(spec["fraction"])
                prediction = normalized_in_aperture(
                    (1.0 - fraction) * local_cache[width]
                    + fraction * route_cache[route_key],
                    context.aperture,
                )
            metrics = shape_metrics(prediction, context.target_mean, context.aperture)
            raw_metrics = shape_metrics(prediction, context.target_raw_mean, context.aperture)
            rows.append(
                {
                    "system": context.label,
                    **spec,
                    **metrics,
                    **{f"raw_target_{key}": value for key, value in raw_metrics.items()},
                }
            )
    frame = pd.DataFrame(rows)
    frame.to_csv(output / filename, index=False)
    return frame, specs


def choose_folds(protocol: dict, forward: pd.DataFrame, specs: list[dict]) -> pd.DataFrame:
    acquisition = json.loads(
        (ROOT / protocol["inputs"]["acquisition_protocol"]).read_text(encoding="utf-8")
    )
    spec_by_id = {row["candidate_id"]: row for row in specs}
    rows = []
    for fold_index, fold in enumerate(acquisition["cross_validation"]["folds"]):
        training = set(fold["training"])
        validation = fold["validation"]
        training_rows = forward[forward.system.isin(training)]
        means = training_rows.groupby(["family", "candidate_id"], as_index=False).agg(
            training_mean_JS=("jensen_shannon", "mean"),
            training_mean_Pearson=("pearson", "mean"),
        )
        winners = means.sort_values(
            ["family", "training_mean_JS", "candidate_id"], kind="stable"
        ).groupby("family", as_index=False).first()
        for winner in winners.itertuples(index=False):
            validation_row = forward[
                forward.system.eq(validation) & forward.candidate_id.eq(winner.candidate_id)
            ].iloc[0]
            rows.append(
                {
                    "fold": fold_index,
                    "validation_system": validation,
                    "selection_scope": "within_family",
                    "family": winner.family,
                    "candidate_id": winner.candidate_id,
                    "training_mean_JS": float(winner.training_mean_JS),
                    "training_mean_Pearson": float(winner.training_mean_Pearson),
                    "validation_JS": float(validation_row.jensen_shannon),
                    "validation_Pearson": float(validation_row.pearson),
                    "validation_normalized_RMSE": float(validation_row.normalized_RMSE),
                    "validation_centroid_offset_kpc": float(validation_row.centroid_offset_kpc),
                    **{
                        key: value
                        for key, value in spec_by_id[winner.candidate_id].items()
                        if key not in {"candidate_id", "family"}
                    },
                }
            )
        overall = means.sort_values(["training_mean_JS", "candidate_id"], kind="stable").iloc[0]
        validation_row = forward[
            forward.system.eq(validation) & forward.candidate_id.eq(overall.candidate_id)
        ].iloc[0]
        rows.append(
            {
                "fold": fold_index,
                "validation_system": validation,
                "selection_scope": "overall",
                "family": overall.family,
                "candidate_id": overall.candidate_id,
                "training_mean_JS": float(overall.training_mean_JS),
                "training_mean_Pearson": float(overall.training_mean_Pearson),
                "validation_JS": float(validation_row.jensen_shannon),
                "validation_Pearson": float(validation_row.pearson),
                "validation_normalized_RMSE": float(validation_row.normalized_RMSE),
                "validation_centroid_offset_kpc": float(validation_row.centroid_offset_kpc),
                **{
                    key: value
                    for key, value in spec_by_id[overall.candidate_id].items()
                    if key not in {"candidate_id", "family"}
                },
            }
        )
    return pd.DataFrame(rows)


def validation_uncertainty(
    protocol: dict,
    contexts: list[ClusterContext],
    folds: pd.DataFrame,
    specs: list[dict],
) -> pd.DataFrame:
    by_label = {context.label: context for context in contexts}
    spec_by_id = {row["candidate_id"]: row for row in specs}
    records = []
    for row in folds.itertuples(index=False):
        context = by_label[row.validation_system]
        spec = spec_by_id[row.candidate_id]
        prediction = prediction_for_spec(context, spec, protocol)
        sample_metrics = [
            shape_metrics(prediction, sample, context.aperture)
            for sample in context.target_samples
        ]
        record = {
            "fold": int(row.fold),
            "validation_system": row.validation_system,
            "selection_scope": row.selection_scope,
            "family": row.family,
            "candidate_id": row.candidate_id,
        }
        for metric in ["jensen_shannon", "pearson", "normalized_RMSE", "centroid_offset_kpc"]:
            values = np.asarray([item[metric] for item in sample_metrics], dtype=float)
            record[f"{metric}_p16"] = float(np.quantile(values, 0.16))
            record[f"{metric}_median"] = float(np.median(values))
            record[f"{metric}_p84"] = float(np.quantile(values, 0.84))
        records.append(record)
    return pd.DataFrame(records)


def angle_shuffle_control(protocol: dict, contexts: list[ClusterContext], folds: pd.DataFrame, specs: list[dict]):
    spec_by_id = {row["candidate_id"]: row for row in specs}
    records = []
    for context_index, context in enumerate(contexts):
        local_row = folds[
            folds.validation_system.eq(context.label)
            & folds.selection_scope.eq("within_family")
            & folds.family.eq("local_gaussian")
        ].iloc[0]
        spec = spec_by_id[local_row.candidate_id]
        center = np.sum(context.positions * context.soft_weights[:, None], axis=0)
        relative = context.positions - center
        radius = np.linalg.norm(relative, axis=1)
        rng = np.random.default_rng(20260731 + context_index)
        values = []
        for _ in range(32):
            angle = rng.uniform(0.0, 2.0 * np.pi, size=len(radius))
            shuffled = center + radius[:, None] * np.column_stack([np.cos(angle), np.sin(angle)])
            prediction = normalized_in_aperture(
                deposit_points(context, shuffled, context.soft_weights, spec["width_kpc"]),
                context.aperture,
            )
            values.append(shape_metrics(prediction, context.target_mean, context.aperture)["jensen_shannon"])
        records.append(
            {
                "system": context.label,
                "selected_local_candidate": local_row.candidate_id,
                "observed_angle_JS": float(local_row.validation_JS),
                "shuffle_JS_median": float(np.median(values)),
                "shuffle_JS_p16": float(np.quantile(values, 0.16)),
                "shuffle_JS_p84": float(np.quantile(values, 0.84)),
                "observed_beats_shuffle_fraction": float(np.mean(float(local_row.validation_JS) < np.asarray(values))),
            }
        )
    return pd.DataFrame(records)


def make_figure(protocol, contexts, folds, specs, plans, output):
    spec_by_id = {row["candidate_id"]: row for row in specs}
    figure, axes = plt.subplots(3, 4, figsize=(18, 15), constrained_layout=True)
    for row_index, context in enumerate(contexts):
        local = deposit_points(context, context.positions, context.soft_weights, 10.0)
        target = context.target_mean
        overall = folds[
            folds.validation_system.eq(context.label) & folds.selection_scope.eq("overall")
        ].iloc[0]
        prediction = prediction_for_spec(context, spec_by_id[overall.candidate_id], protocol)
        extent = [context.axis_kpc[0], context.axis_kpc[-1], context.axis_kpc[0], context.axis_kpc[-1]]
        for axis, image, title in [
            (axes[row_index, 0], local, "baryonic F160W tracer"),
            (axes[row_index, 1], target, "Lenstool apparent-mass target"),
            (
                axes[row_index, 3],
                prediction,
                f"held-out {overall.family}\nJS={overall.validation_JS:.3f}, r={overall.validation_Pearson:.2f}",
            ),
        ]:
            show = np.where(context.radius_grid <= 300.0, image, np.nan)
            axis.imshow(show, origin="lower", extent=extent, cmap="magma")
            axis.set(xlim=(-300, 300), ylim=(-300, 300), title=title)
        plan, target_positions, _ = plans[context.label]
        transport_axis = axes[row_index, 2]
        transport_axis.scatter(
            context.positions[:, 0], context.positions[:, 1],
            s=5 + 80 * np.sqrt(context.soft_weights / np.max(context.soft_weights)),
            color="tab:blue", alpha=0.65,
        )
        flat = np.argsort(plan.ravel())[::-1][:80]
        source_index, target_index = np.unravel_index(flat, plan.shape)
        maximum = float(np.max(plan))
        for source, target_cell in zip(source_index, target_index, strict=True):
            transport_axis.plot(
                [context.positions[source, 0], target_positions[target_cell, 0]],
                [context.positions[source, 1], target_positions[target_cell, 1]],
                color="tab:orange", alpha=0.08 + 0.6 * plan[source, target_cell] / maximum,
                linewidth=0.3 + 2.0 * plan[source, target_cell] / maximum,
            )
        transport_axis.set(
            xlim=(-300, 300), ylim=(-300, 300), aspect="equal",
            title="largest minimum-transport paths", xlabel="x (kpc)", ylabel="y (kpc)"
        )
        axes[row_index, 0].set_ylabel(f"{context.label}\ny (kpc)")
    figure.savefig(output, dpi=180)
    plt.close(figure)


def write_summary(report: dict, output: Path):
    def finite(value, default=0.0):
        return float(value) if value is not None and np.isfinite(float(value)) else float(default)

    lines = [
        "# Gravity-arc tomography",
        "",
        "This experiment treats baryonic galaxies as the only sources and the RELICS Lenstool convergence maps as model-dependent locations that a redirected field would need to reproduce. All scores are normalized spatial-shape tests; no absolute missing-gravity amplitude is inferred.",
        "",
        "## Held-out results",
        "",
        "| Validation cluster | selected family | scale (kpc) | fraction | mode | JS | Pearson |",
        "|---|---|---:|---:|---|---:|---:|",
    ]
    for row in report["overall_fold_winners"]:
        lines.append(
            f"| {row['validation_system']} | {row['family']} | {finite(row.get('return_scale_kpc'), finite(row.get('width_kpc'))):.1f} | "
            f"{finite(row.get('fraction')):.2f} | {row.get('landing_mode') if isinstance(row.get('landing_mode'), str) else 'n/a'} | "
            f"{row['validation_JS']:.4f} | {row['validation_Pearson']:.3f} |"
        )
    lines.extend(
        [
            "",
            "The inverse paths are descriptive minimum-cost attributions, not unique physical trajectories. The forward holdouts are the actual universality test. See `report.json`, `inverse_path_statistics.csv`, and `fold_results.csv`.",
        ]
    )
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    protocol_path = ROOT / "configs/gravity_arc_tomography_protocol.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol["status"] != "frozen_before_catalog_to_kappa_spatial_correlation":
        raise RuntimeError("tomography protocol is not frozen")
    output = ROOT / protocol["outputs"]["directory"]
    output.mkdir(parents=True, exist_ok=True)
    contexts = build_contexts(protocol)
    inverse, peaks, plans = inverse_tomography(protocol, contexts, output)
    forward, specs = run_forward_grid(protocol, contexts, output)
    folds = choose_folds(protocol, forward, specs)
    folds.to_csv(output / "fold_results.csv", index=False)
    uncertainty = validation_uncertainty(protocol, contexts, folds, specs)
    uncertainty.to_csv(output / "validation_uncertainty.csv", index=False)
    shuffles = angle_shuffle_control(protocol, contexts, folds, specs)
    shuffles.to_csv(output / "angle_shuffle_control.csv", index=False)
    hard_contexts = []
    for context in contexts:
        selected = context.hard_weights > 0.0
        hard_contexts.append(
            replace(
                context,
                positions=context.positions[selected],
                soft_weights=context.hard_weights[selected]
                / np.sum(context.hard_weights[selected]),
                hard_weights=context.hard_weights[selected]
                / np.sum(context.hard_weights[selected]),
            )
        )
    hard_forward, _ = run_forward_grid(
        protocol,
        hard_contexts,
        output,
        filename="hard_source_forward_grid.csv",
        label="hard-source sensitivity",
    )
    hard_folds = choose_folds(protocol, hard_forward, specs)
    hard_folds.insert(0, "sensitivity", "hard_photoz_members")
    raw_forward = forward.copy()
    for metric in ["jensen_shannon", "pearson", "normalized_RMSE", "centroid_offset_kpc"]:
        raw_forward[metric] = raw_forward[f"raw_target_{metric}"]
    raw_folds = choose_folds(protocol, raw_forward, specs)
    raw_folds.insert(0, "sensitivity", "unsubtracted_positive_kappa")
    sensitivity_folds = pd.concat([hard_folds, raw_folds], ignore_index=True)
    sensitivity_folds.to_csv(output / "sensitivity_fold_results.csv", index=False)
    overall = folds[folds.selection_scope.eq("overall")].to_dict("records")
    within = folds[folds.selection_scope.eq("within_family")]
    comparisons = []
    for validation in within.validation_system.unique():
        block = within[within.validation_system.eq(validation)].set_index("family")
        winner = folds[
            folds.validation_system.eq(validation) & folds.selection_scope.eq("overall")
        ].iloc[0]
        local = block.loc["local_gaussian"]
        central = block.loc["central_halo_null"]
        comparisons.append(
            {
                "validation_system": validation,
                "winner_family": winner.family,
                "winner_JS": float(winner.validation_JS),
                "local_JS": float(local.validation_JS),
                "central_halo_JS": float(central.validation_JS),
                "improvement_over_local_fraction": float(
                    1.0 - winner.validation_JS / local.validation_JS
                ),
                "improvement_over_central_fraction": float(
                    1.0 - winner.validation_JS / central.validation_JS
                ),
                "winner_Pearson": float(winner.validation_Pearson),
                "local_Pearson": float(local.validation_Pearson),
                "central_halo_Pearson": float(central.validation_Pearson),
            }
        )
    useful_gate = all(
        row["improvement_over_local_fraction"] >= 0.10
        and row["improvement_over_central_fraction"] >= 0.10
        and row["winner_Pearson"] >= max(row["local_Pearson"], row["central_halo_Pearson"])
        for row in comparisons
    )
    family_counts = pd.Series([row["family"] for row in overall]).value_counts().to_dict()
    report = {
        "report_version": protocol["protocol_version"],
        "status": "completed normalized gravity-arc inverse and forward transfer test",
        "protocol_sha256": sha256(protocol_path),
        "coverage": {
            "clusters": len(contexts),
            "sources": int(sum(len(context.positions) for context in contexts)),
            "inverse_entropy_scales": len(protocol["inverse_tomography"]["entropy_length_kpc"]),
            "forward_candidates": len(specs),
            "forward_cluster_scores": len(forward),
            "hard_source_forward_cluster_scores": len(hard_forward),
            "leave_one_cluster_out_folds": 3,
            "validation_kappa_realizations_per_fold": 100,
        },
        "inverse_primary": inverse[
            inverse.entropy_length_kpc.eq(protocol["inverse_tomography"]["primary_entropy_length_kpc"])
        ].to_dict("records"),
        "overall_fold_winners": overall,
        "heldout_comparisons": comparisons,
        "useful_spatial_law_gate_passed": useful_gate,
        "winner_family_counts": family_counts,
        "angle_shuffle_control": shuffles.to_dict("records"),
        "sensitivity_overall_winners": sensitivity_folds[
            sensitivity_folds.selection_scope.eq("overall")
        ].to_dict("records"),
        "claim_boundary": [
            protocol["spatial_preprocessing"]["target_model_dependency"],
            protocol["inverse_tomography"]["interpretation_limit"],
            protocol["validation"]["no_truth_probability"],
            protocol["controls"]["later_cross_domain_requirement"],
        ],
    }
    (output / "report.json").write_text(
        json.dumps(json_safe(report), indent=2) + "\n", encoding="utf-8"
    )
    make_figure(protocol, contexts, folds, specs, plans, output / "gravity_arc_tomography.png")
    write_summary(report, output / "SUMMARY.md")
    print(json.dumps(json_safe(report), indent=2))


if __name__ == "__main__":
    main()
