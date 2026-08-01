#!/usr/bin/env python3
"""Test and visualize a baryon-sourced local gravity-routing tensor."""

from __future__ import annotations

import hashlib
import json
import math
import sys
from dataclasses import dataclass
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
sys.path.insert(0, str(ROOT / "scripts"))

from run_gravity_arc_fresh_sample import (  # noqa: E402
    build_source_context,
    regrid_kappa_sky,
)
from run_gravity_arc_tomography import regrid_kappa  # noqa: E402


@dataclass
class SystemData:
    label: str
    cohort: str
    redshift: float
    axis: np.ndarray
    x_grid: np.ndarray
    y_grid: np.ndarray
    radius: np.ndarray
    positions: np.ndarray
    weights: np.ndarray
    range_maps: list[np.ndarray]
    glafic_map: np.ndarray | None


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


def weighted_quantile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    keep = np.isfinite(values) & np.isfinite(weights) & (weights > 0.0)
    if not np.any(keep):
        return float("nan")
    order = np.argsort(values[keep])
    v = values[keep][order]
    w = weights[keep][order]
    cumulative = np.cumsum(w)
    return float(np.interp(float(q) * cumulative[-1], cumulative, v))


def deposit_baryons(data: SystemData, smoothing_kpc: float) -> np.ndarray:
    spacing = float(data.axis[1] - data.axis[0])
    edges = np.concatenate(
        [data.axis - 0.5 * spacing, [data.axis[-1] + 0.5 * spacing]]
    )
    image, _, _ = np.histogram2d(
        data.positions[:, 1],
        data.positions[:, 0],
        bins=[edges, edges],
        weights=data.weights,
    )
    image = gaussian_filter(image, smoothing_kpc / spacing, mode="constant")
    image[data.radius > 350.0] = 0.0
    total = float(np.sum(image))
    if total <= 0.0:
        raise RuntimeError(f"{data.label}: empty baryon map")
    return image / total


def lens_source_map(
    image: np.ndarray,
    radius: np.ndarray,
    spacing_kpc: float,
    smoothing_kpc: float,
    background_annulus: tuple[float, float],
) -> np.ndarray:
    finite = np.isfinite(image)
    annulus = (
        finite
        & (radius >= float(background_annulus[0]))
        & (radius <= float(background_annulus[1]))
    )
    if not np.any(annulus):
        raise RuntimeError("lensing map lacks the background annulus")
    background = float(np.median(image[annulus]))
    source = np.maximum(np.where(finite, image - background, 0.0), 0.0)
    source = gaussian_filter(source, smoothing_kpc / spacing_kpc, mode="constant")
    source[radius > 350.0] = 0.0
    total = float(np.sum(source))
    if total <= 0.0:
        raise RuntimeError("lensing residual map has no positive support")
    return source / total


def poisson_acceleration(source: np.ndarray, spacing_kpc: float) -> tuple[np.ndarray, np.ndarray]:
    """Open-boundary approximation from a two-times zero-padded FFT Poisson solve."""
    ny, nx = source.shape
    padded = np.zeros((2 * ny, 2 * nx), dtype=float)
    y0, x0 = ny // 2, nx // 2
    padded[y0 : y0 + ny, x0 : x0 + nx] = source
    ky = 2.0 * np.pi * np.fft.fftfreq(padded.shape[0], d=spacing_kpc)
    kx = 2.0 * np.pi * np.fft.fftfreq(padded.shape[1], d=spacing_kpc)
    kx_grid, ky_grid = np.meshgrid(kx, ky, indexing="xy")
    k2 = kx_grid**2 + ky_grid**2
    source_hat = np.fft.fft2(padded)
    potential_hat = np.zeros_like(source_hat, dtype=complex)
    nonzero = k2 > 0.0
    potential_hat[nonzero] = -source_hat[nonzero] / k2[nonzero]
    # a=-grad(Phi); Phi_hat=-source_hat/k^2.
    ax_pad = np.fft.ifft2(-1j * kx_grid * potential_hat).real
    ay_pad = np.fft.ifft2(-1j * ky_grid * potential_hat).real
    return (
        ax_pad[y0 : y0 + ny, x0 : x0 + nx],
        ay_pad[y0 : y0 + ny, x0 : x0 + nx],
    )


def field_diagnostics(
    data: SystemData,
    lens_source: np.ndarray,
    baryon_source: np.ndarray,
    protocol: dict,
    method: str,
    realization: int | None,
) -> tuple[dict, dict[str, np.ndarray]]:
    settings = protocol["preprocessing"]
    spacing = float(settings["grid_spacing_kpc"])
    lens_x, lens_y = poisson_acceleration(lens_source, spacing)
    baryon_x, baryon_y = poisson_acceleration(baryon_source, spacing)
    lens_norm = np.hypot(lens_x, lens_y)
    baryon_norm = np.hypot(baryon_x, baryon_y)
    score_aperture = data.radius <= float(settings["field_score_radius_kpc"])
    positive = score_aperture & (lens_source > 0.0)
    percentile = float(settings["field_floor_percentile"])
    lens_floor = float(np.percentile(lens_norm[positive], percentile))
    baryon_floor = float(np.percentile(baryon_norm[positive], percentile))
    mask = positive & (lens_norm > lens_floor) & (baryon_norm > baryon_floor)
    dot = lens_x * baryon_x + lens_y * baryon_y
    cosine = np.clip(
        np.divide(dot, lens_norm * baryon_norm, out=np.zeros_like(dot), where=mask),
        -1.0,
        1.0,
    )
    weights = np.where(mask, lens_source, 0.0)
    weights /= np.sum(weights)
    feasible = mask & (cosine > 0.0)
    sin_abs = np.sqrt(np.maximum(0.0, 1.0 - cosine**2))
    chi = np.full_like(cosine, np.nan)
    chi[feasible] = (1.0 + sin_abs[feasible]) / np.maximum(
        1.0 - sin_abs[feasible], 1.0e-12
    )
    feasible_weight = float(np.sum(weights[feasible]))
    feasible_weights = np.where(feasible, weights, 0.0)
    projection_aperture = score_aperture
    denominator = float(np.sum(baryon_source[projection_aperture] ** 2))
    local_projection = max(
        0.0,
        float(
            np.sum(
                lens_source[projection_aperture] * baryon_source[projection_aperture]
            )
            / denominator
        ),
    )
    arrival_residual = np.maximum(lens_source - local_projection * baryon_source, 0.0)
    arrival_residual[~score_aperture] = 0.0
    residual_total = float(np.sum(arrival_residual))
    if residual_total > 0.0:
        arrival_residual /= residual_total
    residual_mask = mask & (arrival_residual > 0.0)
    residual_weights = np.where(residual_mask, arrival_residual, 0.0)
    if float(np.sum(residual_weights)) > 0.0:
        residual_weights /= np.sum(residual_weights)
    residual_feasible = residual_mask & (cosine > 0.0)
    residual_feasible_weights = np.where(residual_feasible, residual_weights, 0.0)
    record = {
        "system": data.label,
        "cohort": data.cohort,
        "method": method,
        "realization": realization,
        "score_pixels": int(np.sum(mask)),
        "weighted_mean_cosine": float(np.sum(weights * cosine)),
        "weighted_feasible_fraction": feasible_weight,
        "weighted_opposed_fraction_cos_lt_minus_025": float(
            np.sum(weights[mask & (cosine < -0.25)])
        ),
        "weighted_median_angle_deg": weighted_quantile(
            np.degrees(np.arccos(cosine[mask])), weights[mask], 0.5
        ),
        "weighted_median_chi_min_feasible": weighted_quantile(
            chi[feasible], feasible_weights[feasible], 0.5
        ),
        "weighted_p90_chi_min_feasible": weighted_quantile(
            chi[feasible], feasible_weights[feasible], 0.9
        ),
        "baryon_projection_coefficient": local_projection,
        "residual_weighted_mean_cosine": float(np.sum(residual_weights * cosine)),
        "residual_weighted_feasible_fraction": float(
            np.sum(residual_weights[residual_feasible])
        ),
        "residual_weighted_opposed_fraction_cos_lt_minus_025": float(
            np.sum(residual_weights[residual_mask & (cosine < -0.25)])
        ),
        "residual_weighted_median_chi_min_feasible": weighted_quantile(
            chi[residual_feasible], residual_feasible_weights[residual_feasible], 0.5
        ),
        "residual_weighted_p90_chi_min_feasible": weighted_quantile(
            chi[residual_feasible], residual_feasible_weights[residual_feasible], 0.9
        ),
        "lens_field_floor": lens_floor,
        "baryon_field_floor": baryon_floor,
    }
    maps = {
        "lens_source": lens_source,
        "baryon_source": baryon_source,
        "lens_x": lens_x,
        "lens_y": lens_y,
        "baryon_x": baryon_x,
        "baryon_y": baryon_y,
        "cosine": cosine,
        "chi": chi,
        "mask": mask,
        "weights": weights,
        "arrival_residual": arrival_residual,
        "residual_weights": residual_weights,
    }
    return record, maps


def select_peaks(
    image: np.ndarray,
    data: SystemData,
    count: int,
    min_separation_kpc: float,
    score_radius_kpc: float,
) -> list[tuple[int, int]]:
    spacing = float(data.axis[1] - data.axis[0])
    size = max(3, int(round(min_separation_kpc / spacing)) * 2 + 1)
    local_max = image == maximum_filter(image, size=size, mode="constant")
    candidate = np.argwhere(local_max & (data.radius <= score_radius_kpc) & (image > 0.0))
    candidate = sorted(candidate, key=lambda ij: image[tuple(ij)], reverse=True)
    selected: list[tuple[int, int]] = []
    for y_index, x_index in candidate:
        x = data.axis[x_index]
        y = data.axis[y_index]
        if all(
            math.hypot(x - data.axis[other_x], y - data.axis[other_y])
            >= min_separation_kpc
            for other_y, other_x in selected
        ):
            selected.append((int(y_index), int(x_index)))
        if len(selected) >= count:
            break
    return selected


def interpolate_field(field: np.ndarray, x: float, y: float, axis: np.ndarray) -> float:
    spacing = float(axis[1] - axis[0])
    x_pixel = (x - axis[0]) / spacing
    y_pixel = (y - axis[0]) / spacing
    return float(
        map_coordinates(
            field,
            np.asarray([[y_pixel], [x_pixel]]),
            order=1,
            mode="constant",
            cval=np.nan,
            prefilter=False,
        )[0]
    )


def trace_to_baryon(
    data: SystemData,
    start_x: float,
    start_y: float,
    field_x: np.ndarray,
    field_y: np.ndarray,
    protocol: dict,
) -> dict:
    settings = protocol["backtracking"]
    step = float(settings["step_kpc"])
    max_steps = int(float(settings["maximum_path_kpc"]) / step)
    arrival = float(settings["arrival_radius_kpc"])
    x, y = float(start_x), float(start_y)
    path_x = [x]
    path_y = [y]
    reached = False
    source_index = -1
    for _ in range(max_steps):
        distances = np.hypot(data.positions[:, 0] - x, data.positions[:, 1] - y)
        source_index = int(np.argmin(distances))
        if float(distances[source_index]) <= arrival:
            reached = True
            break
        fx = interpolate_field(field_x, x, y, data.axis)
        fy = interpolate_field(field_y, x, y, data.axis)
        norm = math.hypot(fx, fy)
        if not np.isfinite(norm) or norm <= 0.0:
            break
        x += step * fx / norm
        y += step * fy / norm
        path_x.append(x)
        path_y.append(y)
        if math.hypot(x, y) > 400.0:
            break
    if source_index < 0:
        distances = np.hypot(data.positions[:, 0] - x, data.positions[:, 1] - y)
        source_index = int(np.argmin(distances))
    source_x, source_y = data.positions[source_index]
    if reached:
        path_x.append(float(source_x))
        path_y.append(float(source_y))
    direct = math.hypot(start_x - source_x, start_y - source_y)
    path_length = float(
        np.sum(np.hypot(np.diff(np.asarray(path_x)), np.diff(np.asarray(path_y))))
    )
    return {
        "reached_baryon": reached,
        "source_index": source_index,
        "source_x_kpc": float(source_x),
        "source_y_kpc": float(source_y),
        "source_weight": float(data.weights[source_index]),
        "path_length_kpc": path_length,
        "direct_distance_kpc": direct,
        "tortuosity": path_length / direct if direct > 0.0 else 1.0,
        "end_distance_to_source_kpc": float(math.hypot(x - source_x, y - source_y)),
        "path_x_kpc": ";".join(f"{value:.3f}" for value in path_x),
        "path_y_kpc": ";".join(f"{value:.3f}" for value in path_y),
    }


def backtrack_peaks(
    data: SystemData,
    maps: dict[str, np.ndarray],
    protocol: dict,
    method: str,
) -> list[dict]:
    settings = protocol["backtracking"]
    peaks = select_peaks(
        maps["arrival_residual"],
        data,
        int(settings["peaks_per_map"]),
        float(settings["peak_minimum_separation_kpc"]),
        float(protocol["preprocessing"]["field_score_radius_kpc"]),
    )
    rows = []
    for rank, (y_index, x_index) in enumerate(peaks, start=1):
        x = float(data.axis[x_index])
        y = float(data.axis[y_index])
        traced = trace_to_baryon(
            data, x, y, maps["baryon_x"], maps["baryon_y"], protocol
        )
        rows.append(
            {
                "system": data.label,
                "cohort": data.cohort,
                "method": method,
                "peak_rank": rank,
                "peak_x_kpc": x,
                "peak_y_kpc": y,
                "peak_weight": float(maps["arrival_residual"][y_index, x_index]),
                "local_cosine": float(maps["cosine"][y_index, x_index]),
                **traced,
            }
        )
    return rows


def pilot_systems(protocol: dict) -> list[SystemData]:
    acquisition = json.loads(
        (ROOT / protocol["inputs"]["pilot_acquisition"]).read_text(encoding="utf-8")
    )
    sources = pd.read_csv(ROOT / protocol["inputs"]["pilot_sources"])
    settings = protocol["preprocessing"]
    size = int(settings["grid_pixels"])
    spacing = float(settings["grid_spacing_kpc"])
    axis = (np.arange(size) - (size - 1) / 2.0) * spacing
    x_grid, y_grid = np.meshgrid(axis, axis, indexing="xy")
    output = []
    for system in acquisition["systems"]:
        label = system["label"]
        redshift = float(system["cluster_redshift"])
        kpc_per_arcsec = float(Planck18.kpc_proper_per_arcmin(redshift).value / 60.0)
        range_paths = sorted((ROOT / system["lensing_directory"] / "range").glob("*_kappa.fits"))
        header = fits.getheader(range_paths[0])
        pixel_scale_kpc = abs(float(header["CDELT1"])) * 3600.0 * kpc_per_arcsec
        center_x = float(header["CRPIX1"]) - 1.0
        center_y = float(header["CRPIX2"]) - 1.0
        local = sources[sources.system.eq(label)].copy()
        local = local[local.hard_member.astype(str).str.lower().eq("true")]
        local["x_kpc"] = (local.map_x_pixel - center_x) * pixel_scale_kpc
        local["y_kpc"] = (local.map_y_pixel - center_y) * pixel_scale_kpc
        local = local[np.hypot(local.x_kpc, local.y_kpc) <= float(settings["baryon_source_radius_kpc"])]
        positions = local[["x_kpc", "y_kpc"]].to_numpy(float)
        weights = np.maximum(local.f160w_flux_nJy.to_numpy(float), 0.0)
        weights /= np.sum(weights)
        geometry = {"kpc_per_arcsec": kpc_per_arcsec, "x_grid": x_grid, "y_grid": y_grid}
        maps = [regrid_kappa(path, geometry) for path in range_paths]
        output.append(
            SystemData(
                label=label,
                cohort="spent_pilot",
                redshift=redshift,
                axis=axis,
                x_grid=x_grid,
                y_grid=y_grid,
                radius=np.hypot(x_grid, y_grid),
                positions=positions,
                weights=weights,
                range_maps=maps,
                glafic_map=None,
            )
        )
        print(f"loaded pilot {label}: {len(maps)} maps", flush=True)
    return output


def fresh_systems(protocol: dict) -> list[SystemData]:
    acquisition = json.loads(
        (ROOT / protocol["inputs"]["fresh_acquisition"]).read_text(encoding="utf-8")
    )
    analysis = json.loads(
        (ROOT / protocol["inputs"]["fresh_analysis"]).read_text(encoding="utf-8")
    )
    sources = pd.read_csv(ROOT / protocol["inputs"]["fresh_sources"])
    audits = pd.read_csv(ROOT / protocol["inputs"]["fresh_systems_audit"])
    manifest = pd.read_csv(ROOT / protocol["inputs"]["fresh_manifest"])
    settings = {
        "pixels_per_axis": int(protocol["preprocessing"]["grid_pixels"]),
        "grid_spacing_kpc": float(protocol["preprocessing"]["grid_spacing_kpc"]),
        "common_radius_kpc": float(protocol["preprocessing"]["baryon_source_radius_kpc"]),
    }
    holdouts = set(protocol["data"]["new_analysis_holdout_systems"])
    output = []
    for system in acquisition["systems"]:
        label = system["label"]
        audit_row = audits[audits.system.eq(label)].iloc[0]
        context, world = build_source_context(system, audit_row, sources, settings)
        local_manifest = manifest[manifest.system.eq(label)]
        range_rows = local_manifest[
            local_manifest.kind.eq("range_kappa") & local_manifest.method.eq("lenstool")
        ].sort_values("sample_index")
        maps = []
        for index, row in enumerate(range_rows.itertuples(index=False), start=1):
            maps.append(regrid_kappa_sky(ROOT / row.path, world, context.x_grid.shape))
            if index in {25, 50, 75, 100}:
                print(f"loading {label}: {index}/100 Lenstool maps", flush=True)
        glafic_row = local_manifest[
            local_manifest.kind.eq("best_kappa") & local_manifest.method.eq("glafic")
        ].iloc[0]
        glafic = regrid_kappa_sky(ROOT / glafic_row.path, world, context.x_grid.shape)
        cohort = "new_analysis_holdout" if label in holdouts else "development"
        output.append(
            SystemData(
                label=label,
                cohort=cohort,
                redshift=float(system["cluster_redshift"]),
                axis=context.axis_kpc,
                x_grid=context.x_grid,
                y_grid=context.y_grid,
                radius=context.radius_grid,
                positions=context.positions,
                weights=context.hard_weights,
                range_maps=maps,
                glafic_map=glafic,
            )
        )
    return output


def make_figure(
    systems: list[SystemData],
    map_products: dict[str, dict[str, np.ndarray]],
    metrics: pd.DataFrame,
    paths: pd.DataFrame,
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(3, 5, figsize=(19, 11), constrained_layout=True)
    axes = axes.ravel()
    for axis_plot, data in zip(axes[:13], systems):
        maps = map_products[data.label]
        extent = [data.axis[0], data.axis[-1], data.axis[0], data.axis[-1]]
        axis_plot.imshow(
            maps["arrival_residual"],
            origin="lower",
            extent=extent,
            cmap="magma",
            vmin=0.0,
            vmax=float(
                np.percentile(maps["arrival_residual"][data.radius <= 250.0], 99.5)
            ),
        )
        levels = np.quantile(maps["baryon_source"][maps["baryon_source"] > 0], [0.75, 0.9, 0.98])
        axis_plot.contour(
            data.x_grid,
            data.y_grid,
            maps["baryon_source"],
            levels=np.unique(levels),
            colors="cyan",
            linewidths=0.55,
        )
        local_paths = paths[(paths.system == data.label) & (paths.method == "lenstool_ensemble")]
        for row in local_paths.itertuples(index=False):
            px = np.asarray([float(value) for value in row.path_x_kpc.split(";")])
            py = np.asarray([float(value) for value in row.path_y_kpc.split(";")])
            axis_plot.plot(px, py, color="white", lw=0.9, alpha=0.85)
            axis_plot.plot(row.peak_x_kpc, row.peak_y_kpc, "wo", ms=2.5)
            axis_plot.plot(row.source_x_kpc, row.source_y_kpc, "c+", ms=5)
        value = metrics[
            (metrics.system == data.label)
            & (metrics.method == "lenstool_ensemble")
            & metrics.realization.isna()
        ].iloc[0]
        axis_plot.set_title(
            f"{data.label}\nfull/res feasible {100*value.weighted_feasible_fraction:.0f}%/"
            f"{100*value.residual_weighted_feasible_fraction:.0f}% | chi "
            f"{value.weighted_median_chi_min_feasible:.1f}",
            fontsize=8,
        )
        axis_plot.set_xlim(-300, 300)
        axis_plot.set_ylim(-300, 300)
        axis_plot.set_xticks([])
        axis_plot.set_yticks([])
    primary = metrics[(metrics.method == "lenstool_ensemble") & metrics.realization.isna()]
    axes[13].barh(primary.system, 100.0 * primary.weighted_feasible_fraction, color="steelblue")
    axes[13].axvline(90.0, color="black", ls="--", lw=1)
    axes[13].set_xlabel("convergence-weighted feasible area (%)")
    axes[13].tick_params(axis="y", labelsize=6)
    axes[13].set_xlim(0, 100)
    fresh = primary[primary.cohort.ne("spent_pilot")]
    axes[14].scatter(
        fresh.weighted_feasible_fraction,
        fresh.weighted_median_chi_min_feasible,
        c=np.where(fresh.cohort.eq("new_analysis_holdout"), "tab:orange", "tab:blue"),
    )
    for row in fresh.itertuples(index=False):
        axes[14].annotate(row.system, (row.weighted_feasible_fraction, row.weighted_median_chi_min_feasible), fontsize=5)
    axes[14].axvline(0.9, color="black", ls="--", lw=1)
    axes[14].axhline(4.0, color="black", ls="--", lw=1)
    axes[14].set_xlabel("feasible fraction")
    axes[14].set_ylabel("minimum median tensor anisotropy")
    fig.suptitle(
        "P0567 baryon-flux tensor backtracking\n"
        "orange: apparent-dark residual | cyan: baryonic light | white: baryon-field backtracks",
        fontsize=13,
    )
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def write_summary(report: dict, output_path: Path) -> None:
    aggregate = report["aggregate"]
    gates = report["gates"]
    lines = [
        "# P0567 baryon-flux tensor backtracking",
        "",
        "## Outcome",
        "",
        (
            f"Across the ten fresh RELICS systems, {100*aggregate['fresh_equal_system_feasible_fraction']:.1f}% "
            "of the convergence-weighted field lies where a local symmetric positive routing tensor can map "
            "the lensing-inferred field into the baryon-sourced flux."
        ),
        (
            f"The equal-system median lower-bound anisotropy is {aggregate['fresh_median_system_chi_min']:.2f}; "
            f"the median system p90 is {aggregate['fresh_median_system_p90_chi_min']:.2f}."
        ),
        (
            f"The top-peak backtracks reach a catalogued member in {100*aggregate['primary_peak_reach_fraction']:.1f}% "
            "of cases. These are source attributions under the baryonic field, not observed paths."
        ),
        (
            f"When the field-alignment test is weighted specifically by the apparent-dark residual, the "
            f"equal-system feasible fraction is {100*aggregate['fresh_equal_system_residual_feasible_fraction']:.1f}%."
        ),
        "",
        "## Gates",
        "",
        f"- Local feasibility: **{gates['local_feasibility_gate']}**",
        f"- Practical distortion: **{gates['practical_distortion_gate']}**",
        f"- Lenstool/GLAFIC method robustness: **{gates['method_robustness_gate']}**",
        "",
        "Passing a gate means only that this representation is geometrically available. The pointwise tensor "
        "was inferred from the lens map and therefore has not predicted that map.",
        "",
        "## Interpretation",
        "",
        "The test replaces an invented endpoint arc with a field equation. The baryonic source generates a "
        "conserved flux `J_b`; the standard lens map supplies an apparent field `g_L`; and a positive tensor "
        "`K` would relate them through `J_b=K g_L`. Opposed vectors are a hard local failure. Aligned vectors "
        "give a calculable lower bound on how anisotropic spacetime would need to be.",
        "",
        "The next non-circular test is to fit a smooth, low-parameter `K` using only baryonic density, its "
        "gradient, and its tidal tensor in the development systems, freeze it, and predict the three P0567 "
        "holdouts without consulting their convergence maps.",
    ]
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    protocol_path = ROOT / "configs/p0567_baryon_flux_tensor_backtrack_protocol.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol["status"] != "amended_after_total_peak_overlap_audit_before_residual_peak_recomputation":
        raise RuntimeError("P0567 protocol is not in the required frozen state")
    output = ROOT / protocol["outputs"]["directory"]
    output.mkdir(parents=True, exist_ok=True)
    systems = pilot_systems(protocol) + fresh_systems(protocol)
    settings = protocol["preprocessing"]
    uncertainty_indices = [int(value) for value in settings["uncertainty_indices"]]
    metric_rows = []
    uncertainty_rows = []
    path_rows = []
    map_products: dict[str, dict[str, np.ndarray]] = {}
    aggregate_chi = []
    aggregate_chi_weights = []
    for data in systems:
        spacing = float(settings["grid_spacing_kpc"])
        baryon_source = deposit_baryons(data, float(settings["baryon_smoothing_kpc"]))
        map_stack = np.asarray(data.range_maps)
        finite_count = np.sum(np.isfinite(map_stack), axis=0)
        mean_raw = np.divide(
            np.nansum(map_stack, axis=0),
            finite_count,
            out=np.full_like(map_stack[0], np.nan),
            where=finite_count > 0,
        )
        ensemble_source = lens_source_map(
            mean_raw,
            data.radius,
            spacing,
            float(settings["lensing_smoothing_kpc"]),
            tuple(settings["background_annulus_kpc"]),
        )
        record, maps = field_diagnostics(
            data, ensemble_source, baryon_source, protocol, "lenstool_ensemble", None
        )
        metric_rows.append(record)
        map_products[data.label] = maps
        path_rows.extend(backtrack_peaks(data, maps, protocol, "lenstool_ensemble"))
        if data.cohort != "spent_pilot":
            feasible = maps["mask"] & np.isfinite(maps["chi"])
            aggregate_chi.append(maps["chi"][feasible])
            aggregate_chi_weights.append(maps["weights"][feasible] / 10.0)
        for sample_index in uncertainty_indices:
            sample_source = lens_source_map(
                data.range_maps[sample_index],
                data.radius,
                spacing,
                float(settings["lensing_smoothing_kpc"]),
                tuple(settings["background_annulus_kpc"]),
            )
            sample_record, _ = field_diagnostics(
                data,
                sample_source,
                baryon_source,
                protocol,
                "lenstool_realization",
                sample_index,
            )
            uncertainty_rows.append(sample_record)
        if data.glafic_map is not None:
            glafic_source = lens_source_map(
                data.glafic_map,
                data.radius,
                spacing,
                float(settings["lensing_smoothing_kpc"]),
                tuple(settings["background_annulus_kpc"]),
            )
            glafic_record, glafic_maps = field_diagnostics(
                data, glafic_source, baryon_source, protocol, "glafic_best", None
            )
            metric_rows.append(glafic_record)
            path_rows.extend(backtrack_peaks(data, glafic_maps, protocol, "glafic_best"))
        print(
            f"scored {data.label}: feasible={record['weighted_feasible_fraction']:.3f}, "
            f"median_chi={record['weighted_median_chi_min_feasible']:.3f}",
            flush=True,
        )
    metrics = pd.DataFrame(metric_rows)
    uncertainty = pd.DataFrame(uncertainty_rows)
    paths = pd.DataFrame(path_rows)
    metrics.to_csv(output / protocol["outputs"]["field_metrics"], index=False)
    uncertainty.to_csv(output / protocol["outputs"]["uncertainty"], index=False)
    paths.to_csv(output / protocol["outputs"]["peak_backtracks"], index=False)
    primary = metrics[(metrics.method == "lenstool_ensemble") & metrics.realization.isna()]
    fresh = primary[primary.cohort.ne("spent_pilot")]
    development = fresh[fresh.cohort.eq("development")]
    holdout = fresh[fresh.cohort.eq("new_analysis_holdout")]
    combined_chi = np.concatenate(aggregate_chi)
    combined_weights = np.concatenate(aggregate_chi_weights)
    method = metrics.pivot(index="system", columns="method", values="weighted_feasible_fraction")
    method = method.dropna(subset=["lenstool_ensemble", "glafic_best"])
    method_delta = np.abs(method.lenstool_ensemble - method.glafic_best)
    fresh_paths = paths[(paths.cohort.ne("spent_pilot")) & (paths.method == "lenstool_ensemble")]
    equal_feasible = float(fresh.weighted_feasible_fraction.mean())
    min_feasible = float(fresh.weighted_feasible_fraction.min())
    pooled_chi_median = weighted_quantile(combined_chi, combined_weights, 0.5)
    pooled_chi_p90 = weighted_quantile(combined_chi, combined_weights, 0.9)
    gates = {
        "local_feasibility_gate": bool(equal_feasible >= 0.90 and min_feasible >= 0.75),
        "practical_distortion_gate": bool(pooled_chi_median <= 4.0 and pooled_chi_p90 <= 20.0),
        "method_robustness_gate": bool(float(method_delta.median()) <= 0.10),
        "no_formula_promoted": True,
    }
    report = {
        "report_version": "P0567-BARYON-FLUX-TENSOR-BACKTRACK-RESULTS-0.1.1",
        "status": "complete_inverse_geometry_diagnostic",
        "protocol": {"path": str(protocol_path.relative_to(ROOT)), "sha256": sha256(protocol_path)},
        "coverage": {
            "systems": len(systems),
            "spent_pilot_systems": int(np.sum(primary.cohort.eq("spent_pilot"))),
            "development_systems": len(development),
            "new_analysis_holdout_systems": len(holdout),
            "lenstool_realizations_read": int(sum(len(item.range_maps) for item in systems)),
            "uncertainty_field_realizations": len(uncertainty),
            "glafic_method_controls": len(method),
            "peak_backtracks": len(paths),
        },
        "aggregate": {
            "fresh_equal_system_feasible_fraction": equal_feasible,
            "fresh_equal_system_residual_feasible_fraction": float(
                fresh.residual_weighted_feasible_fraction.mean()
            ),
            "fresh_minimum_system_feasible_fraction": min_feasible,
            "fresh_minimum_system_residual_feasible_fraction": float(
                fresh.residual_weighted_feasible_fraction.min()
            ),
            "fresh_pooled_feasible_chi_median": pooled_chi_median,
            "fresh_pooled_feasible_chi_p90": pooled_chi_p90,
            "fresh_median_system_chi_min": float(fresh.weighted_median_chi_min_feasible.median()),
            "fresh_median_system_p90_chi_min": float(fresh.weighted_p90_chi_min_feasible.median()),
            "development_equal_system_feasible_fraction": float(
                development.weighted_feasible_fraction.mean()
            ),
            "holdout_equal_system_feasible_fraction": float(holdout.weighted_feasible_fraction.mean()),
            "development_equal_system_residual_feasible_fraction": float(
                development.residual_weighted_feasible_fraction.mean()
            ),
            "holdout_equal_system_residual_feasible_fraction": float(
                holdout.residual_weighted_feasible_fraction.mean()
            ),
            "median_abs_lenstool_glafic_feasible_difference": float(method_delta.median()),
            "primary_peak_reach_fraction": float(fresh_paths.reached_baryon.mean()),
            "primary_peak_median_path_kpc": float(fresh_paths.path_length_kpc.median()),
            "primary_peak_median_tortuosity": float(
                fresh_paths.loc[fresh_paths.reached_baryon, "tortuosity"].median()
            ),
        },
        "per_system": json_safe(primary.to_dict(orient="records")),
        "uncertainty_summary": json_safe(
            uncertainty.groupby("system")["weighted_feasible_fraction"]
            .quantile([0.16, 0.5, 0.84])
            .unstack()
            .reset_index()
            .rename(columns={0.16: "p16", 0.5: "median", 0.84: "p84"})
            .to_dict(orient="records")
        ),
        "gates": gates,
        "interpretation": {
            "what_is_measured": "A pointwise existence and minimum-distortion bound for a local positive routing tensor, plus baryonic-field source attribution for high-convergence peaks.",
            "what_is_not_measured": "No smooth universal K is fitted, absolute strength is normalized out, and no raw lens observable is predicted.",
            "next_falsifiable_step": "Fit a smooth low-parameter K from baryon-only local invariants on the seven development systems and predict the three P0567 holdouts without reading their target maps during fitting.",
        },
        "claim_limits": protocol["claim_limits"],
    }
    (output / protocol["outputs"]["report"]).write_text(
        json.dumps(json_safe(report), indent=2) + "\n", encoding="utf-8"
    )
    write_summary(report, output / protocol["outputs"]["summary"])
    make_figure(systems, map_products, metrics, paths, output / protocol["outputs"]["figure"])
    print(json.dumps(json_safe(report["aggregate"]), indent=2), flush=True)
    print(json.dumps(gates, indent=2), flush=True)


if __name__ == "__main__":
    main()
