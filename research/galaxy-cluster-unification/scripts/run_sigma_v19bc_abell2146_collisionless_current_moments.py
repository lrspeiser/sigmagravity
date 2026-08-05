#!/usr/bin/env python3
"""Rasterize the frozen V19BB Abell ensemble into collisionless moments."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "sigma_v19bc_abell2146_collisionless_current_moments.json"
SPEED_OF_LIGHT_KM_S = 299792.458


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_parent_hashes(config: dict[str, Any]) -> dict[str, str]:
    actual: dict[str, str] = {}
    for name, spec in config["parents"].items():
        path = ROOT / spec["path"]
        value = sha256(path)
        if value != spec["sha256"]:
            raise ValueError(f"parent hash mismatch for {name}: {value} != {spec['sha256']}")
        actual[name] = value
    return actual


def cic_weights(x: float, y: float, nx: int, ny: int) -> list[tuple[int, int, float]]:
    """Return the unchanged V19BA four-neighbor cloud-in-cell weights."""
    ix = math.floor(x)
    iy = math.floor(y)
    if ix < 0 or iy < 0 or ix + 1 >= nx or iy + 1 >= ny:
        raise ValueError(f"position ({x}, {y}) does not have four in-grid neighbors")
    dx = x - ix
    dy = y - iy
    result = [
        (iy, ix, (1.0 - dx) * (1.0 - dy)),
        (iy, ix + 1, dx * (1.0 - dy)),
        (iy + 1, ix, (1.0 - dx) * dy),
        (iy + 1, ix + 1, dx * dy),
    ]
    if abs(math.fsum(weight for _, _, weight in result) - 1.0) > 1e-14:
        raise ValueError("cloud-in-cell weights do not sum to one")
    if min(weight for _, _, weight in result) < 0.0:
        raise ValueError("cloud-in-cell produced a negative weight")
    return result


def normalized_cauchy_schwarz_margin(rho: float, current: float, second: float) -> float:
    numerator = rho * second - current * current
    denominator = rho * second + current * current
    return 0.0 if denominator == 0.0 else numerator / denominator


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def tangent_roundtrip_error_arcsec(
    wcs: WCS, ra: np.ndarray, dec: np.ndarray, x: np.ndarray, y: np.ndarray
) -> float:
    recovered_ra, recovered_dec = wcs.pixel_to_world_values(x, y)
    dra = (np.asarray(recovered_ra) - ra) * np.cos(np.deg2rad(dec)) * 3600.0
    ddec = (np.asarray(recovered_dec) - dec) * 3600.0
    return float(np.max(np.hypot(dra, ddec)))


def moment_centroid(values: np.ndarray) -> tuple[float, float]:
    weights = np.asarray(values, dtype=float)
    total = float(np.sum(weights))
    if not math.isfinite(total) or total <= 0.0:
        return math.nan, math.nan
    y, x = np.indices(weights.shape, dtype=float)
    return float(np.sum(x * weights) / total), float(np.sum(y * weights) / total)


def centroid_separation_arcsec(
    first: tuple[float, float], second: tuple[float, float], pixel_scale_arcsec: float
) -> float:
    return float(math.hypot(first[0] - second[0], first[1] - second[1]) * pixel_scale_arcsec)


def image_hdu(
    data: np.ndarray,
    wcs_header: fits.Header,
    name: str,
    bunit: str,
    description: str,
    *,
    primary: bool = False,
) -> fits.PrimaryHDU | fits.ImageHDU:
    header = wcs_header.copy()
    header["EXTNAME"] = name
    header["BUNIT"] = bunit
    header["BTYPE"] = description
    hdu_class = fits.PrimaryHDU if primary else fits.ImageHDU
    return hdu_class(data=np.asarray(data), header=header)


def write_moment_fits(
    path: Path,
    wcs_header: fits.Header,
    means: np.ndarray,
    standard_deviations: np.ndarray,
    covariances: np.ndarray,
    analysis_mask: np.ndarray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    units = [
        "relative_F814W_luminosity/pixel",
        "relative_F814W_luminosity*beta/pixel",
        "relative_F814W_luminosity*beta^2/pixel",
    ]
    names = ["LUM", "JLOS", "PLOS"]
    descriptions = [
        "projected relative HST F814W luminosity",
        "signed line-of-sight current moment",
        "positive line-of-sight second moment",
    ]
    hdus: list[fits.PrimaryHDU | fits.ImageHDU] = []
    for index in range(3):
        hdus.append(
            image_hdu(
                means[index],
                wcs_header,
                f"{names[index]}_MEAN",
                units[index],
                f"mean {descriptions[index]}",
                primary=index == 0,
            )
        )
    for index in range(3):
        hdus.append(
            image_hdu(
                standard_deviations[index],
                wcs_header,
                f"{names[index]}_STD",
                units[index],
                f"population standard deviation of {descriptions[index]}",
            )
        )
    covariance_specs = [
        ("COV_LUM_JLOS", "relative_F814W_luminosity^2*beta/pixel^2"),
        ("COV_LUM_PLOS", "relative_F814W_luminosity^2*beta^2/pixel^2"),
        ("COV_JLOS_PLOS", "relative_F814W_luminosity^2*beta^3/pixel^2"),
    ]
    for index, (name, unit) in enumerate(covariance_specs):
        hdus.append(
            image_hdu(
                covariances[index],
                wcs_header,
                name,
                unit,
                "population covariance of collisionless projected moments",
            )
        )
    hdus.append(
        image_hdu(
            np.asarray(analysis_mask, dtype=np.uint8),
            wcs_header,
            "ANALYSIS_MASK",
            "boolean",
            "frozen V19H Abell 2146 X-ray analysis mask",
        )
    )
    fits.HDUList(hdus).writeto(path, overwrite=True, output_verify="exception")


def make_figure(
    path: Path,
    means: np.ndarray,
    analysis_mask: np.ndarray,
    pixel_scale_arcsec: float,
    center_pixel: tuple[float, float],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ny, nx = means.shape[1:]
    center_x, center_y = center_pixel
    extent = [
        -center_x * pixel_scale_arcsec / 60.0,
        (nx - center_x) * pixel_scale_arcsec / 60.0,
        -center_y * pixel_scale_arcsec / 60.0,
        (ny - center_y) * pixel_scale_arcsec / 60.0,
    ]
    positive_luminosity = means[0][means[0] > 0.0]
    positive_second = means[2][means[2] > 0.0]
    luminosity_floor = float(np.min(positive_luminosity)) if positive_luminosity.size else 1e-30
    second_floor = float(np.min(positive_second)) if positive_second.size else 1e-30
    current_limit = float(np.max(np.abs(means[1])))
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.7), constrained_layout=True)
    panels = [
        (
            np.log10(np.maximum(means[0], luminosity_floor)),
            "viridis",
            None,
            None,
            "log10 mean relative F814W luminosity",
        ),
        (
            means[1],
            "coolwarm",
            -current_limit,
            current_limit,
            "mean signed LOS current (luminosity x v/c)",
        ),
        (
            np.log10(np.maximum(means[2], second_floor)),
            "magma",
            None,
            None,
            "log10 mean LOS second moment",
        ),
    ]
    for axis, (image, cmap, vmin, vmax, title) in zip(axes, panels, strict=True):
        plotted = axis.imshow(
            image,
            origin="lower",
            extent=extent,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            interpolation="nearest",
        )
        axis.contour(
            analysis_mask.astype(float),
            levels=[0.5],
            colors="white",
            linewidths=0.6,
            alpha=0.65,
            origin="lower",
            extent=extent,
        )
        axis.set_title(title)
        axis.set_xlabel("tangent-grid x offset (arcmin)")
        axis.set_ylabel("tangent-grid y offset (arcmin)")
        fig.colorbar(plotted, ax=axis, shrink=0.82)
    fig.suptitle(
        "Abell 2146 target-blind collisionless member moments\n"
        "white contour: frozen V19H X-ray analysis mask; no long-wave smoothing applied"
    )
    fig.savefig(path, dpi=180)
    plt.close(fig)


def run(config_path: Path) -> dict[str, Any]:
    config_path = config_path.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config_hash = sha256(config_path)
    implementation = config["implementation"]
    runner_path = (ROOT / implementation["runner"]).resolve()
    if runner_path != Path(__file__).resolve():
        raise ValueError("frozen implementation path does not identify this runner")
    runner_hash = sha256(runner_path)
    if runner_hash != implementation["runner_sha256"]:
        raise ValueError("frozen implementation hash mismatch")
    input_hashes = verify_parent_hashes(config)
    paths = {name: ROOT / spec["path"] for name, spec in config["parents"].items()}
    v19bb_report = json.loads(paths["v19bb_report"].read_text(encoding="utf-8"))
    v19h_report = json.loads(paths["v19h_source_map_report"].read_text(encoding="utf-8"))
    if v19bb_report["decision"] != "passed":
        raise ValueError("V19BB did not authorize the Abell member ensemble")
    if v19bb_report["lensing_or_halo_payload_opened"]:
        raise ValueError("V19BB unexpectedly opened lensing or halo payload")
    if v19h_report["status"] != "both_clusters_passed_frozen_v19h_source_map_gate":
        raise ValueError("V19H did not authorize the frozen source-map grid")
    if v19h_report["lensing_target_opened"]:
        raise ValueError("V19H unexpectedly opened a lensing target")

    with fits.open(paths["v19h_abell2146_analysis_mask"], memmap=False) as hdul:
        analysis_mask = np.asarray(hdul[0].data, dtype=bool)
        wcs = WCS(hdul[0].header)
    wcs_header = wcs.to_header(relax=True)
    ny, nx = analysis_mask.shape
    expected_shape = tuple(config["grid"]["expected_shape_yx"])
    if (ny, nx) != expected_shape:
        raise ValueError(f"grid shape {(ny, nx)} != {expected_shape}")
    pixel_scale_arcsec = float(np.mean(np.abs(wcs.wcs.cdelt)) * 3600.0)
    expected_pixel_scale = float(config["grid"]["expected_pixel_scale_arcsec"])
    if not math.isclose(pixel_scale_arcsec, expected_pixel_scale, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError(f"pixel scale {pixel_scale_arcsec} != {expected_pixel_scale}")

    channel_sum = np.zeros((3, ny, nx), dtype=np.float64)
    channel_square_sum = np.zeros_like(channel_sum)
    cross_sum = np.zeros_like(channel_sum)
    sample_rows: list[dict[str, Any]] = []
    outside_mask_members: set[str] = set()
    max_roundtrip_error = 0.0
    max_deposition_error = 0.0
    minimum_cs_margin = math.inf
    minimum_second_pixel = math.inf
    minimum_global_variance = math.inf
    reference_member_ids: set[str] | None = None
    total_rows = 0
    total_missing_rows = 0
    finite_counts: list[int] = []

    def process_sample(sample_id: int, rows: list[dict[str, str]]) -> None:
        nonlocal max_roundtrip_error
        nonlocal max_deposition_error
        nonlocal minimum_cs_margin
        nonlocal minimum_second_pixel
        nonlocal minimum_global_variance
        nonlocal reference_member_ids
        nonlocal total_rows
        nonlocal total_missing_rows

        expected_members = int(config["population"]["expected_members_per_draw"])
        if len(rows) != expected_members:
            raise ValueError(
                f"sample {sample_id} has {len(rows)} rows, expected {expected_members}"
            )
        member_ids = {row["member_id"] for row in rows}
        if len(member_ids) != expected_members:
            raise ValueError(f"sample {sample_id} contains duplicate members")
        if reference_member_ids is None:
            reference_member_ids = member_ids
        elif member_ids != reference_member_ids:
            raise ValueError(f"sample {sample_id} member inventory changed")

        ra = np.asarray([float(row["ra_deg"]) for row in rows])
        dec = np.asarray([float(row["dec_deg"]) for row in rows])
        x_values, y_values = wcs.world_to_pixel_values(ra, dec)
        x_values = np.asarray(x_values, dtype=float)
        y_values = np.asarray(y_values, dtype=float)
        max_roundtrip_error = max(
            max_roundtrip_error,
            tangent_roundtrip_error_arcsec(wcs, ra, dec, x_values, y_values),
        )

        sparse: dict[int, np.ndarray] = defaultdict(lambda: np.zeros(3, dtype=float))
        source_totals = np.zeros(3, dtype=float)
        source_absolute = np.zeros(3, dtype=float)
        finite_count = 0
        missing_count = 0
        for row, x, y in zip(rows, x_values, y_values, strict=True):
            neighbors = cic_weights(float(x), float(y), nx, ny)
            nearest_x = round(float(x))
            nearest_y = round(float(y))
            if not analysis_mask[nearest_y, nearest_x]:
                outside_mask_members.add(row["member_id"])
            luminosity_text = row["relative_f814w_luminosity"]
            is_finite = bool(luminosity_text)
            expected_state = "measured_f814w" if is_finite else "missing_photometry"
            if row["luminosity_state"] != expected_state:
                raise ValueError(f"sample {sample_id} member {row['member_id']} state mismatch")
            if not is_finite:
                missing_count += 1
                continue
            finite_count += 1
            luminosity = float(luminosity_text)
            beta = float(row["v_los_rest_km_s"]) / SPEED_OF_LIGHT_KM_S
            if not math.isfinite(luminosity) or luminosity <= 0.0 or not math.isfinite(beta):
                raise ValueError(
                    f"sample {sample_id} member {row['member_id']} has invalid moment data"
                )
            values = np.asarray([luminosity, luminosity * beta, luminosity * beta * beta])
            source_totals += values
            source_absolute += np.abs(values)
            for iy, ix, weight in neighbors:
                sparse[iy * nx + ix] += weight * values
        if finite_count + missing_count != expected_members or not sparse:
            raise ValueError(f"sample {sample_id} has invalid finite/missing accounting")
        finite_counts.append(finite_count)
        total_missing_rows += missing_count

        deposited_totals = np.zeros(3, dtype=float)
        masked_luminosity = 0.0
        sample_minimum_margin = math.inf
        for flat_index, values in sparse.items():
            iy, ix = divmod(flat_index, nx)
            deposited_totals += values
            channel_sum[:, iy, ix] += values
            channel_square_sum[:, iy, ix] += values * values
            cross_sum[0, iy, ix] += values[0] * values[1]
            cross_sum[1, iy, ix] += values[0] * values[2]
            cross_sum[2, iy, ix] += values[1] * values[2]
            margin = normalized_cauchy_schwarz_margin(values[0], values[1], values[2])
            sample_minimum_margin = min(sample_minimum_margin, margin)
            minimum_cs_margin = min(minimum_cs_margin, margin)
            minimum_second_pixel = min(minimum_second_pixel, float(values[2]))
            if analysis_mask[iy, ix]:
                masked_luminosity += float(values[0])

        relative_errors = np.abs(deposited_totals - source_totals) / np.maximum(
            source_absolute, np.finfo(float).tiny
        )
        max_deposition_error = max(max_deposition_error, float(np.max(relative_errors)))
        mean_beta = source_totals[1] / source_totals[0]
        beta_variance = source_totals[2] / source_totals[0] - mean_beta * mean_beta
        minimum_global_variance = min(minimum_global_variance, float(beta_variance))
        sample_rows.append(
            {
                "sample_id": sample_id,
                "member_count": len(rows),
                "finite_luminosity_member_count": finite_count,
                "missing_photometry_member_count": missing_count,
                "total_relative_f814w_luminosity": source_totals[0],
                "total_signed_los_current": source_totals[1],
                "total_positive_los_second_moment": source_totals[2],
                "luminosity_weighted_mean_beta_los": mean_beta,
                "luminosity_weighted_beta_los_variance": beta_variance,
                "luminosity_weighted_rms_speed_km_s": math.sqrt(max(0.0, beta_variance))
                * SPEED_OF_LIGHT_KM_S,
                "analysis_mask_luminosity_fraction": masked_luminosity / source_totals[0],
                "maximum_channel_deposition_relative_error": float(np.max(relative_errors)),
                "minimum_normalized_pixel_cauchy_schwarz_margin": sample_minimum_margin,
            }
        )
        total_rows += len(rows)

    with gzip.open(paths["v19bb_ensemble"], "rt", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        active_id: int | None = None
        active_rows: list[dict[str, str]] = []
        for row in reader:
            sample_id = int(row["sample_id"])
            if active_id is None:
                if sample_id != 0:
                    raise ValueError(f"sample sequence starts at {sample_id}, not zero")
                active_id = sample_id
            if sample_id != active_id:
                if sample_id != active_id + 1:
                    raise ValueError(f"sample sequence jumps from {active_id} to {sample_id}")
                process_sample(active_id, active_rows)
                active_id = sample_id
                active_rows = []
            active_rows.append(row)
        if active_id is not None:
            process_sample(active_id, active_rows)

    draws = len(sample_rows)
    if draws != int(config["population"]["expected_draws"]):
        raise ValueError(f"ensemble has {draws} draws")
    means = channel_sum / draws
    variances = np.maximum(channel_square_sum / draws - means * means, 0.0)
    standard_deviations = np.sqrt(variances)
    covariances = np.empty_like(means)
    covariances[0] = cross_sum[0] / draws - means[0] * means[1]
    covariances[1] = cross_sum[1] / draws - means[0] * means[2]
    covariances[2] = cross_sum[2] / draws - means[1] * means[2]

    outputs = config["outputs"]
    map_path = ROOT / outputs["moment_maps"]
    sample_path = ROOT / outputs["sample_global_moments"]
    figure_path = ROOT / outputs["figure"]
    report_path = ROOT / outputs["report"]
    write_moment_fits(
        map_path,
        wcs_header,
        means,
        standard_deviations,
        covariances,
        analysis_mask,
    )
    write_csv(sample_path, sample_rows, list(sample_rows[0]))

    abell_record = next(row for row in v19h_report["clusters"] if row["cluster"] == "ABELL2146")
    final_center = abell_record["final_center"]
    center_pixel = tuple(
        map(float, wcs.world_to_pixel_values(final_center["ra"], final_center["dec"]))
    )
    make_figure(figure_path, means, analysis_mask, pixel_scale_arcsec, center_pixel)

    luminosity_centroid = moment_centroid(means[0])
    second_centroid = moment_centroid(means[2])
    positive_current_centroid = moment_centroid(np.maximum(means[1], 0.0))
    negative_current_centroid = moment_centroid(np.maximum(-means[1], 0.0))
    gates = config["gates"]
    population = config["population"]
    expected_outside = set(gates["exact_outside_analysis_mask_member_ids"])
    gate_results = {
        "all_parent_hashes_exact": True,
        "exact_draw_member_row_and_finite_count_ranges": draws == population["expected_draws"]
        and total_rows == population["expected_ensemble_rows"]
        and min(finite_counts) == population["expected_minimum_finite_luminosity_members_per_draw"]
        and max(finite_counts) == population["expected_maximum_finite_luminosity_members_per_draw"],
        "all_positions_have_four_in_grid_neighbors": True,
        "pixel_scale_exact": math.isclose(
            pixel_scale_arcsec, expected_pixel_scale, rel_tol=0.0, abs_tol=1e-9
        ),
        "wcs_roundtrip": max_roundtrip_error <= gates["maximum_wcs_roundtrip_error_arcsec"],
        "deposition_conservation": max_deposition_error
        <= gates["maximum_per_draw_relative_deposition_error"],
        "pixel_cauchy_schwarz": minimum_cs_margin
        >= gates["minimum_normalized_pixel_cauchy_schwarz_margin"],
        "second_moment_nonnegative": minimum_second_pixel >= 0.0,
        "global_beta_variance_nonnegative": minimum_global_variance
        >= gates["all_global_beta_variances_nonnegative_with_tolerance"],
        "exact_outside_analysis_mask_members": outside_mask_members == expected_outside,
        "missing_photometry_explicit_and_not_deposited": total_missing_rows
        == sum(population["expected_members_per_draw"] - count for count in finite_counts),
        "no_smoothing_or_response_parameter": True,
        "no_absolute_mass_or_transverse_velocity": True,
        "no_lensing_halo_or_gravity_payload": True,
    }
    gate_results = {name: bool(value) for name, value in gate_results.items()}
    decision = "passed" if all(gate_results.values()) else "failed_closed"
    luminosity_fractions = np.asarray(
        [row["analysis_mask_luminosity_fraction"] for row in sample_rows], dtype=float
    )
    rms_speeds = np.asarray(
        [row["luminosity_weighted_rms_speed_km_s"] for row in sample_rows], dtype=float
    )
    global_currents = np.asarray(
        [row["total_signed_los_current"] for row in sample_rows], dtype=float
    )
    global_seconds = np.asarray(
        [row["total_positive_los_second_moment"] for row in sample_rows], dtype=float
    )
    nonzero = (means[0] > 0.0) | (means[2] > 0.0)
    report: dict[str, Any] = {
        "protocol_version": config["protocol_version"],
        "generated_utc": datetime.now(UTC).isoformat(),
        "decision": decision,
        "config": str(config_path.relative_to(ROOT)).replace("\\", "/"),
        "config_sha256": config_hash,
        "implementation": {
            "runner": implementation["runner"],
            "runner_sha256": runner_hash,
        },
        "input_hashes": input_hashes,
        "grid": {
            "shape_yx": [ny, nx],
            "pixel_scale_arcsec": pixel_scale_arcsec,
            "all_rows_have_four_neighbors": True,
            "outside_analysis_mask_member_ids": sorted(
                outside_mask_members, key=lambda value: int(value)
            ),
        },
        "ensemble": {
            "draws": draws,
            "members_per_draw": len(reference_member_ids or set()),
            "rows": total_rows,
            "finite_luminosity_members_per_draw_minimum": min(finite_counts),
            "finite_luminosity_members_per_draw_median": float(np.median(finite_counts)),
            "finite_luminosity_members_per_draw_maximum": max(finite_counts),
            "missing_photometry_rows": total_missing_rows,
            "maximum_wcs_roundtrip_error_arcsec": max_roundtrip_error,
            "maximum_per_draw_relative_deposition_error": max_deposition_error,
            "minimum_normalized_pixel_cauchy_schwarz_margin": minimum_cs_margin,
            "minimum_second_moment_pixel": minimum_second_pixel,
            "minimum_global_beta_variance": minimum_global_variance,
        },
        "global_moment_diagnostics": {
            "relative_f814w_luminosity_mean": float(
                np.mean([row["total_relative_f814w_luminosity"] for row in sample_rows])
            ),
            "relative_f814w_luminosity_standard_deviation": float(
                np.std([row["total_relative_f814w_luminosity"] for row in sample_rows])
            ),
            "signed_current_mean": float(np.mean(global_currents)),
            "signed_current_standard_deviation": float(np.std(global_currents)),
            "positive_second_moment_mean": float(np.mean(global_seconds)),
            "positive_second_moment_standard_deviation": float(np.std(global_seconds)),
            "luminosity_weighted_rms_speed_km_s_mean": float(np.mean(rms_speeds)),
            "luminosity_weighted_rms_speed_km_s_standard_deviation": float(np.std(rms_speeds)),
            "analysis_mask_luminosity_fraction_minimum": float(np.min(luminosity_fractions)),
            "analysis_mask_luminosity_fraction_median": float(np.median(luminosity_fractions)),
            "analysis_mask_luminosity_fraction_maximum": float(np.max(luminosity_fractions)),
        },
        "target_blind_morphology": {
            "luminosity_centroid_pixel_xy": list(luminosity_centroid),
            "second_moment_centroid_pixel_xy": list(second_centroid),
            "second_moment_to_luminosity_centroid_offset_arcsec": centroid_separation_arcsec(
                second_centroid, luminosity_centroid, pixel_scale_arcsec
            ),
            "positive_current_centroid_pixel_xy": list(positive_current_centroid),
            "negative_current_centroid_pixel_xy": list(negative_current_centroid),
            "opposite_current_centroid_separation_arcsec": centroid_separation_arcsec(
                positive_current_centroid, negative_current_centroid, pixel_scale_arcsec
            ),
            "pearson_luminosity_vs_second_moment_nonzero_pixels": float(
                np.corrcoef(means[0][nonzero], means[2][nonzero])[0, 1]
            ),
        },
        "gate_results": gate_results,
        "outputs": {
            "moment_maps": {
                "path": str(map_path.relative_to(ROOT)).replace("\\", "/"),
                "sha256": sha256(map_path),
                "bytes": map_path.stat().st_size,
            },
            "sample_global_moments": {
                "path": str(sample_path.relative_to(ROOT)).replace("\\", "/"),
                "sha256": sha256(sample_path),
                "bytes": sample_path.stat().st_size,
            },
            "figure": {
                "path": str(figure_path.relative_to(ROOT)).replace("\\", "/"),
                "sha256": sha256(figure_path),
                "bytes": figure_path.stat().st_size,
            },
        },
        "claim_boundary": config["claim_boundary"],
        "smoothing_length_or_response_amplitude_selected": False,
        "absolute_mass_inferred": False,
        "transverse_velocity_imputed": False,
        "missing_luminosity_imputed": False,
        "lensing_or_halo_payload_opened": False,
        "gravity_formula_or_parameter_changed": False,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if decision != "passed":
        raise RuntimeError(f"V19BC failed closed: {gate_results}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    report = run(args.config.resolve())
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
