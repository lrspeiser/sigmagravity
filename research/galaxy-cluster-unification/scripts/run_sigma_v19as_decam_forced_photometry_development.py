#!/usr/bin/env python3
"""Run the V19AS development-only DECam forced-photometry comparison.

The runner deliberately refuses to measure a validation anchor.  It compares
three predetermined aperture/deblending rules on the ten development anchors
and writes a recommendation for a separately frozen validation protocol.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.nddata import Cutout2D
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales
from scipy.ndimage import binary_dilation, gaussian_filter, map_coordinates
from scipy.optimize import least_squares
from skimage.feature import peak_local_max
from skimage.segmentation import watershed

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "sigma_v19as_decam_forced_photometry_development.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def robust_sigma(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan")
    median = float(np.median(values))
    sigma = 1.4826 * float(np.median(np.abs(values - median)))
    if not np.isfinite(sigma) or sigma <= 0:
        sigma = float(np.std(values))
    return sigma


def fit_background_plane(
    data: np.ndarray,
    center_xy: tuple[float, float],
    pixel_scale_arcsec: float,
    inner_arcsec: float,
    outer_arcsec: float,
    excluded: np.ndarray,
) -> tuple[np.ndarray, float, int]:
    yy, xx = np.indices(data.shape, dtype=float)
    cx, cy = center_xy
    radius_arcsec = np.hypot(xx - cx, yy - cy) * pixel_scale_arcsec
    mask = (
        np.isfinite(data)
        & ~excluded
        & (radius_arcsec >= inner_arcsec)
        & (radius_arcsec <= outer_arcsec)
    )
    design = np.column_stack(
        [np.ones(mask.sum()), (xx[mask] - cx), (yy[mask] - cy)]
    )
    values = data[mask]
    if values.size < 100:
        raise RuntimeError("fewer than 100 usable background pixels")
    keep = np.ones(values.size, dtype=bool)
    coefficients = np.zeros(3, dtype=float)
    for _ in range(5):
        coefficients, *_ = np.linalg.lstsq(design[keep], values[keep], rcond=None)
        residual = values - design @ coefficients
        sigma = robust_sigma(residual[keep])
        if not np.isfinite(sigma) or sigma <= 0:
            raise RuntimeError("non-positive background noise")
        new_keep = np.abs(residual) <= 3.0 * sigma
        if np.array_equal(new_keep, keep):
            break
        keep = new_keep
        if keep.sum() < 100:
            raise RuntimeError("background clipping left fewer than 100 pixels")
    plane = coefficients[0] + coefficients[1] * (xx - cx) + coefficients[2] * (yy - cy)
    sigma = robust_sigma((data - plane)[mask & np.isfinite(data)])
    return plane, sigma, int(keep.sum())


def validation_exclusion_mask(
    shape: tuple[int, int],
    wcs: WCS,
    validation_coordinates: list[tuple[float, float]],
    radius_arcsec: float,
    pixel_scale_arcsec: float,
) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    yy, xx = np.indices(shape, dtype=float)
    radius_pixel = radius_arcsec / pixel_scale_arcsec
    for ra_deg, dec_deg in validation_coordinates:
        px, py = wcs.world_to_pixel(SkyCoord(ra_deg, dec_deg, unit="deg"))
        if -radius_pixel <= px < shape[1] + radius_pixel and -radius_pixel <= py < shape[0] + radius_pixel:
            mask |= np.hypot(xx - px, yy - py) <= radius_pixel
    return mask


def neighbour_mask(
    signal: np.ndarray,
    center_xy: tuple[float, float],
    fwhm_pixel: float,
    noise: float,
    excluded: np.ndarray,
    threshold_sigma: float,
) -> tuple[np.ndarray, int]:
    """Return non-target watershed pixels in detected positive-source islands."""

    cx, cy = center_xy
    psf_sigma = max(fwhm_pixel / 2.355, 0.7)
    smoothed = gaussian_filter(np.nan_to_num(signal, nan=0.0), psf_sigma)
    valid_noise = robust_sigma(smoothed[~excluded & np.isfinite(smoothed)])
    if not np.isfinite(valid_noise) or valid_noise <= 0:
        valid_noise = noise / max(2.0 * math.sqrt(math.pi) * psf_sigma, 1.0)
    detected = (smoothed > threshold_sigma * valid_noise) & ~excluded
    yy, xx = np.indices(signal.shape, dtype=float)
    detected |= np.hypot(xx - cx, yy - cy) <= max(fwhm_pixel, 2.0)

    min_distance = max(2, round(0.75 * fwhm_pixel))
    peaks = peak_local_max(
        smoothed,
        min_distance=min_distance,
        threshold_abs=threshold_sigma * valid_noise,
        labels=detected.astype(np.uint8),
        exclude_border=False,
    )
    target_yx = np.array([round(cy), round(cx)], dtype=int)
    target_yx[0] = np.clip(target_yx[0], 0, signal.shape[0] - 1)
    target_yx[1] = np.clip(target_yx[1], 0, signal.shape[1] - 1)

    markers = np.zeros(signal.shape, dtype=np.int32)
    markers[tuple(target_yx)] = 1
    next_label = 2
    for py, px in peaks:
        if math.hypot(px - cx, py - cy) < 0.75 * fwhm_pixel:
            continue
        markers[py, px] = next_label
        next_label += 1
    labels = watershed(-smoothed, markers=markers, mask=detected)
    neighbours = (labels > 1) | excluded
    neighbours = binary_dilation(neighbours, iterations=max(1, round(0.25 * fwhm_pixel)))
    return neighbours, next_label - 2


def aperture_fluxes(
    signal: np.ndarray,
    center_xy: tuple[float, float],
    radius_pixel: float,
    neighbours: np.ndarray,
) -> dict[str, tuple[float, int, int]]:
    """Measure raw, area-scaled, and 180-degree reconstructed aperture flux."""

    cx, cy = center_xy
    yy, xx = np.indices(signal.shape, dtype=float)
    aperture = np.hypot(xx - cx, yy - cy) <= radius_pixel
    finite = aperture & np.isfinite(signal)
    clean = finite & ~neighbours
    total_pixels = int(finite.sum())
    clean_pixels = int(clean.sum())
    if total_pixels == 0 or clean_pixels == 0:
        return {name: (float("nan"), clean_pixels, total_pixels) for name in (
            "raw", "area_scaled", "rotate180"
        )}

    raw = float(np.sum(signal[finite]))
    area_scaled = float(np.sum(signal[clean]) * total_pixels / clean_pixels)

    values = signal.copy()
    missing_y, missing_x = np.where(finite & neighbours)
    if missing_y.size:
        mirror_y = 2.0 * cy - missing_y
        mirror_x = 2.0 * cx - missing_x
        mirrored = map_coordinates(
            signal,
            [mirror_y, mirror_x],
            order=1,
            mode="constant",
            cval=np.nan,
        )
        mirror_mask_value = map_coordinates(
            neighbours.astype(float),
            [mirror_y, mirror_x],
            order=0,
            mode="constant",
            cval=1.0,
        )
        usable = np.isfinite(mirrored) & (mirror_mask_value < 0.5)
        values[missing_y[usable], missing_x[usable]] = mirrored[usable]
        values[missing_y[~usable], missing_x[~usable]] = np.nan
    reconstructed = aperture & np.isfinite(values)
    if reconstructed.sum() == 0:
        rotate180 = float("nan")
    else:
        rotate180 = float(np.sum(values[reconstructed]) * total_pixels / reconstructed.sum())

    return {
        "raw": (raw, total_pixels, total_pixels),
        "area_scaled": (area_scaled, clean_pixels, total_pixels),
        "rotate180": (rotate180, int(reconstructed.sum()), total_pixels),
    }


def affine_fit(features: np.ndarray, values: np.ndarray, ridge: float) -> np.ndarray:
    scale = 0.15

    def residual(parameters: np.ndarray) -> np.ndarray:
        data_residual = (features @ parameters - values) / scale
        penalty = math.sqrt(ridge) * parameters[1:]
        return np.concatenate([data_residual, penalty])

    result = least_squares(residual, np.zeros(features.shape[1]), loss="soft_l1")
    if not result.success:
        raise RuntimeError("development color regression failed")
    return result.x


def loo_color_error(
    aggregate: dict[tuple[str, str], float],
    bri: dict[str, dict[str, str]],
    development_ids: list[str],
    ridge: float,
) -> tuple[float, int]:
    outputs = [("g", "r"), ("r", "i"), ("i", "z")]
    errors: list[float] = []
    complete = 0
    for held_out in development_ids:
        required = [(held_out, band) in aggregate for band in ("g", "r", "i", "z")]
        if not all(required):
            continue
        complete += 1
        training = [member for member in development_ids if member != held_out]
        for first, second in outputs:
            usable = [
                member
                for member in training
                if (member, first) in aggregate and (member, second) in aggregate
            ]
            if len(usable) < 6:
                continue
            features = []
            values = []
            for member in usable:
                row = bri[member]
                features.append(
                    [
                        1.0,
                        (float(row["B"]) - float(row["R"]) - 2.4) / 1.0,
                        (float(row["R"]) - float(row["I"]) - 1.1) / 0.5,
                    ]
                )
                values.append(aggregate[(member, first)] - aggregate[(member, second)])
            parameters = affine_fit(np.asarray(features), np.asarray(values), ridge)
            row = bri[held_out]
            held_features = np.asarray(
                [
                    1.0,
                    (float(row["B"]) - float(row["R"]) - 2.4) / 1.0,
                    (float(row["R"]) - float(row["I"]) - 1.1) / 0.5,
                ]
            )
            predicted = float(held_features @ parameters)
            observed = aggregate[(held_out, first)] - aggregate[(held_out, second)]
            errors.append(abs(predicted - observed))
    return (float(np.median(errors)) if errors else float("inf"), complete)


def summarize(
    measurements: list[dict[str, Any]],
    config: dict[str, Any],
    bri: dict[str, dict[str, str]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    grouped: dict[tuple[str, float, str, str], list[float]] = defaultdict(list)
    catalog_deltas: dict[tuple[str, float], list[float]] = defaultdict(list)
    totals: dict[tuple[str, float], int] = defaultdict(int)
    valid: dict[tuple[str, float], int] = defaultdict(int)
    for row in measurements:
        key = (row["variant"], float(row["aperture_diameter_arcsec"]))
        totals[key] += 1
        magnitude = float(row["magnitude"])
        if np.isfinite(magnitude):
            valid[key] += 1
            grouped[(row["variant"], key[1], row["member_id"], row["filter"])].append(
                magnitude
            )
            catalog = float(row["catalog_mag_aper4"])
            if key[1] == 4.0 and np.isfinite(catalog) and catalog < 90:
                catalog_deltas[key].append(magnitude - catalog)

    aggregate_rows: list[dict[str, Any]] = []
    ranking_rows: list[dict[str, Any]] = []
    development_ids = config["split"]["development_ids"]
    for variant in config["measurement"]["variants"]:
        for diameter in config["measurement"]["aperture_diameters_arcsec"]:
            aggregate: dict[tuple[str, str], float] = {}
            scatters: list[float] = []
            for member in development_ids:
                for band in config["measurement"]["color_filters"]:
                    values = np.asarray(grouped.get((variant, float(diameter), member, band), []))
                    median = float(np.median(values)) if values.size else float("nan")
                    scatter = robust_sigma(values) if values.size >= 2 else float("nan")
                    if np.isfinite(median):
                        aggregate[(member, band)] = median
                    if np.isfinite(scatter):
                        scatters.append(scatter)
                    aggregate_rows.append(
                        {
                            "variant": variant,
                            "aperture_diameter_arcsec": diameter,
                            "member_id": member,
                            "filter": band,
                            "valid_exposures": int(values.size),
                            "median_magnitude": median,
                            "robust_scatter_mag": scatter,
                        }
                    )
            loo_error, complete = loo_color_error(
                aggregate,
                bri,
                development_ids,
                float(config["development_ranking"]["ridge_penalty"]),
            )
            key = (variant, float(diameter))
            ranking_rows.append(
                {
                    "variant": variant,
                    "aperture_diameter_arcsec": diameter,
                    "valid_measurement_fraction": valid[key] / totals[key],
                    "complete_griz_development_objects": complete,
                    "median_repeatability_scatter_mag": float(np.median(scatters))
                    if scatters
                    else float("inf"),
                    "median_absolute_catalog_aper4_delta_mag": float(
                        np.median(np.abs(catalog_deltas[key]))
                    )
                    if catalog_deltas[key]
                    else float("nan"),
                    "leave_one_out_color_mae_mag": loo_error,
                }
            )
    return aggregate_rows, ranking_rows


def choose_recommendation(rows: list[dict[str, Any]], development_count: int) -> dict[str, Any]:
    def score(row: dict[str, Any]) -> tuple[float, ...]:
        complete_penalty = development_count - int(row["complete_griz_development_objects"])
        return (
            float(complete_penalty),
            -float(row["valid_measurement_fraction"]),
            float(row["median_repeatability_scatter_mag"]),
            float(row["leave_one_out_color_mae_mag"]),
            float(row["median_absolute_catalog_aper4_delta_mag"])
            if np.isfinite(float(row["median_absolute_catalog_aper4_delta_mag"]))
            else float("inf"),
        )

    return min(rows, key=score)


def run(config_path: Path) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    for parent in config["parent_artifacts"]:
        path = ROOT / parent["path"]
        if sha256(path) != parent["sha256"]:
            raise RuntimeError(f"parent artifact hash changed: {parent['path']}")

    manifest = read_csv(ROOT / config["inputs"]["anchor_measurements"])
    plans = read_csv(ROOT / config["inputs"]["resolved_image_plan"])
    catalog_rows = read_csv(ROOT / config["inputs"]["catalog_measurements"])
    bri_rows = read_csv(ROOT / config["inputs"]["commissioning_sample"])
    bri = {row["object_id"]: row for row in bri_rows}

    expected_development = set(config["split"]["development_ids"])
    expected_validation = set(config["split"]["validation_ids"])
    actual_development = {row["member_id"] for row in manifest if row["split"] == "development"}
    actual_validation = {row["member_id"] for row in manifest if row["split"] == "validation"}
    if actual_development != expected_development or actual_validation != expected_validation:
        raise RuntimeError("frozen anchor split changed")
    development_rows = [row for row in manifest if row["member_id"] in expected_development]
    if any(row["split"] != "development" for row in development_rows):
        raise RuntimeError("validation anchor reached development measurement list")
    if len(development_rows) != int(config["gates"]["exact_development_measurements"]):
        raise RuntimeError("development measurement count changed")

    validation_coordinates = sorted(
        {
            (float(row["ra_deg"]), float(row["dec_deg"]))
            for row in manifest
            if row["member_id"] in expected_validation
        }
    )
    plan_lookup = {(row["exposure"], row["sia_extension"]): row for row in plans}
    catalog_lookup = {row["measid"]: row for row in catalog_rows}
    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in development_rows:
        grouped[(row["exposure"], row["sia_extension"])].append(row)

    outputs = config["outputs"]
    measurement_rows: list[dict[str, Any]] = []
    group_audit: list[dict[str, Any]] = []
    for group_key in sorted(grouped):
        if group_key not in plan_lookup:
            raise RuntimeError(f"missing image plan for {group_key}")
        plan = plan_lookup[group_key]
        image_path = ROOT / plan["output_path"]
        with fits.open(image_path, memmap=True, checksum=False) as hdul:
            primary = hdul[0].header
            image_hdu = next(hdu for hdu in hdul if hdu.data is not None and hdu.data.ndim == 2)
            image = image_hdu.data
            image_wcs = WCS(image_hdu.header)
            scales = proj_plane_pixel_scales(image_wcs.celestial) * 3600.0
            pixel_scale = float(np.mean(scales))
            fwhm_pixel = float(image_hdu.header.get("FWHM", np.nan))
            if not np.isfinite(fwhm_pixel) or fwhm_pixel <= 0:
                seeing = float(primary.get("SEEING", np.nan))
                fwhm_pixel = seeing / pixel_scale
            magzero = float(primary.get("MAGZERO", np.nan))
            if not np.isfinite(magzero):
                raise RuntimeError(f"missing MAGZERO in {image_path.name}")

            for row in grouped[group_key]:
                position = SkyCoord(float(row["ra_deg"]), float(row["dec_deg"]), unit="deg")
                cutout = Cutout2D(
                    image,
                    position,
                    size=float(config["measurement"]["cutout_size_arcsec"]) * u.arcsec,
                    wcs=image_wcs,
                    mode="partial",
                    fill_value=np.nan,
                    copy=True,
                )
                cutout_wcs = cutout.wcs.celestial
                cx, cy = cutout_wcs.world_to_pixel(position)
                excluded = validation_exclusion_mask(
                    cutout.data.shape,
                    cutout_wcs,
                    validation_coordinates,
                    float(config["split"]["validation_mask_radius_arcsec"]),
                    pixel_scale,
                )
                plane, noise, background_pixels = fit_background_plane(
                    np.asarray(cutout.data, dtype=float),
                    (float(cx), float(cy)),
                    pixel_scale,
                    float(config["measurement"]["background_annulus_arcsec"][0]),
                    float(config["measurement"]["background_annulus_arcsec"][1]),
                    excluded,
                )
                signal = np.asarray(cutout.data, dtype=float) - plane
                neighbours, neighbour_count = neighbour_mask(
                    signal,
                    (float(cx), float(cy)),
                    fwhm_pixel,
                    noise,
                    excluded,
                    float(config["measurement"]["detection_threshold_sigma"]),
                )
                catalog = catalog_lookup[row["measid"]]
                for diameter in config["measurement"]["aperture_diameters_arcsec"]:
                    fluxes = aperture_fluxes(
                        signal,
                        (float(cx), float(cy)),
                        0.5 * float(diameter) / pixel_scale,
                        neighbours,
                    )
                    for variant, (flux, usable_pixels, total_pixels) in fluxes.items():
                        magnitude = (
                            magzero - 2.5 * math.log10(flux)
                            if np.isfinite(flux) and flux > 0
                            else float("nan")
                        )
                        measurement_rows.append(
                            {
                                "member_id": row["member_id"],
                                "nsc_id": row["nsc_id"],
                                "measid": row["measid"],
                                "exposure": row["exposure"],
                                "sia_extension": row["sia_extension"],
                                "filter": row["filter"],
                                "variant": variant,
                                "aperture_diameter_arcsec": diameter,
                                "flux": flux,
                                "magnitude": magnitude,
                                "magzero": magzero,
                                "pixel_scale_arcsec": pixel_scale,
                                "fwhm_pixel": fwhm_pixel,
                                "background_noise": noise,
                                "background_pixels": background_pixels,
                                "detected_neighbours": neighbour_count,
                                "usable_aperture_pixels": usable_pixels,
                                "total_aperture_pixels": total_pixels,
                                "catalog_mag_aper4": catalog["mag_aper4"],
                                "catalog_magerr_aper4": catalog["magerr_aper4"],
                                "catalog_flags": catalog["flags"],
                                "image_path": plan["output_path"],
                                "image_sha256": plan.get("sha256", ""),
                            }
                        )
            group_audit.append(
                {
                    "exposure": group_key[0],
                    "sia_extension": group_key[1],
                    "development_members": ";".join(sorted(row["member_id"] for row in grouped[group_key])),
                    "image_path": plan["output_path"],
                    "image_sha256": sha256(image_path),
                    "validation_coordinates_masked_before_detection": True,
                }
            )

    aggregate_rows, ranking_rows = summarize(measurement_rows, config, bri)
    recommendation = choose_recommendation(ranking_rows, len(expected_development))

    measurement_fields = list(measurement_rows[0])
    aggregate_fields = list(aggregate_rows[0])
    ranking_fields = list(ranking_rows[0])
    group_fields = list(group_audit[0])
    measurement_path = ROOT / outputs["measurements"]
    aggregate_path = ROOT / outputs["aggregates"]
    ranking_path = ROOT / outputs["ranking"]
    group_path = ROOT / outputs["group_audit"]
    write_csv(measurement_path, measurement_rows, measurement_fields)
    write_csv(aggregate_path, aggregate_rows, aggregate_fields)
    write_csv(ranking_path, ranking_rows, ranking_fields)
    write_csv(group_path, group_audit, group_fields)

    report = {
        "protocol_version": config["protocol_version"],
        "decision": "development_completed_validation_still_sealed",
        "counts": {
            "development_anchors": len(expected_development),
            "validation_anchors_measured": 0,
            "development_measurements": len(development_rows),
            "development_image_groups": len(grouped),
            "output_measurement_rows": len(measurement_rows),
        },
        "recommendation_for_separate_validation_freeze": recommendation,
        "ranking": ranking_rows,
        "leakage_controls": {
            "validation_rows_rejected_by_runner": True,
            "validation_coordinate_masks_applied_before_detection": True,
            "lensing_halo_gravity_data_read": False,
        },
        "outputs": {
            "measurements": measurement_path.relative_to(ROOT).as_posix(),
            "measurements_sha256": sha256(measurement_path),
            "aggregates": aggregate_path.relative_to(ROOT).as_posix(),
            "aggregates_sha256": sha256(aggregate_path),
            "ranking": ranking_path.relative_to(ROOT).as_posix(),
            "ranking_sha256": sha256(ranking_path),
            "group_audit": group_path.relative_to(ROOT).as_posix(),
            "group_audit_sha256": sha256(group_path),
        },
        "claim_boundary": config["claim_boundary"],
    }
    report_path = ROOT / outputs["report"]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    report = run(args.config.resolve())
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
