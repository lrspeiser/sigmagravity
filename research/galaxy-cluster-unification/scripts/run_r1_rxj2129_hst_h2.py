#!/usr/bin/env python3
"""Execute the frozen RX J2129 HST H2 two-band centroid measurement."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.wcs import WCS
from scipy import ndimage
from scipy.optimize import least_squares
from scipy.signal import fftconvolve


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/r1_rxj2129_hst_h2_centroid_execution_protocol.json"


def _resolve(relative: str) -> Path:
    return ROOT / relative


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _seed(label: str) -> int:
    return int.from_bytes(hashlib.sha256(label.encode()).digest()[:8], "big")


def _drizzled_wcs(header: fits.Header) -> WCS:
    wcs = WCS(header)
    ctypes = [str(value).upper() for value in wcs.wcs.ctype]
    if all("-SIP" not in value for value in ctypes):
        wcs.sip = None
    return wcs


def _pixel_affine(coefficients: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    center, scale = 2499.5, 2500.0
    matrix = np.array(
        [
            [coefficients[1, 0], coefficients[2, 0]],
            [coefficients[1, 1], coefficients[2, 1]],
        ],
        dtype=float,
    )
    offset = center + scale * coefficients[0] - matrix @ np.array([center, center])
    return matrix, offset


def _to_reference(xy: np.ndarray, matrix: np.ndarray, offset: np.ndarray) -> np.ndarray:
    return np.asarray(xy, dtype=float) @ matrix.T + offset


def _to_native(xy: np.ndarray, matrix: np.ndarray, offset: np.ndarray) -> np.ndarray:
    inverse = np.linalg.inv(matrix)
    return (np.asarray(xy, dtype=float) - offset) @ inverse.T


def _psf_kernel(
    coefficients: np.ndarray,
    x: float,
    y: float,
    rules: dict[str, Any],
) -> np.ndarray:
    design = np.array([1.0, (x - 2499.5) / 2500.0, (y - 2499.5) / 2500.0])
    log_fwhm, e1, e2, log_beta = design @ np.asarray(coefficients, dtype=float)
    fwhm = float(np.exp(log_fwhm))
    beta = float(np.exp(log_beta))
    ellipticity = float(np.hypot(e1, e2))
    theta = 0.5 * math.atan2(e2, e1)
    if not np.isfinite([fwhm, beta, ellipticity]).all() or ellipticity >= 0.95:
        raise ValueError("invalid local PSF field")
    ratio = (1.0 + ellipticity) / (1.0 - ellipticity)
    major, minor = fwhm * math.sqrt(ratio), fwhm / math.sqrt(ratio)
    if not (1.0 <= major <= 8.0 and 1.0 <= minor <= 8.0 and 1.2 <= beta <= 10.0):
        raise ValueError("local PSF is outside frozen bounds")
    size = int(rules["stamp_size_pixels"])
    yy, xx = np.indices((size, size), dtype=float)
    center = (size - 1) / 2.0
    cosine, sine = math.cos(theta), math.sin(theta)
    xp = cosine * (xx - center) + sine * (yy - center)
    yp = -sine * (xx - center) + cosine * (yy - center)
    factor = 2.0 * math.sqrt(2.0 ** (1.0 / beta) - 1.0)
    radius2 = (xp / (major / factor)) ** 2 + (yp / (minor / factor)) ** 2
    kernel = (1.0 + radius2) ** (-beta)
    return kernel / kernel.sum()


def _source_and_plane(
    parameters: np.ndarray,
    xx: np.ndarray,
    yy: np.ndarray,
    start_xy: np.ndarray,
    psf: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cx, cy = start_xy + parameters[:2]
    source = np.zeros_like(xx, dtype=float)
    cursor = 2
    for _ in range(3):
        log_flux, log_sigma, axis_ratio, theta = parameters[cursor : cursor + 4]
        cursor += 4
        sigma_major = math.exp(float(log_sigma))
        sigma_minor = sigma_major * float(axis_ratio)
        cosine, sine = math.cos(float(theta)), math.sin(float(theta))
        xp = cosine * (xx - cx) + sine * (yy - cy)
        yp = -sine * (xx - cx) + cosine * (yy - cy)
        component = np.exp(-0.5 * ((xp / sigma_major) ** 2 + (yp / sigma_minor) ** 2))
        total = component.sum()
        if total <= 0 or not np.isfinite(total):
            raise ValueError("invalid intrinsic Gaussian")
        source += math.exp(float(log_flux)) * component / total
    source = fftconvolve(source, psf, mode="same")
    c0, slope_x, slope_y = parameters[-3:]
    plane = c0 + slope_x * (xx - start_xy[0]) + slope_y * (yy - start_xy[1])
    return source, plane, source + plane


def _initial_plane(
    data: np.ndarray,
    weight: np.ndarray,
    xx: np.ndarray,
    yy: np.ndarray,
    start_xy: np.ndarray,
    background: np.ndarray,
) -> tuple[np.ndarray, float]:
    if int(background.sum()) < 30:
        raise ValueError("fewer than 30 valid segmentation-zero background pixels")
    design = np.column_stack(
        (
            np.ones(int(background.sum())),
            xx[background] - start_xy[0],
            yy[background] - start_xy[1],
        )
    )
    root_weight = np.sqrt(weight[background])
    coefficients = np.linalg.lstsq(
        design * root_weight[:, None], data[background] * root_weight, rcond=None
    )[0]
    sigma = float(np.median(1.0 / np.sqrt(weight[background])))
    return coefficients, max(sigma, np.finfo(float).eps)


def _fit_stamp(
    data: np.ndarray,
    weight: np.ndarray,
    valid: np.ndarray,
    background: np.ndarray,
    xx: np.ndarray,
    yy: np.ndarray,
    start_xy: np.ndarray,
    psf: np.ndarray,
    config: dict[str, Any],
    initial_override: np.ndarray | None = None,
    max_nfev: int | None = None,
) -> dict[str, Any]:
    source_rules = config["source_model"]
    if int(valid.sum()) < int(config["masking"]["minimum_fit_pixels"]):
        raise ValueError("fewer than the frozen minimum fit pixels")
    plane0, median_sigma = _initial_plane(data, weight, xx, yy, start_xy, background)
    plane_image = plane0[0] + plane0[1] * (xx - start_xy[0]) + plane0[2] * (yy - start_xy[1])
    target = valid & ~background
    positive_flux = float(np.clip(data[target] - plane_image[target], 0, None).sum())
    f0 = max(positive_flux, median_sigma * math.sqrt(int(valid.sum())), 1e-12)
    lower = [-18.0, -18.0]
    upper = [18.0, 18.0]
    for _ in range(3):
        lower.extend(
            [
                math.log(1e-4 * f0),
                math.log(float(source_rules["sigma_major_bounds_pixels"][0])),
                float(source_rules["axis_ratio_bounds"][0]),
                float(source_rules["position_angle_bounds_radians"][0]),
            ]
        )
        upper.extend(
            [
                math.log(100.0 * f0),
                math.log(float(source_rules["sigma_major_bounds_pixels"][1])),
                float(source_rules["axis_ratio_bounds"][1]),
                float(source_rules["position_angle_bounds_radians"][1]),
            ]
        )
    lower.extend(
        [
            plane0[0] - 20 * median_sigma,
            plane0[1] - 5 * median_sigma,
            plane0[2] - 5 * median_sigma,
        ]
    )
    upper.extend(
        [
            plane0[0] + 20 * median_sigma,
            plane0[1] + 5 * median_sigma,
            plane0[2] + 5 * median_sigma,
        ]
    )
    lower_array, upper_array = np.asarray(lower), np.asarray(upper)
    base = [0.0, 0.0]
    for fraction, sigma in zip((0.5, 0.3, 0.2), (1.0, 3.0, 6.0)):
        base.extend([math.log(fraction * f0), math.log(sigma), 0.7, 0.0])
    base.extend(plane0.tolist())
    base = np.asarray(base, dtype=float)

    def residual(parameters: np.ndarray) -> np.ndarray:
        try:
            model = _source_and_plane(parameters, xx, yy, start_xy, psf)[2]
        except (ValueError, OverflowError):
            return np.full(int(valid.sum()), 1e30)
        return ((data - model) * np.sqrt(weight))[valid]

    if initial_override is None:
        starts = [base.copy()]
        for k in range(11):
            start = base.copy()
            angle = 2 * math.pi * k / 11
            start[:2] = [1.5 * math.cos(angle), 1.5 * math.sin(angle)]
            starts.append(start)
    else:
        starts = [np.clip(np.asarray(initial_override), lower_array + 1e-10, upper_array - 1e-10)]
    fits = []
    for start in starts:
        try:
            result = least_squares(
                residual,
                start,
                bounds=(lower_array, upper_array),
                loss="linear",
                x_scale="jac",
                max_nfev=max_nfev or 2000,
                ftol=1e-9,
                xtol=1e-9,
                gtol=1e-9,
            )
        except Exception:
            continue
        chi2 = float(np.sum(residual(result.x) ** 2))
        if result.success and np.isfinite(chi2) and np.isfinite(result.x).all():
            fits.append((chi2, result))
    if not fits:
        raise ValueError("no converged finite optimizer start")
    chi2, best = min(fits, key=lambda item: item[0])
    source, plane, model = _source_and_plane(best.x, xx, yy, start_xy, psf)
    noise = math.sqrt(float(np.sum(1.0 / weight[valid])))
    snr = float(source[valid].sum() / max(noise, np.finfo(float).eps))
    return {
        "parameters": best.x,
        "source": source,
        "plane": plane,
        "model": model,
        "residual": data - model,
        "centroid_native_xy": start_xy + best.x[:2],
        "chi_square": chi2,
        "degrees_of_freedom": int(valid.sum() - best.x.size),
        "source_flux_SNR": snr,
        "optimizer_evaluations": int(best.nfev),
        "optimizer_starts_converged": len(fits),
    }


def _background_patches(residual: np.ndarray, background: np.ndarray) -> list[np.ndarray]:
    patches = []
    for y in range(residual.shape[0] - 2):
        for x in range(residual.shape[1] - 2):
            selection = background[y : y + 3, x : x + 3]
            if selection.all():
                patches.append(residual[y : y + 3, x : x + 3].copy())
    return patches


def _resample_blocks(
    patches: list[np.ndarray], shape: tuple[int, int], rng: np.random.Generator
) -> np.ndarray:
    result = np.empty(shape, dtype=float)
    for y in range(0, shape[0], 3):
        for x in range(0, shape[1], 3):
            patch = patches[int(rng.integers(0, len(patches)))]
            height, width = min(3, shape[0] - y), min(3, shape[1] - x)
            result[y : y + height, x : x + width] = patch[:height, :width]
    return result


def _poisson_noise(
    source: np.ndarray, header: fits.Header, rng: np.random.Generator
) -> np.ndarray:
    unit = str(header.get("BUNIT", "")).upper().replace(" ", "")
    exposure = float(header.get("EXPTIME", 0.0))
    if unit in {"ELECTRONS/S", "ELECTRON/S", "E-/S"} and exposure > 0:
        expected = np.clip(source, 0, None) * exposure
        return (rng.poisson(expected) - expected) / exposure
    if unit in {"ELECTRONS", "ELECTRON", "E-"}:
        expected = np.clip(source, 0, None)
        return rng.poisson(expected) - expected
    raise ValueError(f"unsupported BUNIT for frozen Poisson rule: {unit!r}")


def _stamp(
    science: np.ndarray,
    weight: np.ndarray,
    union_mask: np.ndarray,
    start_xy: np.ndarray,
    size: int,
    matrix: np.ndarray | None,
    offset: np.ndarray | None,
) -> dict[str, np.ndarray]:
    half = size // 2
    center = np.rint(start_xy).astype(int)
    x0, x1 = center[0] - half, center[0] + half + 1
    y0, y1 = center[1] - half, center[1] + half + 1
    if x0 < 0 or y0 < 0 or x1 > science.shape[1] or y1 > science.shape[0]:
        raise ValueError("stamp crosses image boundary")
    data = np.asarray(science[y0:y1, x0:x1], dtype=float)
    local_weight = np.asarray(weight[y0:y1, x0:x1], dtype=float)
    yy, xx = np.indices(data.shape, dtype=float)
    xx += x0
    yy += y0
    if matrix is None:
        sampled_mask = np.asarray(union_mask[y0:y1, x0:x1], dtype=np.uint8)
    else:
        reference = _to_reference(np.column_stack((xx.ravel(), yy.ravel())), matrix, offset)
        sampled_mask = ndimage.map_coordinates(
            union_mask.astype(float),
            [reference[:, 1], reference[:, 0]],
            order=0,
            mode="constant",
            cval=0,
        ).reshape(data.shape).astype(np.uint8)
    labels, _ = ndimage.label(sampled_mask > 0, structure=np.ones((3, 3), dtype=bool))
    local_x = int(np.rint(start_xy[0])) - x0
    local_y = int(np.rint(start_xy[1])) - y0
    target_label = int(labels[local_y, local_x])
    if target_label == 0:
        raise ValueError("published-coordinate pixel is not in a frozen target component")
    other_component = (labels > 0) & (labels != target_label)
    valid = np.isfinite(data) & np.isfinite(local_weight) & (local_weight > 0) & ~other_component
    background = valid & (sampled_mask == 0)
    return {
        "data": data,
        "weight": local_weight,
        "valid": valid,
        "background": background,
        "xx": xx,
        "yy": yy,
        "sampled_mask": sampled_mask,
    }


def _sky_offsets(
    reference_xy: np.ndarray, wcs: WCS, published: SkyCoord
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    coordinates = wcs.pixel_to_world(reference_xy[..., 0], reference_xy[..., 1])
    coordinates = coordinates.transform_to(published.frame)
    offsets = coordinates.transform_to(published.skyoffset_frame())
    return np.asarray(offsets.lon.arcsec), np.asarray(offsets.lat.arcsec), coordinates


def _validate_hashes(config: dict[str, Any]) -> None:
    records = [config["parent_protocol"], config["H1_gate"]]
    records.extend(config["inputs"][name] for name in (
        "coordinate_ledger", "registration_draws", "union_segmentation", "PSF_field"
    ))
    for record in records:
        path = _resolve(record.get("path", record.get("report")))
        if not path.is_file() or path.stat().st_size != int(record["bytes"]):
            raise ValueError(f"missing or size-mismatched frozen input: {path}")
        if _sha256(path) != record["sha256"]:
            raise ValueError(f"checksum mismatch: {path}")
    if _sha256(Path(__file__)) != config["implementation"]["runner_sha256"]:
        raise ValueError("runner differs from its pre-pixel frozen checksum")
    audit_path = _resolve(config["implementation"]["static_audit_report"])
    if not audit_path.is_file():
        raise ValueError("required pre-pixel H2 static audit report is absent")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if (
        audit.get("status") != "pass"
        or audit.get("runner_sha256") != config["implementation"]["runner_sha256"]
        or not all(audit.get("gates", {}).values())
        or audit.get("HST_arc_pixels_accessed_during_this_static_audit") is not False
    ):
        raise ValueError("pre-pixel H2 static audit is not valid for this runner")


def _fit_band(
    image_id: str,
    band: str,
    published: SkyCoord,
    start_native: np.ndarray,
    science: np.ndarray,
    weight: np.ndarray,
    header: fits.Header,
    union_mask: np.ndarray,
    field: np.ndarray,
    field_draws: np.ndarray,
    matrix: np.ndarray,
    offset: np.ndarray,
    f814_wcs: WCS,
    config: dict[str, Any],
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    size = int(config["pixel_geometry"]["stamp_size_pixels"])
    is_f125 = band == "F125W"
    stamp = _stamp(
        science,
        weight,
        union_mask,
        start_native,
        size,
        matrix if is_f125 else None,
        offset if is_f125 else None,
    )
    psf = _psf_kernel(field, start_native[0], start_native[1], config["local_PSF"])
    fit = _fit_stamp(
        stamp["data"], stamp["weight"], stamp["valid"], stamp["background"],
        stamp["xx"], stamp["yy"], start_native, psf, config,
    )
    reference_xy = _to_reference(fit["centroid_native_xy"][None, :], matrix, offset)[0] if is_f125 else fit["centroid_native_xy"]
    east, north, sky = _sky_offsets(reference_xy[None, :], f814_wcs, published)
    patches = _background_patches(fit["residual"], stamp["background"])
    if len(patches) < int(config["masking"]["minimum_background_3x3_blocks"]):
        raise ValueError("fewer than the frozen minimum background 3x3 patches")
    draws = int(config["bootstrap"]["draws"])
    draw_offsets = np.full((draws, 2), np.nan)
    draw_reference = np.full((draws, 2), np.nan)
    for draw in range(draws):
        rng = np.random.default_rng(_seed(f"RXJ2129|H2|{image_id}|{band}|{draw}"))
        try:
            draw_psf = _psf_kernel(
                field_draws[draw], start_native[0], start_native[1], config["local_PSF"]
            )
            source, plane, _ = _source_and_plane(
                fit["parameters"], stamp["xx"], stamp["yy"], start_native, draw_psf
            )
            synthetic = (
                source + plane + _resample_blocks(patches, source.shape, rng)
                + _poisson_noise(source, header, rng)
            )
            refit = _fit_stamp(
                synthetic, stamp["weight"], stamp["valid"], stamp["background"],
                stamp["xx"], stamp["yy"], start_native, draw_psf, config,
                initial_override=fit["parameters"], max_nfev=1000,
            )
            draw_xy = refit["centroid_native_xy"]
            if is_f125:
                draw_xy = _to_reference(draw_xy[None, :], matrix, offset)[0]
            draw_reference[draw] = draw_xy
            draw_east, draw_north, _ = _sky_offsets(draw_xy[None, :], f814_wcs, published)
            draw_offsets[draw] = [draw_east[0], draw_north[0]]
        except Exception:
            continue
    successful = np.isfinite(draw_offsets).all(axis=1)
    fraction = float(successful.mean())
    standard_error = (
        np.std(draw_offsets[successful], axis=0, ddof=1)
        if successful.sum() >= 2 else np.array([np.nan, np.nan])
    )
    floored = np.maximum(
        standard_error,
        float(config["likelihood_and_metrics"]["minimum_per_coordinate_standard_error_arcsec"]),
    )
    central_limit = size * float(config["likelihood_and_metrics"]["centroid_inside_central_stamp_fraction"]) / 2
    row = {
        "image_id": image_id,
        "band": band,
        "fit_success": True,
        "failure_reason": "",
        "published_start_native_x": float(start_native[0]),
        "published_start_native_y": float(start_native[1]),
        "centroid_native_x": float(fit["centroid_native_xy"][0]),
        "centroid_native_y": float(fit["centroid_native_xy"][1]),
        "centroid_reference_x": float(reference_xy[0]),
        "centroid_reference_y": float(reference_xy[1]),
        "centroid_ra_deg": float(sky.ra.deg[0]),
        "centroid_dec_deg": float(sky.dec.deg[0]),
        "published_offset_east_arcsec": float(east[0]),
        "published_offset_north_arcsec": float(north[0]),
        "source_flux_SNR": float(fit["source_flux_SNR"]),
        "chi_square": float(fit["chi_square"]),
        "degrees_of_freedom": int(fit["degrees_of_freedom"]),
        "fit_pixels": int(stamp["valid"].sum()),
        "background_3x3_patch_count": len(patches),
        "optimizer_starts_converged": int(fit["optimizer_starts_converged"]),
        "bootstrap_draws_successful": int(successful.sum()),
        "bootstrap_successful_fraction": fraction,
        "standard_error_east_arcsec_raw": float(standard_error[0]),
        "standard_error_north_arcsec_raw": float(standard_error[1]),
        "standard_error_east_arcsec_floored": float(floored[0]),
        "standard_error_north_arcsec_floored": float(floored[1]),
        "centroid_inside_central_half": bool(np.all(np.abs(fit["centroid_native_xy"] - start_native) <= central_limit)),
    }
    return row, draw_offsets, draw_reference


def _failure_row(image_id: str, band: str, error: Exception) -> dict[str, Any]:
    return {"image_id": image_id, "band": band, "fit_success": False, "failure_reason": str(error)}


def _write_diagnostic(path: Path, image_ledger: pd.DataFrame, band_ledger: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for band, marker in (("F814W", "o"), ("F125W", "s")):
        rows = band_ledger[(band_ledger["band"] == band) & band_ledger["fit_success"]]
        axes[0].scatter(rows["published_offset_east_arcsec"], rows["published_offset_north_arcsec"], label=band, marker=marker)
        axes[1].scatter(rows["source_flux_SNR"], rows["bootstrap_successful_fraction"], label=band, marker=marker)
    axes[0].set(xlabel="east offset from published (arcsec)", ylabel="north offset (arcsec)")
    axes[1].axhline(0.95, color="k", ls="--", lw=0.8)
    axes[1].set(xlabel="source flux S/N", ylabel="bootstrap success fraction")
    axes[2].bar(image_ledger["image_id"].astype(str), image_ledger["cross_band_separation_arcsec"].fillna(0))
    axes[2].axhline(0.2, color="k", ls="--", lw=0.8)
    axes[2].tick_params(axis="x", rotation=90)
    axes[2].set(ylabel="cross-band separation (arcsec)")
    axes[0].legend()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def run(config: dict[str, Any]) -> dict[str, Any]:
    _validate_hashes(config)
    if not config["authorization"]["execute_H2_after_static_audit"]:
        raise ValueError("H2 arc-pixel execution is not authorized by the frozen config")
    parent = json.loads(_resolve(config["parent_protocol"]["path"]).read_text())
    h1 = json.loads(_resolve(config["H1_gate"]["report"]).read_text())
    if h1["status"] != "pass" or h1["authorization"]["execute_H2_arc_centroids"] is not True:
        raise ValueError("H1 does not authorize H2")
    ledger = pd.read_csv(_resolve(config["inputs"]["coordinate_ledger"]["path"]), dtype={"image_id": str})
    ledger = ledger[(ledger["likelihood_included"] == True) & (ledger["redshift_kind"] == "spectroscopic")].reset_index(drop=True)  # noqa: E712
    if len(ledger) != int(config["inputs"]["coordinate_ledger"]["required_rows"]):
        raise ValueError("immutable H2 ledger row count changed")
    registration = np.load(_resolve(config["inputs"]["registration_draws"]["path"]), allow_pickle=False)
    matrix, offset = registration["pixel_affine_matrix"], registration["pixel_affine_offset"]
    psf_fields = np.load(_resolve(config["inputs"]["PSF_field"]["path"]), allow_pickle=False)
    with fits.open(_resolve(config["inputs"]["union_segmentation"]["path"]), memmap=True) as hdus:
        union_mask = np.asarray(hdus[0].data, dtype=np.uint8)
    images: dict[str, tuple[np.ndarray, np.ndarray, fits.Header]] = {}
    for band in ("F814W", "F125W"):
        item = parent["inputs"][band]
        if _sha256(_resolve(item["path"])) != item["sha256"] or _sha256(_resolve(item["weight_path"])) != item["weight_sha256"]:
            raise ValueError(f"{band} science/weight checksum changed")
        images[band] = (
            fits.getdata(_resolve(item["path"]), memmap=True),
            fits.getdata(_resolve(item["weight_path"]), memmap=True),
            fits.getheader(_resolve(item["path"])),
        )
    f814_wcs = _drizzled_wcs(images["F814W"][2])
    band_rows: list[dict[str, Any]] = []
    image_rows: list[dict[str, Any]] = []
    all_offsets = np.full((len(ledger), 2, int(config["bootstrap"]["draws"]), 2), np.nan)
    all_reference = np.full_like(all_offsets, np.nan)
    for image_index, item in ledger.iterrows():
        image_id = str(item["image_id"])
        published = SkyCoord(float(item["ra_deg"]), float(item["dec_deg"]), unit="deg")
        x_ref, y_ref = f814_wcs.world_to_pixel(published)
        reference_start = np.array([float(x_ref), float(y_ref)])
        fitted_rows: dict[str, dict[str, Any]] = {}
        for band_index, band in enumerate(("F814W", "F125W")):
            start = reference_start if band == "F814W" else _to_native(reference_start[None, :], matrix, offset)[0]
            science, weight, header = images[band]
            try:
                row, draw_offsets, draw_reference = _fit_band(
                    image_id, band, published, start, science, weight, header, union_mask,
                    psf_fields[f"{band.lower()}_field_coefficients"],
                    psf_fields[f"{band.lower()}_field_bootstrap"], matrix, offset,
                    f814_wcs, config,
                )
                all_offsets[image_index, band_index] = draw_offsets
                all_reference[image_index, band_index] = draw_reference
            except Exception as error:
                row = _failure_row(image_id, band, error)
            band_rows.append(row)
            fitted_rows[band] = row
            print(f"H2 {image_id} {band}: {'fit' if row['fit_success'] else row['failure_reason']}", flush=True)
        complete = all(row["fit_success"] for row in fitted_rows.values())
        separation = np.nan
        cross_band_gate = False
        if complete:
            first, second = fitted_rows["F814W"], fitted_rows["F125W"]
            separation = float(
                SkyCoord(first["centroid_ra_deg"], first["centroid_dec_deg"], unit="deg")
                .separation(SkyCoord(second["centroid_ra_deg"], second["centroid_dec_deg"], unit="deg"))
                .arcsec
            )
            band_gates = []
            for row in fitted_rows.values():
                band_gates.append(bool(
                    row["source_flux_SNR"] >= float(config["image_acceptance"]["minimum_source_flux_SNR_each_band"])
                    and row["bootstrap_successful_fraction"] >= float(config["image_acceptance"]["minimum_successful_bootstrap_fraction_each_band"])
                    and row["centroid_inside_central_half"]
                    and row["standard_error_east_arcsec_floored"] <= float(config["image_acceptance"]["maximum_per_coordinate_standard_error_arcsec"])
                    and row["standard_error_north_arcsec_floored"] <= float(config["image_acceptance"]["maximum_per_coordinate_standard_error_arcsec"])
                ))
            cross_band_gate = separation <= float(config["image_acceptance"]["maximum_cross_band_centroid_difference_arcsec"])
            accepted = bool(all(band_gates) and cross_band_gate)
        else:
            accepted = False
        image_rows.append({
            "image_id": image_id,
            "source_family": str(item["source_family"]),
            "published_ra_deg": float(item["ra_deg"]),
            "published_dec_deg": float(item["dec_deg"]),
            "both_band_fits_complete": complete,
            "cross_band_separation_arcsec": separation,
            "cross_band_gate": cross_band_gate,
            "accepted": accepted,
        })
    band_ledger, image_ledger = pd.DataFrame(band_rows), pd.DataFrame(image_rows)
    accepted_ids = set(image_ledger.loc[image_ledger["accepted"], "image_id"].astype(str))
    required_inner = set(config["image_acceptance"]["required_inner_images"])
    gates = {
        "all_immutable_images_attempted": len(image_ledger) == 21 and len(band_ledger) == 42,
        "minimum_images_accepted": int(image_ledger["accepted"].sum()) >= int(config["image_acceptance"]["minimum_total_images_accepted"]),
        "all_required_inner_images_accepted": required_inner.issubset(accepted_ids),
    }
    passed = bool(all(gates.values()))
    outputs = config["outputs"]
    band_path, image_path = _resolve(outputs["H2_band_fit_ledger"]), _resolve(outputs["H2_image_ledger"])
    draws_path, diagnostic_path = _resolve(outputs["H2_centroid_draws"]), _resolve(outputs["H2_diagnostic"])
    for path in (band_path, image_path, draws_path):
        path.parent.mkdir(parents=True, exist_ok=True)
    band_ledger.to_csv(band_path, index=False)
    image_ledger.to_csv(image_path, index=False)
    np.savez_compressed(
        draws_path,
        image_ids=ledger["image_id"].astype(str).to_numpy(dtype="U16"),
        bands=np.asarray(["F814W", "F125W"], dtype="U8"),
        east_north_arcsec=all_offsets,
        reference_pixels=all_reference,
        successful=np.isfinite(all_offsets).all(axis=-1),
    )
    _write_diagnostic(diagnostic_path, image_ledger, band_ledger)
    output_records = {}
    for name in ("H2_band_fit_ledger", "H2_image_ledger", "H2_centroid_draws", "H2_diagnostic"):
        path = _resolve(outputs[name])
        output_records[name] = {"path": outputs[name], "bytes": path.stat().st_size, "sha256": _sha256(path)}
    report = {
        "report_version": "R1B3-RXJ2129-HST-H2-0.1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_version": config["protocol_version"],
        "status": "pass" if passed else "fail",
        "scope": "independent two-band HST arc centroid measurement only",
        "images_attempted": len(image_ledger),
        "band_fits_attempted": len(band_ledger),
        "images_accepted": int(image_ledger["accepted"].sum()),
        "required_inner_images": sorted(required_inner),
        "accepted_inner_images": sorted(required_inner & accepted_ids),
        "gates": gates,
        "outputs": output_records,
        "authorization": {
            "assemble_H3_covariance": passed,
            "use_lens_or_gravity_model": False,
            "infer_dynamical_or_Weyl_response": False,
            "fit_new_force_or_action": False,
        },
    }
    report_path = _resolve(outputs["H2_report"])
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def self_test(config: dict[str, Any]) -> dict[str, Any]:
    size = int(config["pixel_geometry"]["stamp_size_pixels"])
    yy, xx = np.indices((size, size), dtype=float)
    start = np.array([(size - 1) / 2, (size - 1) / 2])
    field = np.zeros((3, 4))
    field[0] = [math.log(3.0), 0.05, -0.03, math.log(2.5)]
    psf = _psf_kernel(field, 2499.5, 2499.5, config["local_PSF"])
    truth = [0.35, -0.25]
    for flux, sigma, ratio, theta in ((80, 1, 0.7, 0.1), (45, 3, 0.6, -0.4), (25, 6, 0.8, 0.8)):
        truth.extend([math.log(flux), math.log(sigma), ratio, theta])
    truth.extend([0.02, 0.0002, -0.0001])
    truth = np.asarray(truth)
    source, plane, model = _source_and_plane(truth, xx, yy, start, psf)
    rng = np.random.default_rng(20260727)
    weight = np.full((size, size), 4e4)
    data = model + rng.normal(0, 1 / np.sqrt(weight))
    valid = np.ones_like(data, dtype=bool)
    radius = np.hypot(xx - start[0], yy - start[1])
    background = radius >= 12
    fit = _fit_stamp(data, weight, valid, background, xx, yy, start, psf, config)
    patches = _background_patches(fit["residual"], background)
    sampled = _resample_blocks(patches, data.shape, rng)
    centroid_error = float(np.linalg.norm(fit["centroid_native_xy"] - (start + truth[:2])))
    test_wcs = WCS(naxis=2)
    test_wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    test_wcs.wcs.crpix = [1.0, 1.0]
    test_wcs.wcs.crval = [322.4, 0.09]
    test_wcs.wcs.cdelt = [-0.065 / 3600, 0.065 / 3600]
    test_wcs.wcs.radesys = "FK5"
    test_wcs.wcs.equinox = 2000.0
    frame_east, frame_north, _ = _sky_offsets(
        np.array([[0.0, 0.0]]), test_wcs, SkyCoord(322.4, 0.09, unit="deg", frame="icrs")
    )
    gates = {
        "PSF_normalized": bool(abs(psf.sum() - 1) < 1e-12),
        "PSF_finite_nonnegative": bool(np.isfinite(psf).all() and (psf >= 0).all()),
        "synthetic_centroid_recovered_within_0p15_pixel": centroid_error <= 0.15,
        "twelve_starts_executed": fit["optimizer_starts_converged"] == 12,
        "moving_block_shape_exact": bool(
            sampled.shape == data.shape and np.isfinite(sampled).all()
        ),
        "equivalent_astrometric_frames_are_transformed_before_offsets": bool(
            np.isfinite(frame_east[0]) and np.isfinite(frame_north[0])
        ),
    }
    return {"status": "pass" if all(gates.values()) else "fail", "centroid_error_pixels": centroid_error, "gates": gates, "HST_pixels_accessed": False}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    report = self_test(config) if args.self_test else run(config)
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report["status"] == "pass" else 1)


if __name__ == "__main__":
    main()
