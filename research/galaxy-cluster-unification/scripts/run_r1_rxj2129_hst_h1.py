"""Execute the frozen RX J2129 HST H1 registration, mask, and PSF gate."""

from __future__ import annotations

import hashlib
import json
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
from scipy.interpolate import RegularGridInterpolator
from scipy.optimize import least_squares
from scipy.spatial import cKDTree


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/r1_rxj2129_hst_centroid_covariance_protocol.json"
PIXEL_CENTER = 2499.5
PIXEL_SCALE = 2500.0


def _resolve(path: str) -> Path:
    return ROOT / path


def _seed(text: str) -> int:
    return int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "big")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_image(specification: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, fits.Header]:
    science_path = _resolve(specification["path"])
    weight_path = _resolve(specification["weight_path"])
    if science_path.stat().st_size != specification["bytes"]:
        raise ValueError(f"Science byte count changed: {science_path}")
    if weight_path.stat().st_size != specification["weight_bytes"]:
        raise ValueError(f"Weight byte count changed: {weight_path}")
    if _sha256(science_path).upper() != specification["sha256"]:
        raise ValueError(f"Science checksum changed: {science_path}")
    if _sha256(weight_path).upper() != specification["weight_sha256"]:
        raise ValueError(f"Weight checksum changed: {weight_path}")
    with fits.open(science_path, memmap=True) as science_hdul, fits.open(
        weight_path, memmap=True
    ) as weight_hdul:
        science = np.asarray(science_hdul[0].data, dtype=np.float32)
        weight = np.asarray(weight_hdul[0].data, dtype=np.float32)
        header = science_hdul[0].header.copy()
    if science.shape != tuple(specification["shape"]) or weight.shape != science.shape:
        raise ValueError("Frozen HST shape changed")
    return science, weight, header


def _expand_tiles(values: np.ndarray, shape: tuple[int, int], tile_size: int) -> np.ndarray:
    y_centers = np.minimum(
        np.arange(values.shape[0], dtype=float) * tile_size + 0.5 * (tile_size - 1),
        shape[0] - 1,
    )
    x_centers = np.minimum(
        np.arange(values.shape[1], dtype=float) * tile_size + 0.5 * (tile_size - 1),
        shape[1] - 1,
    )
    interpolator = RegularGridInterpolator(
        (y_centers, x_centers), values, bounds_error=False, fill_value=None
    )
    expanded = np.empty(shape, dtype=np.float32)
    xx = np.arange(shape[1], dtype=float)
    for start in range(0, shape[0], 256):
        stop = min(start + 256, shape[0])
        yy = np.arange(start, stop, dtype=float)
        grid_y, grid_x = np.meshgrid(yy, xx, indexing="ij")
        points = np.column_stack((grid_y.ravel(), grid_x.ravel()))
        expanded[start:stop] = interpolator(points).reshape(stop - start, shape[1])
    return expanded


def _standardize(
    science: np.ndarray,
    weight: np.ndarray,
    settings: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    tile = int(settings["tile_size_pixels"])
    ny = (science.shape[0] + tile - 1) // tile
    nx = (science.shape[1] + tile - 1) // tile
    locations = np.full((ny, nx), np.nan, dtype=float)
    scales = np.full((ny, nx), np.nan, dtype=float)
    fractions = np.zeros((ny, nx), dtype=float)
    minimum_fraction = float(settings["minimum_valid_fraction_per_tile"])
    for iy in range(ny):
        ys = slice(iy * tile, min((iy + 1) * tile, science.shape[0]))
        for ix in range(nx):
            xs = slice(ix * tile, min((ix + 1) * tile, science.shape[1]))
            data = science[ys, xs]
            local_weight = weight[ys, xs]
            valid = np.isfinite(data) & np.isfinite(local_weight) & (local_weight > 0)
            fractions[iy, ix] = valid.mean()
            if fractions[iy, ix] < minimum_fraction:
                continue
            sample = np.asarray(data[valid], dtype=float)
            location = float(np.median(sample))
            mad_scale = float(1.4826 * np.median(np.abs(sample - location)))
            weight_floor = float(np.median(1.0 / np.sqrt(local_weight[valid])))
            locations[iy, ix] = location
            scales[iy, ix] = max(mad_scale, weight_floor, np.finfo(float).eps)
    valid_tile_fraction = float(np.isfinite(locations).mean())
    if valid_tile_fraction == 0:
        raise ValueError("No background tile meets the frozen per-tile coverage rule")
    for array in (locations, scales):
        invalid = ~np.isfinite(array)
        if invalid.any():
            _, indices = ndimage.distance_transform_edt(invalid, return_indices=True)
            array[invalid] = array[tuple(index[invalid] for index in indices)]
    background = _expand_tiles(locations, science.shape, tile)
    robust_scale = _expand_tiles(scales, science.shape, tile)
    valid = np.isfinite(science) & np.isfinite(weight) & (weight > 0)
    inverse_weight_sigma = np.full(science.shape, np.inf, dtype=np.float32)
    inverse_weight_sigma[valid] = 1.0 / np.sqrt(weight[valid])
    sigma = np.maximum(inverse_weight_sigma, robust_scale)
    standardized = np.zeros(science.shape, dtype=np.float32)
    standardized[valid] = (science[valid] - background[valid]) / sigma[valid]
    report = {
        "tile_shape": [ny, nx],
        "valid_tile_fraction_before_nearest_fill": valid_tile_fraction,
        "minimum_pixel_valid_fraction": float(fractions.min()),
        "median_background": float(np.median(locations)),
        "median_robust_sigma": float(np.median(scales)),
        "valid_pixel_fraction": float(valid.mean()),
    }
    return standardized, report


def _detect_sources(
    label: str,
    standardized: np.ndarray,
    science: np.ndarray,
    header: fits.Header,
    rules: dict[str, Any],
) -> pd.DataFrame:
    threshold = float(rules["threshold_local_sigma"])
    structure = np.ones((3, 3), dtype=bool)
    labels, count = ndimage.label(standardized >= threshold, structure=structure)
    slices = ndimage.find_objects(labels)
    rows: list[dict[str, Any]] = []
    radius = float(rules["centroid_aperture_radius_pixels"])
    edge = int(rules["minimum_edge_distance_pixels"])
    exptime = float(header.get("EXPTIME", 1.0))
    is_rate = "/S" in str(header.get("BUNIT", "")).upper()
    for component_id, component_slice in enumerate(slices, start=1):
        if component_slice is None:
            continue
        component = labels[component_slice] == component_id
        connected_pixels = int(component.sum())
        if connected_pixels < int(rules["minimum_connected_pixels"]):
            continue
        y0, x0 = component_slice[0].start, component_slice[1].start
        local_z = standardized[component_slice]
        weights = np.clip(local_z, 0.0, None) * component
        total = float(weights.sum())
        if total <= 0:
            continue
        local_y, local_x = np.indices(component.shape, dtype=float)
        x = float(x0 + (weights * local_x).sum() / total)
        y = float(y0 + (weights * local_y).sum() / total)
        half = 13
        xi, yi = int(round(x)), int(round(y))
        if xi - half < 0 or yi - half < 0 or xi + half >= science.shape[1] or yi + half >= science.shape[0]:
            continue
        stamp_z = standardized[yi - half : yi + half + 1, xi - half : xi + half + 1]
        stamp_science = science[yi - half : yi + half + 1, xi - half : xi + half + 1]
        yy, xx = np.indices(stamp_z.shape, dtype=float)
        expected_x = half + x - xi
        expected_y = half + y - yi
        rr = np.hypot(xx - expected_x, yy - expected_y)
        aperture = rr <= radius
        snr = float(stamp_z[aperture].sum() / np.sqrt(aperture.sum()))
        moment_weight = np.clip(stamp_z, 0.0, None) * aperture
        moment_sum = float(moment_weight.sum())
        if moment_sum <= 0:
            continue
        cx = float((moment_weight * xx).sum() / moment_sum)
        cy = float((moment_weight * yy).sum() / moment_sum)
        dx, dy = xx - cx, yy - cy
        covariance = np.array(
            [
                [(moment_weight * dx * dx).sum(), (moment_weight * dx * dy).sum()],
                [(moment_weight * dx * dy).sum(), (moment_weight * dy * dy).sum()],
            ],
            dtype=float,
        ) / moment_sum
        eigenvalues = np.linalg.eigvalsh(covariance)
        minor, major = np.sqrt(np.clip(eigenvalues, 1e-12, None))
        fwhm = float(2.355 * np.sqrt(minor * major))
        ellipticity = float((major - minor) / (major + minor))
        peak_stamp = stamp_science[
            int(round(cy)) - 1 : int(round(cy)) + 2,
            int(round(cx)) - 1 : int(round(cx)) + 2,
        ]
        peak_native = float(np.nanmax(peak_stamp))
        peak_electrons = peak_native * exptime if is_rate else peak_native
        equal_peaks = int(np.count_nonzero(peak_stamp == peak_native))
        preliminary = {
            "edge": bool(edge <= x < science.shape[1] - edge and edge <= y < science.shape[0] - edge),
            "snr": bool(snr >= float(rules["minimum_total_flux_SNR"])),
            "fwhm": bool(float(rules["minimum_moment_FWHM_pixels"]) <= fwhm <= float(rules["maximum_moment_FWHM_pixels"])),
            "ellipticity": bool(ellipticity <= float(rules["maximum_moment_ellipticity"])),
            "saturation": bool(peak_electrons < float(rules["saturation_limit_electrons_per_pixel"]) and equal_peaks < 2),
        }
        rows.append(
            {
                "band": label,
                "detection_id": f"{label}_{len(rows) + 1:05d}",
                "x_pixel_zero_indexed": x,
                "y_pixel_zero_indexed": y,
                "connected_pixels": connected_pixels,
                "total_flux_snr": snr,
                "positive_standardized_flux": moment_sum,
                "moment_fwhm_pixels": fwhm,
                "moment_ellipticity": ellipticity,
                "peak_electrons_per_pixel": peak_electrons,
                "equal_central_peak_pixels": equal_peaks,
                **{f"{key}_gate_passed": value for key, value in preliminary.items()},
                "preliminary_passed": bool(all(preliminary.values())),
            }
        )
    del labels
    table = pd.DataFrame(rows)
    if table.empty:
        raise ValueError(f"No {label} detections survived connected-pixel screening")
    positions = table[["x_pixel_zero_indexed", "y_pixel_zero_indexed"]].to_numpy()
    fluxes = table["positive_standardized_flux"].to_numpy()
    tree = cKDTree(positions)
    isolation_pass = np.ones(len(table), dtype=bool)
    radius_isolation = float(rules["isolation_radius_pixels"])
    max_ratio = float(rules["maximum_neighbor_to_source_flux_ratio"])
    for index, neighbors in enumerate(tree.query_ball_point(positions, radius_isolation)):
        neighbors = [item for item in neighbors if item != index]
        if neighbors and np.max(fluxes[neighbors] / max(fluxes[index], 1e-12)) > max_ratio:
            isolation_pass[index] = False
    table["isolation_gate_passed"] = isolation_pass
    table["accepted"] = table["preliminary_passed"] & isolation_pass
    wcs = WCS(header)
    # These are distortion-corrected drizzled mosaics. Their CTYPE values omit
    # -SIP, so the retained detector-frame SIP cards must not be applied again.
    if not any("-SIP" in str(header.get(key, "")) for key in ("CTYPE1", "CTYPE2")):
        wcs.sip = None
    sky = wcs.pixel_to_world(
        table["x_pixel_zero_indexed"].to_numpy(),
        table["y_pixel_zero_indexed"].to_numpy(),
    )
    table["ra_deg"] = sky.ra.deg
    table["dec_deg"] = sky.dec.deg
    table["raw_connected_component_count"] = count
    return table


def _normalized_design(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.column_stack(
        (
            np.ones(len(x)),
            (x - PIXEL_CENTER) / PIXEL_SCALE,
            (y - PIXEL_CENTER) / PIXEL_SCALE,
        )
    )


def _fit_affine(x: np.ndarray, y: np.ndarray, target: np.ndarray) -> np.ndarray:
    design = _normalized_design(x, y)
    normalized_target = (target - PIXEL_CENTER) / PIXEL_SCALE
    coefficients = np.linalg.lstsq(design, normalized_target, rcond=None)[0]
    nu = 4.0
    for _ in range(50):
        previous = coefficients.copy()
        residual = (normalized_target - design @ coefficients) * PIXEL_SCALE
        radial = np.hypot(residual[:, 0], residual[:, 1])
        scale = max(1.4826 * np.median(np.abs(radial - np.median(radial))), 1e-3)
        weights = (nu + 2.0) / (nu + (radial / scale) ** 2)
        root = np.sqrt(weights)
        coefficients = np.linalg.lstsq(
            design * root[:, None], normalized_target * root[:, None], rcond=None
        )[0]
        denominator = max(float(np.linalg.norm(previous)), 1e-12)
        if np.linalg.norm(coefficients - previous) / denominator <= 1e-10:
            break
    return coefficients


def _predict_affine(coefficients: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return PIXEL_CENTER + PIXEL_SCALE * (_normalized_design(x, y) @ coefficients)


def _pixel_affine(coefficients: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    matrix = np.array(
        [
            [coefficients[1, 0], coefficients[2, 0]],
            [coefficients[1, 1], coefficients[2, 1]],
        ]
    )
    offset = np.array(
        [
            PIXEL_CENTER + PIXEL_SCALE * coefficients[0, 0],
            PIXEL_CENTER + PIXEL_SCALE * coefficients[0, 1],
        ]
    ) - matrix @ np.array([PIXEL_CENTER, PIXEL_CENTER])
    return matrix, offset


def _match_and_register(
    f125: pd.DataFrame,
    f814: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any], np.ndarray]:
    left = f125[f125["accepted"]].reset_index(drop=True)
    right = f814[f814["accepted"]].reset_index(drop=True)
    left_sky = SkyCoord(left["ra_deg"].to_numpy(), left["dec_deg"].to_numpy(), unit="deg")
    right_sky = SkyCoord(right["ra_deg"].to_numpy(), right["dec_deg"].to_numpy(), unit="deg")
    nearest_right, separation, _ = left_sky.match_to_catalog_sky(right_sky)
    nearest_left, _, _ = right_sky.match_to_catalog_sky(left_sky)
    radius = float(config["sky_match_radius_arcsec"])
    pairs = [
        (left_index, int(right_index), float(separation[left_index].arcsec))
        for left_index, right_index in enumerate(nearest_right)
        if nearest_left[int(right_index)] == left_index
        and separation[left_index].arcsec <= radius
    ]
    rows = []
    for match_id, (left_index, right_index, initial_separation) in enumerate(pairs, start=1):
        lrow, rrow = left.iloc[left_index], right.iloc[right_index]
        rows.append(
            {
                "match_id": match_id,
                "F125W_detection_id": lrow["detection_id"],
                "F814W_detection_id": rrow["detection_id"],
                "F125W_x": lrow["x_pixel_zero_indexed"],
                "F125W_y": lrow["y_pixel_zero_indexed"],
                "F814W_x": rrow["x_pixel_zero_indexed"],
                "F814W_y": rrow["y_pixel_zero_indexed"],
                "initial_WCS_separation_arcsec": initial_separation,
                "F125W_snr": lrow["total_flux_snr"],
                "F814W_snr": rrow["total_flux_snr"],
            }
        )
    matches = pd.DataFrame(rows)
    if len(matches) < 6:
        raise ValueError("Too few mutual matches to fit a six-parameter affine transform")
    x = matches["F125W_x"].to_numpy(dtype=float)
    y = matches["F125W_y"].to_numpy(dtype=float)
    target = matches[["F814W_x", "F814W_y"]].to_numpy(dtype=float)
    coefficients = _fit_affine(x, y, target)
    predicted = _predict_affine(coefficients, x, y)
    residual = (target - predicted) * 0.065
    matches["affine_delta_ra_like_arcsec"] = residual[:, 0]
    matches["affine_delta_dec_like_arcsec"] = residual[:, 1]
    matches["affine_residual_arcsec"] = np.hypot(residual[:, 0], residual[:, 1])
    loo = np.empty((len(matches), 2), dtype=float)
    for omitted in range(len(matches)):
        keep = np.arange(len(matches)) != omitted
        trial = _fit_affine(x[keep], y[keep], target[keep])
        loo[omitted] = (
            target[omitted]
            - _predict_affine(trial, x[omitted : omitted + 1], y[omitted : omitted + 1])[0]
        ) * 0.065
    matches["leave_one_out_residual_arcsec"] = np.hypot(loo[:, 0], loo[:, 1])
    draws_requested = int(config["bootstrap_draws"])
    draws = np.full((draws_requested, 3, 2), np.nan, dtype=float)
    complete = 0
    for draw in range(draws_requested):
        rng = np.random.default_rng(_seed(f"RXJ2129|H1-registration|{draw}"))
        indices = rng.integers(0, len(matches), len(matches))
        if np.linalg.matrix_rank(_normalized_design(x[indices], y[indices])) < 3:
            continue
        draws[draw] = _fit_affine(x[indices], y[indices], target[indices])
        complete += 1
    report = {
        "accepted_F125W_sources": int(left.shape[0]),
        "accepted_F814W_sources": int(right.shape[0]),
        "mutual_match_count": int(len(matches)),
        "affine_coefficients_normalized": coefficients.tolist(),
        "fit_RMS_arcsec": float(np.sqrt(np.mean(matches["affine_residual_arcsec"] ** 2))),
        "cross_validated_RMS_arcsec": float(np.sqrt(np.mean(matches["leave_one_out_residual_arcsec"] ** 2))),
        "maximum_cross_validated_registration_RMS_arcsec": 0.12,
        "bootstrap_draws_requested": draws_requested,
        "bootstrap_draws_complete": complete,
        "bootstrap_full_rank_fraction": complete / draws_requested,
    }
    return matches, report, draws


def _warp_to_reference(image: np.ndarray, coefficients: np.ndarray) -> np.ndarray:
    forward, offset = _pixel_affine(coefficients)
    inverse = np.linalg.inv(forward)
    inverse_offset = -inverse @ offset
    matrix_yx = np.array(
        [[inverse[1, 1], inverse[1, 0]], [inverse[0, 1], inverse[0, 0]]]
    )
    offset_yx = np.array([inverse_offset[1], inverse_offset[0]])
    return ndimage.affine_transform(
        image,
        matrix_yx,
        offset=offset_yx,
        output_shape=image.shape,
        order=3,
        mode="constant",
        cval=0.0,
        prefilter=True,
    ).astype(np.float32)


def _segment(
    f814_standardized: np.ndarray,
    f125_standardized: np.ndarray,
    coefficients: np.ndarray,
    rules: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    registered = _warp_to_reference(f125_standardized, coefficients)
    detection = (f814_standardized + registered) / np.sqrt(2.0)
    labels, count = ndimage.label(
        detection >= float(rules["threshold_sigma"]),
        structure=np.ones((3, 3), dtype=bool),
    )
    sizes = np.bincount(labels.ravel())
    keep = sizes >= int(rules["minimum_connected_pixels"])
    keep[0] = False
    mask = keep[labels]
    mask = ndimage.binary_dilation(
        mask,
        structure=np.ones((3, 3), dtype=bool),
        iterations=int(rules["dilation_pixels"]),
    )
    report = {
        "raw_component_count": int(count),
        "retained_component_count": int(np.count_nonzero(keep)),
        "masked_pixels_after_dilation": int(mask.sum()),
        "masked_fraction_after_dilation": float(mask.mean()),
    }
    return mask.astype(np.uint8), report


def _fit_moffat(
    science: np.ndarray,
    weight: np.ndarray,
    x: float,
    y: float,
    rules: dict[str, Any],
) -> dict[str, Any]:
    size = int(rules["stamp_size_pixels"])
    half = size // 2
    xi, yi = int(round(x)), int(round(y))
    if xi - half < 0 or yi - half < 0 or xi + half >= science.shape[1] or yi + half >= science.shape[0]:
        return {"fit_success": False, "failure": "edge"}
    data = np.asarray(science[yi - half : yi + half + 1, xi - half : xi + half + 1], dtype=float)
    local_weight = np.asarray(weight[yi - half : yi + half + 1, xi - half : xi + half + 1], dtype=float)
    yy, xx = np.indices(data.shape, dtype=float)
    valid = np.isfinite(data) & np.isfinite(local_weight) & (local_weight > 0)
    rr = np.hypot(xx - half, yy - half)
    background = float(np.median(data[valid & (rr >= 11)]))
    amplitude = max(float(np.nanmax(data) - background), 1e-8)

    def model(parameters: np.ndarray) -> np.ndarray:
        bg, log_amp, dx, dy, log_fwhm_a, log_fwhm_b, theta, beta = parameters
        cosine, sine = np.cos(theta), np.sin(theta)
        xp = cosine * (xx - half - dx) + sine * (yy - half - dy)
        yp = -sine * (xx - half - dx) + cosine * (yy - half - dy)
        fwhm_factor = 2.0 * np.sqrt(2.0 ** (1.0 / beta) - 1.0)
        alpha_a = np.exp(log_fwhm_a) / fwhm_factor
        alpha_b = np.exp(log_fwhm_b) / fwhm_factor
        radius2 = (xp / alpha_a) ** 2 + (yp / alpha_b) ** 2
        return bg + np.exp(log_amp) * (1.0 + radius2) ** (-beta)

    sigma = np.full(data.shape, np.inf, dtype=float)
    sigma[valid] = 1.0 / np.sqrt(local_weight[valid])
    scale = max(float(np.median(sigma[valid])), np.finfo(float).eps)

    def residual(parameters: np.ndarray) -> np.ndarray:
        return ((data - model(parameters)) / np.maximum(sigma, scale))[valid]

    fwhm_bounds = rules["FWHM_bounds_pixels"]
    lower = [background - 5 * scale, np.log(amplitude) - 5, -2, -2, np.log(fwhm_bounds[0]), np.log(fwhm_bounds[0]), -np.pi, float(rules["beta_bounds"][0])]
    upper = [background + 5 * scale, np.log(amplitude) + 5, 2, 2, np.log(fwhm_bounds[1]), np.log(fwhm_bounds[1]), np.pi, float(rules["beta_bounds"][1])]
    initial = [background, np.log(amplitude), x - xi, y - yi, np.log(3.0), np.log(3.0), 0.0, 2.5]
    try:
        fit = least_squares(residual, initial, bounds=(lower, upper), loss="soft_l1", max_nfev=500)
    except Exception as error:
        return {"fit_success": False, "failure": str(error)}
    bg, log_amp, dx, dy, log_fwhm_a, log_fwhm_b, theta, beta = fit.x
    fwhm_a, fwhm_b = np.exp(log_fwhm_a), np.exp(log_fwhm_b)
    if fwhm_b > fwhm_a:
        fwhm_a, fwhm_b = fwhm_b, fwhm_a
        theta += 0.5 * np.pi
    ellipticity = (fwhm_a - fwhm_b) / (fwhm_a + fwhm_b)
    accepted = bool(
        fit.success
        and fwhm_bounds[0] <= fwhm_a <= fwhm_bounds[1]
        and fwhm_bounds[0] <= fwhm_b <= fwhm_bounds[1]
        and rules["beta_bounds"][0] <= beta <= rules["beta_bounds"][1]
    )
    return {
        "fit_success": accepted,
        "optimizer_success": bool(fit.success),
        "x_fit": float(xi + dx),
        "y_fit": float(yi + dy),
        "fwhm_major_pixels": float(fwhm_a),
        "fwhm_minor_pixels": float(fwhm_b),
        "geometric_fwhm_pixels": float(np.sqrt(fwhm_a * fwhm_b)),
        "ellipticity_e1": float(ellipticity * np.cos(2 * theta)),
        "ellipticity_e2": float(ellipticity * np.sin(2 * theta)),
        "beta": float(beta),
        "reduced_weighted_residual": float(np.mean(residual(fit.x) ** 2)),
    }


def _psf_field(
    band: str,
    science: np.ndarray,
    weight: np.ndarray,
    sources: pd.DataFrame,
    arc_sky: SkyCoord,
    header: fits.Header,
    rules: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any], np.ndarray, np.ndarray]:
    accepted = sources[sources["accepted"]].copy().reset_index(drop=True)
    source_sky = SkyCoord(accepted["ra_deg"].to_numpy(), accepted["dec_deg"].to_numpy(), unit="deg")
    separation = source_sky[:, None].separation(arc_sky[None, :]).arcsec.min(axis=1)
    accepted["minimum_arc_separation_arcsec"] = separation
    candidates = accepted[separation >= float(rules["arc_exclusion_radius_arcsec"])].copy()
    rows = []
    for source in candidates.itertuples(index=False):
        fit = _fit_moffat(
            science,
            weight,
            float(source.x_pixel_zero_indexed),
            float(source.y_pixel_zero_indexed),
            rules,
        )
        rows.append(
            {
                "band": band,
                "detection_id": source.detection_id,
                "x_pixel": source.x_pixel_zero_indexed,
                "y_pixel": source.y_pixel_zero_indexed,
                "minimum_arc_separation_arcsec": source.minimum_arc_separation_arcsec,
                **fit,
            }
        )
    ledger = pd.DataFrame(rows)
    successful = ledger[ledger["fit_success"]].copy()
    if len(successful) < 3:
        report = {
            "band": band,
            "candidates_after_arc_exclusion": int(len(candidates)),
            "successful_star_fits": int(len(successful)),
            "successful_fit_fraction": float(len(successful) / max(len(candidates), 1)),
            "field_coefficients": None,
            "bootstrap_draws_requested": int(rules["bootstrap_draws"]),
            "bootstrap_draws_complete": 0,
        }
        return (
            ledger,
            report,
            np.full((3, 4), np.nan),
            np.full((int(rules["bootstrap_draws"]), 3, 4), np.nan),
        )
    parameters = np.column_stack(
        (
            np.log(successful["geometric_fwhm_pixels"]),
            successful["ellipticity_e1"],
            successful["ellipticity_e2"],
            np.log(successful["beta"]),
        )
    )
    design = _normalized_design(successful["x_pixel"].to_numpy(), successful["y_pixel"].to_numpy())
    coefficients = np.linalg.lstsq(design, parameters, rcond=None)[0]
    draws_requested = int(rules["bootstrap_draws"])
    draws = np.full((draws_requested, 3, 4), np.nan, dtype=float)
    complete = 0
    for draw in range(draws_requested):
        rng = np.random.default_rng(_seed(f"RXJ2129|H1-PSF|{band}|{draw}"))
        indices = rng.integers(0, len(successful), len(successful))
        if np.linalg.matrix_rank(design[indices]) < 3:
            continue
        draws[draw] = np.linalg.lstsq(design[indices], parameters[indices], rcond=None)[0]
        complete += 1
    report = {
        "band": band,
        "candidates_after_arc_exclusion": int(len(candidates)),
        "successful_star_fits": int(len(successful)),
        "successful_fit_fraction": float(len(successful) / max(len(candidates), 1)),
        "field_coefficients": coefficients.tolist(),
        "bootstrap_draws_requested": draws_requested,
        "bootstrap_draws_complete": complete,
    }
    return ledger, report, coefficients, draws


def _write_diagnostic(
    path: Path,
    f814_z: np.ndarray,
    f125_z: np.ndarray,
    detections: dict[str, pd.DataFrame],
    matches: pd.DataFrame,
    segmentation: np.ndarray,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11, 10))
    stride = 10
    axes[0, 0].imshow(np.clip(f814_z[::stride, ::stride], -1, 8), origin="lower", cmap="gray")
    selected = detections["F814W"].query("accepted")
    axes[0, 0].scatter(selected["x_pixel_zero_indexed"] / stride, selected["y_pixel_zero_indexed"] / stride, s=8, facecolors="none", edgecolors="tab:red")
    axes[0, 0].set_title(f"F814W accepted compact sources ({len(selected)})")
    axes[0, 1].imshow(np.clip(f125_z[::stride, ::stride], -1, 8), origin="lower", cmap="gray")
    selected = detections["F125W"].query("accepted")
    axes[0, 1].scatter(selected["x_pixel_zero_indexed"] / stride, selected["y_pixel_zero_indexed"] / stride, s=8, facecolors="none", edgecolors="tab:orange")
    axes[0, 1].set_title(f"F125W accepted compact sources ({len(selected)})")
    axes[1, 0].scatter(
        matches["affine_delta_ra_like_arcsec"],
        matches["affine_delta_dec_like_arcsec"],
        s=14,
    )
    axes[1, 0].axhline(0, color="0.5", lw=0.7)
    axes[1, 0].axvline(0, color="0.5", lw=0.7)
    axes[1, 0].set_xlabel("affine residual x (arcsec)")
    axes[1, 0].set_ylabel("affine residual y (arcsec)")
    axes[1, 0].set_title("Matched-source residuals")
    axes[1, 1].imshow(segmentation[::stride, ::stride], origin="lower", cmap="binary")
    axes[1, 1].set_title("Frozen union segmentation mask")
    for axis in axes.flat:
        axis.set_aspect("equal")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def run() -> dict[str, Any]:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    if config["authorization"]["use_lens_or_gravity_model"]:
        raise ValueError("H1 cannot use a lens or gravity model")
    freeze = config["H1_execution_freeze"]
    outputs = config["outputs"]
    data: dict[str, tuple[np.ndarray, np.ndarray, fits.Header]] = {}
    standardized: dict[str, np.ndarray] = {}
    detections: dict[str, pd.DataFrame] = {}
    backgrounds: dict[str, Any] = {}
    for band in ("F814W", "F125W"):
        science, weight, header = _read_image(config["inputs"][band])
        z, background = _standardize(science, weight, freeze["background"])
        detections[band] = _detect_sources(
            band,
            z,
            science,
            header,
            freeze["compact_source_detection"],
        )
        data[band] = (science, weight, header)
        standardized[band] = z
        backgrounds[band] = background
    match_ledger, registration, registration_draws = _match_and_register(
        detections["F125W"],
        detections["F814W"],
        freeze["matching_and_affine_fit"],
    )
    coefficients = np.asarray(registration["affine_coefficients_normalized"], dtype=float)
    segmentation, segmentation_report = _segment(
        standardized["F814W"],
        standardized["F125W"],
        coefficients,
        freeze["segmentation_implementation"],
    )
    arc_table = pd.read_csv(_resolve(config["inputs"]["coordinate_ledger"]))
    arc_table = arc_table[
        arc_table["likelihood_included"].astype(bool)
        & (arc_table["redshift_kind"] == "spectroscopic")
    ]
    arc_sky = SkyCoord(arc_table["ra_deg"].to_numpy(), arc_table["dec_deg"].to_numpy(), unit="deg")
    psf_ledgers = []
    psf_reports = {}
    psf_payload: dict[str, np.ndarray] = {}
    for band in ("F814W", "F125W"):
        science, weight, header = data[band]
        ledger, psf_report, field, draws = _psf_field(
            band,
            science,
            weight,
            detections[band],
            arc_sky,
            header,
            freeze["spatial_PSF"],
        )
        psf_ledgers.append(ledger)
        psf_reports[band] = psf_report
        psf_payload[f"{band.lower()}_field_coefficients"] = field
        psf_payload[f"{band.lower()}_field_bootstrap"] = draws

    detection_path = _resolve(outputs["H1_detection_ledger"])
    match_path = _resolve(outputs["H1_match_ledger"])
    registration_path = _resolve(outputs["H1_registration_draws"])
    segmentation_path = _resolve(outputs["H1_union_segmentation"])
    psf_ledger_path = _resolve(outputs["H1_PSF_star_ledger"])
    psf_path = _resolve(outputs["H1_PSF_field"])
    for path in (detection_path, match_path, registration_path, segmentation_path, psf_ledger_path, psf_path):
        path.parent.mkdir(parents=True, exist_ok=True)
    pd.concat(detections.values(), ignore_index=True).to_csv(detection_path, index=False)
    match_ledger.to_csv(match_path, index=False)
    affine_matrix, affine_offset = _pixel_affine(coefficients)
    np.savez_compressed(
        registration_path,
        coefficients_normalized=coefficients,
        pixel_affine_matrix=affine_matrix,
        pixel_affine_offset=affine_offset,
        bootstrap_coefficients_normalized=registration_draws,
    )
    segmentation_header = data["F814W"][2].copy()
    segmentation_header["BUNIT"] = "MASK"
    segmentation_header.add_history("Frozen RX J2129 HST H1 union segmentation, protocol 0.2")
    fits.PrimaryHDU(segmentation, header=segmentation_header).writeto(
        segmentation_path, overwrite=True, checksum=True
    )
    pd.concat(psf_ledgers, ignore_index=True).to_csv(psf_ledger_path, index=False)
    np.savez_compressed(psf_path, **psf_payload)
    _write_diagnostic(
        _resolve(outputs["H1_diagnostic"]),
        standardized["F814W"],
        standardized["F125W"],
        detections,
        match_ledger,
        segmentation,
    )

    gates = {
        "minimum_registration_matches": registration["mutual_match_count"] >= int(config["registration"]["minimum_matches"]),
        "cross_validated_registration_RMS": registration["cross_validated_RMS_arcsec"] <= float(config["registration"]["maximum_cross_validated_registration_RMS_arcsec"]),
        "registration_bootstrap": registration["bootstrap_full_rank_fraction"] >= float(freeze["matching_and_affine_fit"]["minimum_full_rank_draw_fraction"]) and registration["bootstrap_draws_complete"] == int(config["registration"]["bootstrap_draws"]),
        "union_segmentation_nonempty": segmentation_report["retained_component_count"] > 0,
        "minimum_PSF_stars_each_band": all(report["successful_star_fits"] >= int(config["psf"]["minimum_stars"]) for report in psf_reports.values()),
        "PSF_fit_fraction_each_band": all(report["successful_fit_fraction"] >= float(freeze["spatial_PSF"]["minimum_successful_fit_fraction"]) for report in psf_reports.values()),
        "PSF_bootstrap_each_band": all(report["bootstrap_draws_complete"] == int(freeze["spatial_PSF"]["bootstrap_draws"]) for report in psf_reports.values()),
    }
    passed = bool(all(gates.values()))
    output_records = {}
    for name, relative in outputs.items():
        path = _resolve(relative)
        if path.is_file() and name != "H1_report":
            output_records[name] = {
                "path": relative,
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
    report = {
        "report_version": "R1B3-RXJ2129-HST-H1-0.1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_version": config["protocol_version"],
        "status": "pass" if passed else "fail",
        "scope": "H1 registration, union segmentation, and spatial empirical PSF only",
        "backgrounds": backgrounds,
        "detections": {
            band: {
                "connected_components_screened": int(len(table)),
                "accepted_compact_sources": int(table["accepted"].sum()),
            }
            for band, table in detections.items()
        },
        "registration": registration,
        "segmentation": segmentation_report,
        "PSF": psf_reports,
        "gates": gates,
        "outputs": output_records,
        "authorization": {
            "execute_H2_arc_centroids": passed,
            "assemble_H3_covariance": False,
            "use_lens_or_gravity_model": False,
            "infer_dynamical_or_Weyl_response": False,
            "fit_new_force_or_action": False,
        },
    }
    report_path = _resolve(outputs["H1_report"])
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    report = run()
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report["status"] == "pass" else 1)


if __name__ == "__main__":
    main()
