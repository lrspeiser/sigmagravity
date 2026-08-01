"""Foreground-star astrometry for optical baryonic maps."""

from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path

import astropy.units as u
import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
from astropy.stats import sigma_clipped_stats
from astropy.wcs import WCS
from astropy.wcs.utils import fit_wcs_from_points
from scipy.spatial import cKDTree
from skimage.feature import peak_local_max
from skimage.measure import ransac
from skimage.transform import SimilarityTransform

Array = np.ndarray

ORIENTATION_MATRICES = {
    "identity": np.array([[1.0, 0.0], [0.0, 1.0]]),
    "rotate_90": np.array([[0.0, -1.0], [1.0, 0.0]]),
    "rotate_180": np.array([[-1.0, 0.0], [0.0, -1.0]]),
    "rotate_270": np.array([[0.0, 1.0], [-1.0, 0.0]]),
    "flip_x": np.array([[-1.0, 0.0], [0.0, 1.0]]),
    "flip_y": np.array([[1.0, 0.0], [0.0, -1.0]]),
    "swap": np.array([[0.0, 1.0], [1.0, 0.0]]),
    "negative_swap": np.array([[0.0, -1.0], [-1.0, 0.0]]),
}


@dataclass(frozen=True)
class AstrometricFit:
    wcs: WCS
    diagnostics: dict[str, float | int | str]
    matched_pixel_xy: Array
    matched_sky: SkyCoord
    residual_pixel: Array


def read_vizier_gaia_tsv(path: Path) -> pd.DataFrame:
    lines = [
        line
        for line in Path(path).read_text(encoding="utf-8", errors="strict").splitlines()
        if line.strip() and not line.startswith("#")
    ]
    if len(lines) < 4 or lines[0].split("\t")[:4] != [
        "Source",
        "RA_ICRS",
        "DE_ICRS",
        "Gmag",
    ]:
        raise ValueError(f"{path} is not the frozen Gaia VizieR table")
    frame = pd.read_csv(io.StringIO("\n".join([lines[0], *lines[3:]])), sep="\t")
    for column in ("RA_ICRS", "DE_ICRS", "Gmag"):
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    return frame


def detect_registration_sources(
    image: Array,
    *,
    pixel_scale_arcsec: float,
    sigma_clip: float,
    threshold_sigma: float,
    minimum_separation_arcsec: float,
    exclusion_radius_arcsec: float,
    maximum_sources: int,
) -> tuple[Array, dict[str, float | int]]:
    values = np.squeeze(np.asarray(image, dtype=float))
    if values.ndim != 2 or not np.all(np.isfinite(values)):
        raise ValueError("registration image must be finite and two-dimensional")
    if pixel_scale_arcsec <= 0.0:
        raise ValueError("pixel scale must be positive")
    mean, median, deviation = sigma_clipped_stats(
        values, sigma=float(sigma_clip), maxiters=10
    )
    minimum_distance = max(3, int(minimum_separation_arcsec / pixel_scale_arcsec))
    peaks_yx = peak_local_max(
        values,
        min_distance=minimum_distance,
        threshold_abs=float(median + threshold_sigma * deviation),
        num_peaks=int(maximum_sources),
        exclude_border=10,
    )
    detected_xy = peaks_yx[:, ::-1].astype(float)
    ny, nx = values.shape
    radius = np.hypot(detected_xy[:, 0] - nx / 2.0, detected_xy[:, 1] - ny / 2.0)
    detected_xy = detected_xy[radius > exclusion_radius_arcsec / pixel_scale_arcsec]
    if len(detected_xy) < 30:
        raise ValueError("too few non-galaxy optical peaks for an astrometric solution")
    return detected_xy, {
        "background_mean_counts": float(mean),
        "background_median_counts": float(median),
        "background_sigma_counts": float(deviation),
        "detected_sources": len(detected_xy),
        "minimum_detection_separation_pixel": int(minimum_distance),
    }


def _deduplicate_nearest(
    predicted_xy: Array, detected_xy: Array, maximum_distance: float
) -> tuple[Array, Array]:
    distances, indices = cKDTree(detected_xy).query(
        predicted_xy, distance_upper_bound=float(maximum_distance)
    )
    candidates = [
        (float(distance), int(source), int(detection))
        for source, (distance, detection) in enumerate(zip(distances, indices, strict=True))
        if np.isfinite(distance) and detection < len(detected_xy)
    ]
    selected_sources: list[int] = []
    selected_detections: list[int] = []
    used_detections: set[int] = set()
    for _, source, detection in sorted(candidates):
        if detection in used_detections:
            continue
        used_detections.add(detection)
        selected_sources.append(source)
        selected_detections.append(detection)
    return np.asarray(selected_sources, dtype=int), np.asarray(selected_detections, dtype=int)


def _translation_mode(
    predicted_xy: Array,
    detected_xy: Array,
    *,
    maximum_translation_pixel: float,
    bin_width_pixel: float,
) -> tuple[Array, int]:
    tree = cKDTree(detected_xy)
    offsets = []
    for predicted in predicted_xy:
        offsets.extend(
            detected_xy[index] - predicted
            for index in tree.query_ball_point(predicted, maximum_translation_pixel)
        )
    if not offsets:
        raise ValueError("no candidate catalog-to-image translations")
    offsets_array = np.asarray(offsets, dtype=float)
    bins_per_axis = int(np.ceil(2.0 * maximum_translation_pixel / bin_width_pixel))
    indices = np.floor(
        (offsets_array + maximum_translation_pixel) / bin_width_pixel
    ).astype(int)
    valid = np.all((indices >= 0) & (indices < bins_per_axis), axis=1)
    histogram = np.zeros((bins_per_axis, bins_per_axis), dtype=int)
    np.add.at(histogram, (indices[valid, 0], indices[valid, 1]), 1)
    peak = np.unravel_index(int(np.argmax(histogram)), histogram.shape)
    translation = (
        np.asarray(peak, dtype=float) * bin_width_pixel
        - maximum_translation_pixel
        + bin_width_pixel / 2.0
    )
    return translation, int(histogram[peak])


def solve_foreground_star_wcs(
    image: Array,
    *,
    catalog_center: SkyCoord,
    catalog_pixel_scale_arcsec: float,
    gaia_sky: SkyCoord,
    settings: dict,
) -> AstrometricFit:
    values = np.squeeze(np.asarray(image, dtype=float))
    ny, nx = values.shape
    detected_xy, detection_report = detect_registration_sources(
        values,
        pixel_scale_arcsec=catalog_pixel_scale_arcsec,
        sigma_clip=float(settings["sigma_clip"]),
        threshold_sigma=float(settings["detection_threshold_sigma"]),
        minimum_separation_arcsec=float(settings["minimum_detection_separation_arcsec"]),
        exclusion_radius_arcsec=float(settings["galaxy_exclusion_radius_arcsec"]),
        maximum_sources=int(settings["maximum_detected_sources"]),
    )
    east, north = catalog_center.spherical_offsets_to(gaia_sky)
    conventional_offsets = np.column_stack(
        [
            -east.to_value(u.arcsec) / catalog_pixel_scale_arcsec,
            north.to_value(u.arcsec) / catalog_pixel_scale_arcsec,
        ]
    )
    image_center = np.array([nx / 2.0, ny / 2.0])
    margin = float(settings["maximum_translation_pixel"])
    candidates = []
    for orientation in settings["orientation_candidates"]:
        matrix = ORIENTATION_MATRICES[orientation]
        predicted_all = conventional_offsets @ matrix.T + image_center
        inside = (
            (predicted_all[:, 0] > -margin)
            & (predicted_all[:, 0] < nx + margin)
            & (predicted_all[:, 1] > -margin)
            & (predicted_all[:, 1] < ny + margin)
        )
        predicted = predicted_all[inside]
        sky = gaia_sky[inside]
        translation, histogram_peak = _translation_mode(
            predicted,
            detected_xy,
            maximum_translation_pixel=margin,
            bin_width_pixel=float(settings["translation_histogram_bin_pixel"]),
        )
        source_indices, detection_indices = _deduplicate_nearest(
            predicted + translation,
            detected_xy,
            float(settings["initial_match_radius_pixel"]),
        )
        if len(source_indices) < 3:
            continue
        model, inliers = ransac(
            (predicted[source_indices], detected_xy[detection_indices]),
            SimilarityTransform,
            min_samples=3,
            residual_threshold=float(settings["ransac_residual_pixel"]),
            max_trials=int(settings["ransac_max_trials"]),
            rng=int(settings["random_seed"]),
        )
        if model is None or inliers is None or int(np.count_nonzero(inliers)) < 3:
            continue
        chosen_source = source_indices[inliers]
        chosen_detection = detection_indices[inliers]
        candidates.append(
            {
                "orientation": orientation,
                "predicted": predicted[chosen_source],
                "detected": detected_xy[chosen_detection],
                "sky": sky[chosen_source],
                "inliers": int(np.count_nonzero(inliers)),
                "translation": translation,
                "histogram_peak": histogram_peak,
                "similarity_scale": float(model.scale),
                "similarity_rotation_deg": float(np.degrees(model.rotation)),
                "catalog_sources_in_search": len(predicted),
            }
        )
    if not candidates:
        raise ValueError("no astrometric orientation produced a valid solution")
    candidates.sort(key=lambda row: (-row["inliers"], abs(row["similarity_scale"] - 1.0)))
    best = candidates[0]
    fitted_wcs = fit_wcs_from_points(
        (best["detected"][:, 0], best["detected"][:, 1]),
        best["sky"],
        projection=str(settings["final_projection"]),
    )
    fitted_pixel = np.column_stack(fitted_wcs.world_to_pixel(best["sky"]))
    residual = np.linalg.norm(fitted_pixel - best["detected"], axis=1)
    diagnostics: dict[str, float | int | str] = {
        **detection_report,
        "orientation": str(best["orientation"]),
        "gaia_inliers": int(best["inliers"]),
        "second_best_inliers": int(candidates[1]["inliers"]) if len(candidates) > 1 else 0,
        "catalog_sources_in_search": int(best["catalog_sources_in_search"]),
        "translation_x_pixel": float(best["translation"][0]),
        "translation_y_pixel": float(best["translation"][1]),
        "translation_histogram_peak": int(best["histogram_peak"]),
        "similarity_scale": float(best["similarity_scale"]),
        "similarity_rotation_deg": float(best["similarity_rotation_deg"]),
        "median_residual_pixel": float(np.median(residual)),
        "p95_residual_pixel": float(np.quantile(residual, 0.95)),
        "maximum_residual_pixel": float(np.max(residual)),
    }
    return AstrometricFit(
        wcs=fitted_wcs,
        diagnostics=diagnostics,
        matched_pixel_xy=np.asarray(best["detected"], dtype=float),
        matched_sky=best["sky"],
        residual_pixel=residual,
    )
