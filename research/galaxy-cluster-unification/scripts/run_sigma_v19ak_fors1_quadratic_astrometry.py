#!/usr/bin/env python3
"""Run the frozen V19AK local-quadratic foreground-star astrometry audit."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
BASE_SCRIPT = ROOT / "scripts" / "run_sigma_v19ai_fors1_subpixel_astrometry.py"
DEFAULT_CONFIG = ROOT / "configs" / "sigma_v19ak_fors1_quadratic_astrometry.json"
SPEC = importlib.util.spec_from_file_location("sigma_v19ai_frozen_base", BASE_SCRIPT)
BASE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(BASE)

sha256 = BASE.sha256
load_json = BASE.load_json
fit_and_loo = BASE.fit_and_loo
json_wcs = BASE.json_wcs
center_separations = BASE.center_separations


def refine_centroid(
    image: np.ndarray, initial_x: float, initial_y: float, settings: dict[str, Any]
) -> dict[str, Any]:
    values = np.asarray(image, dtype=np.float64)
    half = int(settings["stamp_half_width_pixel"])
    peak_half = int(settings["peak_stencil_half_width_pixel"])
    ny, nx = values.shape
    ix, iy = int(round(initial_x)), int(round(initial_y))
    result: dict[str, Any] = {
        "initial_x_pixel": float(initial_x),
        "initial_y_pixel": float(initial_y),
        "accepted": False,
        "rejection_reason": None,
    }
    if (
        ix - half < 0
        or iy - half < 0
        or ix + half >= nx
        or iy + half >= ny
        or peak_half != 1
    ):
        result["rejection_reason"] = "edge_truncation"
        return result

    peak = values[iy - peak_half : iy + peak_half + 1, ix - peak_half : ix + peak_half + 1]
    if peak.shape != (3, 3) or not np.all(np.isfinite(peak)):
        result["rejection_reason"] = "nonfinite_peak_stencil"
        return result
    local_y, local_x = np.mgrid[-1:2, -1:2]
    design = np.column_stack(
        [
            local_x.ravel() ** 2,
            (local_x * local_y).ravel(),
            local_y.ravel() ** 2,
            local_x.ravel(),
            local_y.ravel(),
            np.ones(9),
        ]
    )
    coefficients, *_ = np.linalg.lstsq(design, peak.ravel(), rcond=None)
    a, b, c, d, e, _ = (float(value) for value in coefficients)
    hessian = np.array([[2.0 * a, b], [b, 2.0 * c]], dtype=float)
    eigenvalues = np.linalg.eigvalsh(hessian)
    if not np.all(np.isfinite(eigenvalues)) or np.any(eigenvalues >= 0):
        result["rejection_reason"] = "nonconcave_peak"
        return result
    offset = np.linalg.solve(hessian, -np.array([d, e], dtype=float))
    offset_x, offset_y = float(offset[0]), float(offset[1])
    refined_x, refined_y = ix + offset_x, iy + offset_y
    shift = float(np.hypot(refined_x - initial_x, refined_y - initial_y))
    result.update(
        {
            "refined_x_pixel": refined_x,
            "refined_y_pixel": refined_y,
            "centroid_shift_pixel": shift,
            "quadratic_offset_x_pixel": offset_x,
            "quadratic_offset_y_pixel": offset_y,
            "quadratic_hessian_eigenvalue_min": float(eigenvalues[0]),
            "quadratic_hessian_eigenvalue_max": float(eigenvalues[1]),
        }
    )
    if shift > float(settings["maximum_shift_from_v19ah_peak_pixel"]):
        result["rejection_reason"] = "centroid_shift"
        return result

    x0, x1 = ix - half, ix + half + 1
    y0, y1 = iy - half, iy + half + 1
    stamp = values[y0:y1, x0:x1]
    yy, xx = np.mgrid[y0:y1, x0:x1]
    rr = np.hypot(xx - refined_x, yy - refined_y)
    aperture = rr <= float(settings["aperture_radius_pixel"])
    annulus = (
        (rr >= float(settings["background_annulus_inner_pixel"]))
        & (rr <= float(settings["background_annulus_outer_pixel"]))
    )
    if float(np.mean(np.isfinite(stamp[aperture]))) < float(
        settings["required_finite_aperture_fraction"]
    ):
        result["rejection_reason"] = "nonfinite_aperture"
        return result
    annulus_values = stamp[annulus & np.isfinite(stamp)]
    if annulus_values.size < 20:
        result["rejection_reason"] = "insufficient_background_annulus"
        return result
    background = float(np.median(annulus_values))
    weight = np.where(aperture, np.maximum(stamp - background, 0.0), 0.0)
    total = float(np.sum(weight))
    result.update({"background_adu": background, "net_weight_adu": total})
    if not math.isfinite(total) or total <= 0:
        result["rejection_reason"] = "nonpositive_net_weight"
        return result
    dx, dy = xx - refined_x, yy - refined_y
    mxx = float(np.sum(weight * dx * dx) / total)
    myy = float(np.sum(weight * dy * dy) / total)
    mxy = float(np.sum(weight * dx * dy) / total)
    covariance = np.array([[mxx, mxy], [mxy, myy]], dtype=float)
    moment_eigenvalues = np.linalg.eigvalsh(covariance)
    if np.any(moment_eigenvalues <= 0) or not np.all(np.isfinite(moment_eigenvalues)):
        result["rejection_reason"] = "invalid_second_moments"
        return result
    sigma_major = float(np.sqrt(moment_eigenvalues[1]))
    sigma_minor = float(np.sqrt(moment_eigenvalues[0]))
    fwhm = float(2.354820045 * math.sqrt((mxx + myy) / 2.0))
    ellipticity = float(1.0 - sigma_minor / sigma_major)
    result.update(
        {
            "fwhm_pixel": fwhm,
            "ellipticity": ellipticity,
            "moment_xx": mxx,
            "moment_yy": myy,
            "moment_xy": mxy,
        }
    )
    if not (
        float(settings["minimum_fwhm_pixel"])
        <= fwhm
        <= float(settings["maximum_fwhm_pixel"])
    ):
        result["rejection_reason"] = "fwhm"
        return result
    if ellipticity > float(settings["maximum_ellipticity"]):
        result["rejection_reason"] = "ellipticity"
        return result
    result["accepted"] = True
    result["rejection_reason"] = ""
    return result


def validate_config(config_path: Path, config: dict[str, Any]) -> dict[str, str]:
    if config["status"] != "frozen_before_any_v19ak_foreground_star_quadratic_offset":
        raise RuntimeError("V19AK protocol is not frozen")
    runner = ROOT / config["implementation"]["runner"]
    if sha256(runner) != config["implementation"]["runner_sha256"]:
        raise RuntimeError("frozen V19AK runner hash mismatch")
    if sha256(BASE_SCRIPT) != config["implementation"]["frozen_base_runner_sha256"]:
        raise RuntimeError("frozen V19AI base runner hash mismatch")
    hashes = {
        "config": sha256(config_path),
        "runner": sha256(runner),
        "frozen_base_runner": sha256(BASE_SCRIPT),
    }
    for artifact in config["parent_artifacts"]:
        path = ROOT / artifact["path"]
        actual = sha256(path)
        if actual != artifact["sha256"]:
            raise RuntimeError(f"V19AK parent hash mismatch: {artifact['path']}")
        hashes[artifact["path"]] = actual
    if len(config["science_products"]) != int(config["gates"]["exact_filter_count"]):
        raise RuntimeError("V19AK filter count changed")
    for product in config["science_products"]:
        path = ROOT / product["path"]
        actual = sha256(path)
        if actual != product["sha256"]:
            raise RuntimeError(f"V19AK science hash mismatch: {product['filter']}")
        hashes[product["path"]] = actual
    prohibited = [
        "detect_or_rematch_sources",
        "inspect_member_or_candidate_coordinates_or_cutouts",
        "fit_photometry_or_deblending",
        "infer_stellar_mass_or_current",
        "read_lensing_or_halo_payload",
        "change_gravity_physics_or_parameters",
        "open_holdout",
    ]
    if any(config["authorization"][name] for name in prohibited):
        raise RuntimeError("V19AK authorizes a prohibited action")
    return hashes


def run(config_path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    original_validator = BASE.validate_config
    original_refiner = BASE.refine_centroid
    BASE.validate_config = validate_config
    BASE.refine_centroid = refine_centroid
    try:
        return BASE.run(config_path)
    finally:
        BASE.validate_config = original_validator
        BASE.refine_centroid = original_refiner


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    report = run(args.config)
    print(
        json.dumps(
            {
                "status": report["status"],
                "filters": report["filters"],
                "global_gates": report["global_gates"],
                "failures": report["failures"],
            },
            indent=2,
        )
    )
    return 0 if report["all_subpixel_astrometry_gates_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
