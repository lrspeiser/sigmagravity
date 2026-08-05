#!/usr/bin/env python3
"""Run the frozen V19AL shared-PSF foreground-star astrometry audit."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from astropy.io import fits
from scipy.optimize import least_squares


ROOT = Path(__file__).resolve().parents[1]
BASE_SCRIPT = ROOT / "scripts" / "run_sigma_v19ai_fors1_subpixel_astrometry.py"
DEFAULT_CONFIG = ROOT / "configs" / "sigma_v19al_fors1_shared_psf_astrometry.json"
SPEC = importlib.util.spec_from_file_location("sigma_v19ai_frozen_base", BASE_SCRIPT)
BASE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(BASE)

sha256 = BASE.sha256
load_json = BASE.load_json
fit_and_loo = BASE.fit_and_loo
json_wcs = BASE.json_wcs
center_separations = BASE.center_separations

_SHARED_SIGMA_BY_IMAGE: dict[str, float] = {}
_SHARED_PSF_DIAGNOSTICS: dict[str, Any] = {}


def image_fingerprint(image: np.ndarray) -> str:
    values = np.asarray(image, dtype=np.float64)
    flat = values.ravel()
    indices = np.linspace(0, flat.size - 1, 64, dtype=np.int64)
    sample = np.nan_to_num(flat[indices], nan=0.0, posinf=1e300, neginf=-1e300)
    digest = hashlib.sha256()
    digest.update(np.asarray(values.shape, dtype=np.int64).tobytes())
    digest.update(sample.astype("<f8", copy=False).tobytes())
    return digest.hexdigest()


def extract_fit_data(
    image: np.ndarray, initial_x: float, initial_y: float, settings: dict[str, Any]
) -> dict[str, Any]:
    values = np.asarray(image, dtype=np.float64)
    half = int(settings["stamp_half_width_pixel"])
    ix, iy = int(round(initial_x)), int(round(initial_y))
    ny, nx = values.shape
    if ix - half < 0 or iy - half < 0 or ix + half >= nx or iy + half >= ny:
        return {"ok": False, "reason": "edge_truncation"}
    stamp = values[iy - half : iy + half + 1, ix - half : ix + half + 1]
    yy, xx = np.mgrid[iy - half : iy + half + 1, ix - half : ix + half + 1]
    local_x, local_y = xx - float(initial_x), yy - float(initial_y)
    radius = np.hypot(local_x, local_y)
    fit_mask = radius <= float(settings["fit_radius_pixel"])
    annulus = (
        (radius >= float(settings["background_annulus_inner_pixel"]))
        & (radius <= float(settings["background_annulus_outer_pixel"]))
    )
    if not np.all(np.isfinite(stamp[fit_mask])):
        return {"ok": False, "reason": "nonfinite_fit_pixels"}
    annulus_values = stamp[annulus & np.isfinite(stamp)]
    if annulus_values.size < 20:
        return {"ok": False, "reason": "insufficient_background_annulus"}
    background = float(np.median(annulus_values))
    noise = float(1.4826 * np.median(np.abs(annulus_values - background)))
    if not math.isfinite(noise) or noise <= 0:
        noise = float(np.std(annulus_values))
    if not math.isfinite(noise) or noise <= 0:
        noise = 1.0
    data = stamp[fit_mask].astype(float)
    x = local_x[fit_mask].astype(float)
    y = local_y[fit_mask].astype(float)
    amplitude = max(float(np.max(data) - background), noise)
    return {
        "ok": True,
        "data": data,
        "x": x,
        "y": y,
        "background": background,
        "noise": noise,
        "amplitude": amplitude,
    }


def fit_star(
    image: np.ndarray,
    initial_x: float,
    initial_y: float,
    settings: dict[str, Any],
    shared_sigma: float | None,
) -> dict[str, Any]:
    extracted = extract_fit_data(image, initial_x, initial_y, settings)
    base_result: dict[str, Any] = {
        "initial_x_pixel": float(initial_x),
        "initial_y_pixel": float(initial_y),
        "accepted": False,
        "rejection_reason": "",
    }
    if not extracted["ok"]:
        base_result["rejection_reason"] = extracted["reason"]
        return base_result
    x, y, data = extracted["x"], extracted["y"], extracted["data"]
    noise = float(extracted["noise"])
    amplitude = float(extracted["amplitude"])
    background = float(extracted["background"])
    maximum_shift = float(settings["maximum_shift_from_v19ah_peak_pixel"])
    minimum_sigma = float(settings["minimum_preliminary_sigma_pixel"])
    maximum_sigma = float(settings["maximum_preliminary_sigma_pixel"])
    logamp0 = math.log(amplitude)
    logamp_lo = math.log(max(noise * 0.01, np.finfo(float).tiny))
    logamp_hi = math.log(max(amplitude * 100.0, noise * 100.0, 1.0))

    if shared_sigma is None:
        initial = np.array([0.0, 0.0, logamp0, background, 0.0, 0.0, math.log(1.5)])
        lower = np.array([-maximum_shift, -maximum_shift, logamp_lo, -np.inf, -np.inf, -np.inf, math.log(minimum_sigma)])
        upper = np.array([maximum_shift, maximum_shift, logamp_hi, np.inf, np.inf, np.inf, math.log(maximum_sigma)])

        def residual(parameters: np.ndarray) -> np.ndarray:
            dx, dy, logamp, b0, bx, by, logsigma = parameters
            sigma = np.exp(logsigma)
            model = b0 + bx * x + by * y + np.exp(logamp) * np.exp(
                -0.5 * ((x - dx) ** 2 + (y - dy) ** 2) / sigma**2
            )
            return (model - data) / noise

    else:
        initial = np.array([0.0, 0.0, logamp0, background, 0.0, 0.0])
        lower = np.array([-maximum_shift, -maximum_shift, logamp_lo, -np.inf, -np.inf, -np.inf])
        upper = np.array([maximum_shift, maximum_shift, logamp_hi, np.inf, np.inf, np.inf])

        def residual(parameters: np.ndarray) -> np.ndarray:
            dx, dy, logamp, b0, bx, by = parameters
            model = b0 + bx * x + by * y + np.exp(logamp) * np.exp(
                -0.5 * ((x - dx) ** 2 + (y - dy) ** 2) / shared_sigma**2
            )
            return (model - data) / noise

    fit = least_squares(
        residual,
        initial,
        bounds=(lower, upper),
        loss=str(settings["robust_loss"]),
        f_scale=float(settings["robust_loss_scale"]),
        max_nfev=int(settings["maximum_function_evaluations"]),
    )
    dx, dy, logamp, fitted_background, slope_x, slope_y = (
        float(value) for value in fit.x[:6]
    )
    sigma = float(np.exp(fit.x[6])) if shared_sigma is None else float(shared_sigma)
    fitted_amplitude = float(np.exp(logamp))
    shift = float(np.hypot(dx, dy))
    normalized_rmse = float(np.sqrt(np.mean(residual(fit.x) ** 2)))
    amplitude_snr = fitted_amplitude / noise
    result = {
        **base_result,
        "refined_x_pixel": float(initial_x + dx),
        "refined_y_pixel": float(initial_y + dy),
        "centroid_shift_pixel": shift,
        "background_adu": fitted_background,
        "background_slope_x_adu_per_pixel": slope_x,
        "background_slope_y_adu_per_pixel": slope_y,
        "net_weight_adu": fitted_amplitude,
        "fit_amplitude_adu": fitted_amplitude,
        "fit_noise_adu": noise,
        "fit_amplitude_snr": amplitude_snr,
        "fit_normalized_rmse": normalized_rmse,
        "optimizer_success": bool(fit.success),
        "optimizer_nfev": int(fit.nfev),
        "shared_psf_sigma_pixel": sigma,
        "fwhm_pixel": float(2.354820045 * sigma),
        "ellipticity": 0.0,
        "moment_xx": float(sigma**2),
        "moment_yy": float(sigma**2),
        "moment_xy": 0.0,
    }
    bound_margin = float(settings["minimum_centroid_bound_margin_pixel"])
    if not fit.success:
        result["rejection_reason"] = "optimizer_failure"
        return result
    if maximum_shift - max(abs(dx), abs(dy)) < bound_margin:
        result["rejection_reason"] = "centroid_at_optimizer_bound"
        return result
    if shift > maximum_shift:
        result["rejection_reason"] = "centroid_shift"
        return result
    if amplitude_snr < float(settings["minimum_amplitude_snr"]):
        result["rejection_reason"] = "amplitude_snr"
        return result
    if normalized_rmse > float(settings["maximum_normalized_rmse"]):
        result["rejection_reason"] = "normalized_rmse"
        return result
    if not (
        float(settings["minimum_fwhm_pixel"])
        <= result["fwhm_pixel"]
        <= float(settings["maximum_fwhm_pixel"])
    ):
        result["rejection_reason"] = "fwhm"
        return result
    result["accepted"] = True
    return result


def prepare_shared_psf(config: dict[str, Any]) -> tuple[dict[str, float], dict[str, Any]]:
    settings = config["shared_psf"]
    sigmas_by_image: dict[str, float] = {}
    diagnostics: dict[str, Any] = {}
    for product in config["science_products"]:
        with fits.open(ROOT / product["path"], memmap=False) as hdul:
            image = np.asarray(hdul[0].data, dtype=np.float64)
        matches = pd.read_csv(ROOT / product["matches"], dtype={"source_id": str})
        fits_by_star = [
            fit_star(
                image,
                float(row["image_x_pixel"]),
                float(row["image_y_pixel"]),
                settings,
                shared_sigma=None,
            )
            for _, row in matches.iterrows()
        ]
        accepted = [result for result in fits_by_star if result["accepted"]]
        if len(accepted) < int(settings["minimum_preliminary_psf_stars_per_filter"]):
            raise RuntimeError(
                f"{product['filter']} has only {len(accepted)} preliminary PSF stars"
            )
        sigma_values = np.asarray(
            [result["shared_psf_sigma_pixel"] for result in accepted], dtype=float
        )
        shared_sigma = float(np.median(sigma_values))
        fingerprint = image_fingerprint(image)
        if fingerprint in sigmas_by_image:
            raise RuntimeError("nonunique image fingerprint")
        sigmas_by_image[fingerprint] = shared_sigma
        diagnostics[product["filter"]] = {
            "input_stars": int(len(matches)),
            "accepted_preliminary_psf_stars": int(len(accepted)),
            "rejection_reasons": {
                str(key): int(value)
                for key, value in pd.Series(
                    [result["rejection_reason"] for result in fits_by_star if not result["accepted"]],
                    dtype=str,
                )
                .value_counts()
                .items()
            },
            "shared_sigma_pixel": shared_sigma,
            "shared_fwhm_pixel": float(2.354820045 * shared_sigma),
            "sigma_mad_pixel": float(
                1.4826 * np.median(np.abs(sigma_values - np.median(sigma_values)))
            ),
            "image_fingerprint": fingerprint,
        }
    return sigmas_by_image, diagnostics


def refine_centroid(
    image: np.ndarray, initial_x: float, initial_y: float, settings: dict[str, Any]
) -> dict[str, Any]:
    fingerprint = image_fingerprint(image)
    if fingerprint not in _SHARED_SIGMA_BY_IMAGE:
        return {
            "initial_x_pixel": float(initial_x),
            "initial_y_pixel": float(initial_y),
            "accepted": False,
            "rejection_reason": "missing_shared_psf",
        }
    return fit_star(
        image,
        initial_x,
        initial_y,
        settings,
        shared_sigma=_SHARED_SIGMA_BY_IMAGE[fingerprint],
    )


def validate_config(config_path: Path, config: dict[str, Any]) -> dict[str, str]:
    if config["status"] != "frozen_before_any_v19al_shared_psf_fit":
        raise RuntimeError("V19AL protocol is not frozen")
    runner = ROOT / config["implementation"]["runner"]
    if sha256(runner) != config["implementation"]["runner_sha256"]:
        raise RuntimeError("frozen V19AL runner hash mismatch")
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
            raise RuntimeError(f"V19AL parent hash mismatch: {artifact['path']}")
        hashes[artifact["path"]] = actual
    if len(config["science_products"]) != int(config["gates"]["exact_filter_count"]):
        raise RuntimeError("V19AL filter count changed")
    for product in config["science_products"]:
        path = ROOT / product["path"]
        actual = sha256(path)
        if actual != product["sha256"]:
            raise RuntimeError(f"V19AL science hash mismatch: {product['filter']}")
        hashes[product["path"]] = actual
    prohibited = [
        "detect_or_rematch_sources",
        "inspect_member_or_candidate_coordinates_or_cutouts",
        "fit_science_photometry_or_member_deblending",
        "infer_stellar_mass_or_current",
        "read_lensing_or_halo_payload",
        "change_gravity_physics_or_parameters",
        "open_holdout",
    ]
    if any(config["authorization"][name] for name in prohibited):
        raise RuntimeError("V19AL authorizes a prohibited action")
    return hashes


def run(config_path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    global _SHARED_SIGMA_BY_IMAGE, _SHARED_PSF_DIAGNOSTICS
    config_path = config_path.resolve()
    config = load_json(config_path)
    validate_config(config_path, config)
    _SHARED_SIGMA_BY_IMAGE, _SHARED_PSF_DIAGNOSTICS = prepare_shared_psf(config)
    original_validator = BASE.validate_config
    original_refiner = BASE.refine_centroid
    BASE.validate_config = validate_config
    BASE.refine_centroid = refine_centroid
    try:
        report = BASE.run(config_path)
    finally:
        BASE.validate_config = original_validator
        BASE.refine_centroid = original_refiner
    report["shared_psf_calibration"] = _SHARED_PSF_DIAGNOSTICS
    report["gravity_parameters_fit"] = 0
    report_path = ROOT / config["outputs"]["report"]
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    report = run(args.config)
    print(
        json.dumps(
            {
                "status": report["status"],
                "shared_psf_calibration": report["shared_psf_calibration"],
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
