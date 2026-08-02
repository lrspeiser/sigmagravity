#!/usr/bin/env python3
"""Score frozen P0708 LOS fields against untouched LITTLE THINGS moment-1 maps."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.convolution import Gaussian2DKernel, convolve_fft
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.wcs import WCS
from scipy.interpolate import RegularGridInterpolator

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from voidscreen.galaxy_maps import aips_clean_beam_degrees, sky_pixels_to_disk_coordinates

DEFAULT_UNLOCK = ROOT / "results/p0633_external_validation/unlock_manifest.json"
METADATA = ROOT / "results/p0637_little_things_photometric_metadata/photometric_inputs.csv"
AUDIT = ROOT / "results/p0639_registered_baryonic_maps/map_audit.csv"
PREDICTION_DIR = ROOT / "results/p0708_external_prediction_lock/galaxies"
MOMENT_DIR = ROOT / "data/raw/p0633_little_things_kinematics"
BARYON_DIR = ROOT / "data/raw/p0633_little_things_baryons"
OUTPUT = ROOT / "results/p0712_external_galaxy_velocity_field_validation"

MODEL_CANDIDATE = "P0707_time_potential"
MODEL_NEWTONIAN = "Newtonian_3D"
MODEL_AQUAL = "AQUAL_simple_mu_3D"
MODEL_QUMOND = "QUMOND_simple_nu_3D"
MODELS = [MODEL_CANDIDATE, MODEL_NEWTONIAN, MODEL_AQUAL, MODEL_QUMOND]
SYSTEMIC_KM_S = {
    "CVnIdwA": 307.9,
    "DDO47": 272.8,
    "DDO50": 156.7,
    "DDO52": 396.2,
    "DDO53": 20.4,
    "DDO87": 338.7,
    "DDO101": 586.6,
    "DDO126": 214.3,
    "DDO133": 331.3,
    "DDO210": -140.0,
    "DDO216": -188.0,
    "NGC1569": -75.6,
    "UGC8508": 59.9,
}
CHANNEL_WIDTH_KM_S = {
    "CVnIdwA": 1.3,
    "DDO47": 2.6,
    "DDO50": 2.6,
    "DDO52": 2.6,
    "DDO53": 2.6,
    "DDO87": 2.6,
    "DDO101": 2.6,
    "DDO126": 2.6,
    "DDO133": 2.6,
    "DDO210": 1.3,
    "DDO216": 1.3,
    "NGC1569": 2.6,
    "UGC8508": 1.3,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def image2(path: Path) -> tuple[np.ndarray, fits.Header]:
    return np.squeeze(fits.getdata(path)).astype(float), fits.getheader(path)


def weighted_rmse(residual: np.ndarray, weight: np.ndarray) -> float:
    return float(np.sqrt(np.sum(weight * np.square(residual)) / np.sum(weight)))


def velocity_unit_scale(header: fits.Header) -> float:
    unit = str(header.get("BUNIT", "")).upper().replace(" ", "")
    if unit in {"METR/SEC", "M/S", "M.S-1"}:
        return 1.0 / 1000.0
    if "KM" in unit:
        return 1.0
    raise ValueError(f"unrecognized velocity unit: {header.get('BUNIT')}")


def model_sky_field(
    disk_axis: np.ndarray,
    disk_field: np.ndarray,
    major: np.ndarray,
    minor: np.ndarray,
) -> np.ndarray:
    interpolator = RegularGridInterpolator(
        (disk_axis, disk_axis),
        disk_field,
        bounds_error=False,
        fill_value=np.nan,
    )
    return interpolator(np.column_stack([major.ravel(), minor.ravel()])).reshape(
        major.shape
    )


def beam_convolve_velocity(
    velocity: np.ndarray,
    intensity: np.ndarray,
    header: fits.Header,
) -> tuple[np.ndarray, dict[str, float]]:
    bmaj_deg, bmin_deg, bpa_deg = aips_clean_beam_degrees(header)
    pixel_x_deg = abs(float(header["CDELT1"]))
    pixel_y_deg = abs(float(header["CDELT2"]))
    major_sigma_x_pixels = bmaj_deg / pixel_x_deg / 2.354820045
    major_sigma_y_pixels = bmaj_deg / pixel_y_deg / 2.354820045
    minor_sigma_x_pixels = bmin_deg / pixel_x_deg / 2.354820045
    minor_sigma_y_pixels = bmin_deg / pixel_y_deg / 2.354820045
    major_sigma_pixels = math.sqrt(major_sigma_x_pixels * major_sigma_y_pixels)
    minor_sigma_pixels = math.sqrt(minor_sigma_x_pixels * minor_sigma_y_pixels)
    theta = np.deg2rad(90.0 + bpa_deg)
    size = int(max(9, 2 * math.ceil(4.0 * major_sigma_pixels) + 1))
    if size % 2 == 0:
        size += 1
    kernel = Gaussian2DKernel(
        major_sigma_pixels,
        minor_sigma_pixels,
        theta=theta,
        x_size=size,
        y_size=size,
    ).array
    support = np.isfinite(velocity) & np.isfinite(intensity) & (intensity > 0.0)
    numerator = convolve_fft(
        np.where(support, velocity * intensity, 0.0),
        kernel,
        boundary="fill",
        fill_value=0.0,
        normalize_kernel=True,
        allow_huge=True,
    )
    denominator = convolve_fft(
        np.where(support, intensity, 0.0),
        kernel,
        boundary="fill",
        fill_value=0.0,
        normalize_kernel=True,
        allow_huge=True,
    )
    result = np.divide(
        numerator,
        denominator,
        out=np.full_like(numerator, np.nan),
        where=denominator > np.finfo(float).tiny,
    )
    return result, {
        "beam_major_arcsec": bmaj_deg * 3600.0,
        "beam_minor_arcsec": bmin_deg * 3600.0,
        "beam_pa_deg": bpa_deg,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--unlock", type=Path, default=DEFAULT_UNLOCK)
    args = parser.parse_args()
    unlock = json.loads(args.unlock.resolve().read_text(encoding="utf-8"))
    if unlock["status"] != "authorized_for_exactly_one_external_parse":
        raise RuntimeError("P0709 unlock is missing")
    metadata = pd.read_csv(METADATA).set_index("galaxy")
    audit = pd.read_csv(AUDIT).set_index("galaxy")
    products = pd.DataFrame(unlock["galaxy_moment_products"])
    target_rows = []
    residual_maps = {}
    for galaxy in sorted(SYSTEMIC_KM_S):
        print(f"P0712 {galaxy}", flush=True)
        system_products = products[products["system"] == galaxy].set_index("product")
        moment1_path = MOMENT_DIR / galaxy / system_products.loc["XMOM1", "filename"]
        moment2_path = MOMENT_DIR / galaxy / system_products.loc["XMOM2", "filename"]
        baryon_target = next(
            item
            for item in json.loads(
                (ROOT / "configs/p0636_little_things_baryon_acquisition.json").read_text(
                    encoding="utf-8"
                )
            )["targets"]
            if item["id"] == galaxy
        )
        moment0_path = BARYON_DIR / galaxy / baryon_target["hi_filename"]
        moment1, header1 = image2(moment1_path)
        moment2, header2 = image2(moment2_path)
        moment0, _header0 = image2(moment0_path)
        if moment1.shape != moment2.shape or moment1.shape != moment0.shape:
            raise RuntimeError(f"moment-map shape mismatch for {galaxy}")
        moment1 *= velocity_unit_scale(header1)
        moment2 *= velocity_unit_scale(header2)
        valid_velocity = np.isfinite(moment1)
        yy_all, xx_all = np.nonzero(valid_velocity)
        if len(xx_all) < 100:
            raise RuntimeError(f"too few velocity pixels for {galaxy}")
        bmaj, _, _ = aips_clean_beam_degrees(header1)
        margin = math.ceil(4.0 * bmaj / abs(float(header1["CDELT1"])) / 2.35482)
        y0, y1 = max(0, int(yy_all.min()) - margin), min(moment1.shape[0], int(yy_all.max()) + margin + 1)
        x0, x1 = max(0, int(xx_all.min()) - margin), min(moment1.shape[1], int(xx_all.max()) + margin + 1)
        velocity = moment1[y0:y1, x0:x1]
        dispersion = moment2[y0:y1, x0:x1]
        intensity = np.clip(moment0[y0:y1, x0:x1], 0.0, None)
        yy, xx = np.indices(velocity.shape, dtype=float)
        xx += x0
        yy += y0
        meta = metadata.loc[galaxy]
        center = SkyCoord(
            str(meta["photometric_center_ra_j2000"]),
            str(meta["photometric_center_dec_j2000"]),
            unit=(u.hourangle, u.deg),
        )
        major, minor = sky_pixels_to_disk_coordinates(
            xx,
            yy,
            WCS(header1).celestial,
            center=center,
            position_angle_deg=float(meta["photometric_pa_deg"]),
            inclination_deg=float(meta["derived_photometric_inclination_deg"]),
            distance_mpc=float(meta["distance_mpc"]),
        )
        radius = np.hypot(major, minor)
        observed = velocity - SYSTEMIC_KM_S[galaxy]
        initial_mask = (
            np.isfinite(observed)
            & np.isfinite(dispersion)
            & (dispersion >= 0.0)
            & np.isfinite(intensity)
            & (intensity > 0.0)
            & (radius <= float(audit.loc[galaxy, "hi_r995_kpc"]))
        )
        channel = CHANNEL_WIDTH_KM_S[galaxy]
        initial_weight = np.where(
            initial_mask,
            intensity / np.maximum(np.square(dispersion) + (channel / 2.355) ** 2, 1e-12),
            0.0,
        )
        direction = np.divide(major, radius, out=np.zeros_like(major), where=radius > 0.0)
        signed_covariance = float(np.sum(initial_weight * observed * direction))
        handedness = 1.0 if signed_covariance >= 0.0 else -1.0
        with np.load(PREDICTION_DIR / f"{galaxy}_predictions.npz") as prediction:
            disk_axis = prediction["axis_kpc"].astype(float)
            locked_fields = {
                model: prediction[f"los_{model}"].astype(float) for model in MODELS
            }
        convolved = {}
        beam_diagnostics = None
        for model in MODELS:
            sky_field = handedness * model_sky_field(
                disk_axis, locked_fields[model], major, minor
            )
            convolved[model], beam_diagnostics = beam_convolve_velocity(
                sky_field, intensity, header1
            )
        common = initial_mask.copy()
        for model in MODELS:
            common &= np.isfinite(convolved[model])
        if common.sum() < 100:
            raise RuntimeError(f"too few common velocity pixels for {galaxy}")
        weight = np.where(common, initial_weight, 0.0)
        weight /= np.sum(weight)
        row = {
            "galaxy": galaxy,
            "valid_pixels": int(common.sum()),
            "systemic_velocity_km_s": SYSTEMIC_KM_S[galaxy],
            "channel_width_km_s": channel,
            "receding_side_handedness": int(handedness),
            "handedness_rule": "sign of weighted observed major-axis covariance, shared by every model",
            **beam_diagnostics,
            "moment1_sha256": sha256(moment1_path),
            "moment2_sha256": sha256(moment2_path),
            "moment0_sha256": sha256(moment0_path),
        }
        for model in MODELS:
            residual = convolved[model] - observed
            row[f"weighted_RMSE_{model}"] = weighted_rmse(residual[common], weight[common])
        target_rows.append(row)
        residual_maps[galaxy] = {
            "observed": np.where(common, observed, np.nan),
            "candidate": np.where(common, convolved[MODEL_CANDIDATE] - observed, np.nan),
            "aqual": np.where(common, convolved[MODEL_AQUAL] - observed, np.nan),
            "qumond": np.where(common, convolved[MODEL_QUMOND] - observed, np.nan),
        }

    per_galaxy = pd.DataFrame(target_rows).sort_values("galaxy").reset_index(drop=True)
    sample = {
        model: float(
            np.sqrt(np.mean(np.square(per_galaxy[f"weighted_RMSE_{model}"].to_numpy())))
        )
        for model in MODELS
    }
    best_mond = min([MODEL_AQUAL, MODEL_QUMOND], key=sample.get)
    ratio = sample[MODEL_CANDIDATE] / sample[best_mond]
    gates = unlock["rejection_thresholds"]["galaxy"]
    gate_results = {
        "minimum_valid_galaxies": len(per_galaxy) >= gates["minimum_valid_galaxies"],
        "velocity_field_RMSE": ratio <= gates["velocity_field_RMSE_ratio_to_best_frozen_MOND_max"],
        "no_target_refit": bool(gates["no_target_refit"]),
    }

    OUTPUT.mkdir(parents=True, exist_ok=True)
    per_galaxy.to_csv(OUTPUT / "per_galaxy_velocity_field_scores.csv", index=False)
    figure, axes = plt.subplots(len(per_galaxy), 4, figsize=(14, 3.0 * len(per_galaxy)))
    for row_index, galaxy in enumerate(per_galaxy["galaxy"]):
        maps = residual_maps[galaxy]
        observed_limit = float(np.nanpercentile(np.abs(maps["observed"]), 98))
        residual_limit = max(
            float(np.nanpercentile(np.abs(maps[key]), 98))
            for key in ["candidate", "aqual", "qumond"]
        )
        for column, key in enumerate(["observed", "candidate", "aqual", "qumond"]):
            axis = axes[row_index, column]
            if key == "observed":
                axis.imshow(
                    maps[key], origin="lower", cmap="RdBu_r", vmin=-observed_limit, vmax=observed_limit
                )
            else:
                axis.imshow(
                    maps[key], origin="lower", cmap="RdBu_r", vmin=-residual_limit, vmax=residual_limit
                )
            axis.set_xticks([])
            axis.set_yticks([])
            if row_index == 0:
                axis.set_title(
                    {"observed": "observed - systemic", "candidate": "candidate residual", "aqual": "AQUAL residual", "qumond": "QUMOND residual"}[key]
                )
            if column == 0:
                axis.set_ylabel(galaxy)
    figure.suptitle("P0712 untouched resolved velocity-field validation")
    figure.tight_layout(rect=(0, 0, 1, 0.99))
    figure.savefig(OUTPUT / "velocity_field_residual_atlas.png", dpi=150)
    plt.close(figure)

    report = {
        "report_version": "P0712-EXTERNAL-GALAXY-VELOCITY-FIELD-VALIDATION-1.0.0",
        "status": "pass" if all(gate_results.values()) else "fail",
        "P0633_sample_spent": True,
        "universal_parameter_sha256": unlock["universal_parameter_sha256"],
        "valid_galaxies": len(per_galaxy),
        "sample_weighted_RMSE_km_s": sample,
        "best_frozen_MOND_model": best_mond,
        "candidate_to_best_MOND_RMSE_ratio": ratio,
        "gate_results": gate_results,
        "failed_gates": [name for name, passed in gate_results.items() if not passed],
        "photometric_distance_inclination_PA_refits": 0,
        "per_object_gravity_parameters": 0,
        "ordinary_observational_coordinates": {
            "published_systemic_velocity": True,
            "measured_receding_side_handedness": True,
            "handedness_selected_by_model_RMSE": False,
        },
        "beam_model": "measured elliptical AIPS CLEAN beam, intensity-weighted convolution",
    }
    (OUTPUT / "report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    summary = f"""# P0712 untouched resolved velocity-field validation

- Status: **{report['status'].upper()}**.
- Valid galaxies: **{len(per_galaxy)} / 13**.
- Candidate equal-galaxy weighted RMSE: **{sample[MODEL_CANDIDATE]:.3f} km/s**.
- Best frozen full-field MOND: **{best_mond}**, **{sample[best_mond]:.3f} km/s**.
- Candidate / best-MOND ratio: **{ratio:.4f}** (gate: <= {gates['velocity_field_RMSE_ratio_to_best_frozen_MOND_max']:.2f}).
- Newtonian weighted RMSE: **{sample[MODEL_NEWTONIAN]:.3f} km/s**.
- Photometric distance, inclination, and PA refits / gravity parameters: **0 / 0**.
"""
    (OUTPUT / "SUMMARY.md").write_text(summary, encoding="utf-8")
    print(summary)


if __name__ == "__main__":
    main()
