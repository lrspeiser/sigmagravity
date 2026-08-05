#!/usr/bin/env python3
"""Run the frozen V19AI subpixel foreground-star astrometry audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import astropy.units as u
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.wcs import WCS
from astropy.wcs.utils import fit_wcs_from_points


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "sigma_v19ai_fors1_subpixel_astrometry.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def refine_centroid(
    image: np.ndarray, initial_x: float, initial_y: float, settings: dict[str, Any]
) -> dict[str, Any]:
    values = np.asarray(image, dtype=np.float64)
    half = int(settings["stamp_half_width_pixel"])
    ny, nx = values.shape
    center_x = float(initial_x)
    center_y = float(initial_y)
    initial_center_x = center_x
    initial_center_y = center_y
    result: dict[str, Any] = {
        "initial_x_pixel": initial_center_x,
        "initial_y_pixel": initial_center_y,
        "accepted": False,
        "rejection_reason": None,
    }
    last_weight = None
    last_xx = None
    last_yy = None
    last_background = math.nan

    for _ in range(int(settings["iterations"])):
        x0 = int(math.floor(center_x)) - half
        x1 = int(math.floor(center_x)) + half + 1
        y0 = int(math.floor(center_y)) - half
        y1 = int(math.floor(center_y)) + half + 1
        if x0 < 0 or y0 < 0 or x1 > nx or y1 > ny:
            result["rejection_reason"] = "edge_truncation"
            return result
        stamp = values[y0:y1, x0:x1]
        yy, xx = np.mgrid[y0:y1, x0:x1]
        rr = np.hypot(xx - center_x, yy - center_y)
        aperture = rr <= float(settings["aperture_radius_pixel"])
        annulus = (
            (rr >= float(settings["background_annulus_inner_pixel"]))
            & (rr <= float(settings["background_annulus_outer_pixel"]))
        )
        finite_aperture = np.isfinite(stamp[aperture])
        if float(np.mean(finite_aperture)) < float(settings["required_finite_aperture_fraction"]):
            result["rejection_reason"] = "nonfinite_aperture"
            return result
        annulus_values = stamp[annulus & np.isfinite(stamp)]
        if annulus_values.size < 20:
            result["rejection_reason"] = "insufficient_background_annulus"
            return result
        background = float(np.median(annulus_values))
        weight = np.where(aperture, np.maximum(stamp - background, 0.0), 0.0)
        total = float(np.sum(weight))
        if not math.isfinite(total) or total <= 0:
            result["rejection_reason"] = "nonpositive_net_weight"
            return result
        center_x = float(np.sum(weight * xx) / total)
        center_y = float(np.sum(weight * yy) / total)
        last_weight, last_xx, last_yy, last_background = weight, xx, yy, background

    shift = float(np.hypot(center_x - initial_center_x, center_y - initial_center_y))
    result.update(
        {
            "refined_x_pixel": center_x,
            "refined_y_pixel": center_y,
            "centroid_shift_pixel": shift,
            "background_adu": last_background,
            "net_weight_adu": float(np.sum(last_weight)),
        }
    )
    if shift > float(settings["maximum_shift_from_v19ah_peak_pixel"]):
        result["rejection_reason"] = "centroid_shift"
        return result
    dx = last_xx - center_x
    dy = last_yy - center_y
    total = float(np.sum(last_weight))
    mxx = float(np.sum(last_weight * dx * dx) / total)
    myy = float(np.sum(last_weight * dy * dy) / total)
    mxy = float(np.sum(last_weight * dx * dy) / total)
    covariance = np.array([[mxx, mxy], [mxy, myy]], dtype=float)
    eigenvalues = np.linalg.eigvalsh(covariance)
    if np.any(eigenvalues <= 0) or not np.all(np.isfinite(eigenvalues)):
        result["rejection_reason"] = "invalid_second_moments"
        return result
    sigma_major = float(np.sqrt(eigenvalues[1]))
    sigma_minor = float(np.sqrt(eigenvalues[0]))
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


def fit_and_loo(
    xy: np.ndarray, sky: SkyCoord, projection: str
) -> tuple[WCS, np.ndarray, np.ndarray]:
    fitted = fit_wcs_from_points((xy[:, 0], xy[:, 1]), sky, projection=projection)
    fitted_xy = np.column_stack(fitted.world_to_pixel(sky))
    fitted_residual = np.linalg.norm(fitted_xy - xy, axis=1)
    loo_residual = np.empty(len(xy), dtype=float)
    for index in range(len(xy)):
        keep = np.arange(len(xy)) != index
        loo = fit_wcs_from_points(
            (xy[keep, 0], xy[keep, 1]), sky[keep], projection=projection
        )
        predicted = np.asarray(loo.world_to_pixel(sky[index]), dtype=float)
        loo_residual[index] = float(np.linalg.norm(predicted - xy[index]))
    return fitted, fitted_residual, loo_residual


def json_wcs(wcs: WCS) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for card in wcs.to_header(relax=True).cards:
        if card.keyword:
            value = card.value.item() if isinstance(card.value, np.generic) else card.value
            result[card.keyword] = value
    return result


def validate_config(config_path: Path, config: dict[str, Any]) -> dict[str, str]:
    if config["status"] != "frozen_before_any_v19ai_foreground_star_cutout_or_centroid":
        raise RuntimeError("V19AI protocol is not frozen")
    runner = ROOT / config["implementation"]["runner"]
    if sha256(runner) != config["implementation"]["runner_sha256"]:
        raise RuntimeError("frozen V19AI runner hash mismatch")
    hashes = {"config": sha256(config_path), "runner": sha256(runner)}
    for artifact in config["parent_artifacts"]:
        path = ROOT / artifact["path"]
        actual = sha256(path)
        if actual != artifact["sha256"]:
            raise RuntimeError(f"V19AI parent hash mismatch: {artifact['path']}")
        hashes[artifact["path"]] = actual
    if len(config["science_products"]) != int(config["gates"]["exact_filter_count"]):
        raise RuntimeError("V19AI filter count changed")
    for product in config["science_products"]:
        path = ROOT / product["path"]
        actual = sha256(path)
        if actual != product["sha256"]:
            raise RuntimeError(f"V19AI science hash mismatch: {product['filter']}")
        hashes[product["path"]] = actual
    authorization = config["authorization"]
    prohibited = [
        "detect_or_rematch_sources",
        "inspect_member_or_candidate_coordinates_or_cutouts",
        "fit_photometry_or_deblending",
        "infer_stellar_mass_or_current",
        "read_lensing_or_halo_payload",
        "change_gravity_physics_or_parameters",
        "open_holdout",
    ]
    if any(authorization[name] for name in prohibited):
        raise RuntimeError("V19AI authorizes a prohibited action")
    return hashes


def center_separations(solutions: dict[str, WCS], shape: tuple[int, int]) -> dict[str, float]:
    y, x = (shape[0] - 1) / 2.0, (shape[1] - 1) / 2.0
    names = sorted(solutions)
    centers = {name: solutions[name].pixel_to_world(x, y) for name in names}
    return {
        f"{first}__{second}": float(
            centers[first].separation(centers[second]).to_value(u.arcsec)
        )
        for index, first in enumerate(names)
        for second in names[index + 1 :]
    }


def run(config_path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    config_path = config_path.resolve()
    config = load_json(config_path)
    input_hashes = validate_config(config_path, config)
    output = ROOT / config["outputs"]["directory"]
    output.mkdir(parents=True, exist_ok=True)
    gates_config = config["gates"]
    filter_results: dict[str, Any] = {}
    solutions: dict[str, WCS] = {}
    accepted_ids: dict[str, set[str]] = {}
    image_shape: tuple[int, int] | None = None
    failures: list[dict[str, str]] = []

    for product in config["science_products"]:
        filter_name = product["filter"]
        try:
            with fits.open(ROOT / product["path"], memmap=False) as hdul:
                image = np.asarray(hdul[0].data, dtype=np.float64)
                header = hdul[0].header.copy()
            image_shape = image.shape if image_shape is None else image_shape
            if image.shape != image_shape:
                raise RuntimeError("science shapes differ")
            matches = pd.read_csv(ROOT / product["matches"], dtype={"source_id": str})
            rows: list[dict[str, Any]] = []
            for _, match in matches.iterrows():
                centroid = refine_centroid(
                    image,
                    float(match["image_x_pixel"]),
                    float(match["image_y_pixel"]),
                    config["centroid"],
                )
                rows.append({**match.to_dict(), **centroid})
            refined = pd.DataFrame(rows)
            accepted = refined[refined["accepted"].astype(bool)].copy().reset_index(drop=True)
            if len(accepted) < int(gates_config["minimum_accepted_centroids_per_filter"]):
                raise RuntimeError(f"only {len(accepted)} accepted centroids")
            xy = accepted[["refined_x_pixel", "refined_y_pixel"]].to_numpy(dtype=float)
            sky = SkyCoord(
                accepted["ra_epoch"].to_numpy(dtype=float) * u.deg,
                accepted["dec_epoch"].to_numpy(dtype=float) * u.deg,
                frame="icrs",
            )
            wcs, fitted_residual, loo_residual = fit_and_loo(
                xy, sky, str(config["wcs"]["projection"])
            )
            accepted["fitted_residual_pixel"] = fitted_residual
            accepted["loo_residual_pixel"] = loo_residual
            pixel_scale = 3600.0 * float(
                np.mean([abs(float(header["CDELT1"])), abs(float(header["CDELT2"]))])
            )
            accepted["loo_residual_arcsec"] = loo_residual * pixel_scale
            metrics = {
                "input_v19ah_matches": int(len(matches)),
                "accepted_centroids": int(len(accepted)),
                "rejected_centroids": int(len(matches) - len(accepted)),
                "rejection_reasons": {
                    str(key): int(value)
                    for key, value in refined.loc[~refined["accepted"].astype(bool), "rejection_reason"]
                    .value_counts()
                    .items()
                },
                "median_centroid_shift_pixel": float(np.median(accepted["centroid_shift_pixel"])),
                "p95_centroid_shift_pixel": float(np.quantile(accepted["centroid_shift_pixel"], 0.95)),
                "median_fwhm_pixel": float(np.median(accepted["fwhm_pixel"])),
                "median_ellipticity": float(np.median(accepted["ellipticity"])),
                "fitted_median_residual_pixel": float(np.median(fitted_residual)),
                "fitted_p95_residual_pixel": float(np.quantile(fitted_residual, 0.95)),
                "loo_median_residual_pixel": float(np.median(loo_residual)),
                "loo_p95_residual_pixel": float(np.quantile(loo_residual, 0.95)),
                "loo_p95_residual_arcsec": float(np.quantile(loo_residual, 0.95) * pixel_scale),
                "pixel_scale_arcsec": pixel_scale,
            }
            filter_gates = {
                "minimum_accepted_centroids": len(accepted)
                >= int(gates_config["minimum_accepted_centroids_per_filter"]),
                "fitted_median_residual": metrics["fitted_median_residual_pixel"]
                <= float(gates_config["maximum_fitted_median_residual_pixel"]),
                "fitted_p95_residual": metrics["fitted_p95_residual_pixel"]
                <= float(gates_config["maximum_fitted_p95_residual_pixel"]),
                "loo_median_residual": metrics["loo_median_residual_pixel"]
                <= float(gates_config["maximum_loo_median_residual_pixel"]),
                "loo_p95_residual": metrics["loo_p95_residual_pixel"]
                <= float(gates_config["maximum_loo_p95_residual_pixel"]),
                "loo_p95_residual_arcsec": metrics["loo_p95_residual_arcsec"]
                <= float(gates_config["maximum_loo_p95_residual_arcsec"]),
                "r_improves_v19ah": filter_name != "R_BESS"
                or metrics["fitted_median_residual_pixel"]
                < float(gates_config["r_fitted_median_must_improve_over_v19ah"]),
            }
            accepted_ids[filter_name] = set(accepted["source_id"].astype(str))
            solutions[filter_name] = wcs
            refined_path = output / f"refined_matches_{filter_name}.csv"
            accepted_columns = {
                row["source_id"]: row for row in accepted.to_dict(orient="records")
            }
            output_rows = []
            for row in refined.to_dict(orient="records"):
                if str(row["source_id"]) in accepted_columns:
                    row.update(accepted_columns[str(row["source_id"])])
                output_rows.append(row)
            pd.DataFrame(output_rows).to_csv(refined_path, index=False)
            wcs_path = output / f"wcs_{filter_name}.json"
            wcs_path.write_text(
                json.dumps(
                    {
                        "filter": filter_name,
                        "wcs_header": json_wcs(wcs),
                        "metrics": metrics,
                        "gates": filter_gates,
                        "refined_matches": str(refined_path.relative_to(ROOT)).replace("\\", "/"),
                        "refined_matches_sha256": sha256(refined_path),
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            filter_results[filter_name] = {
                "metrics": metrics,
                "gates": filter_gates,
                "wcs_path": wcs_path,
                "matches_path": refined_path,
                "fitted_residual": fitted_residual,
                "loo_residual": loo_residual,
            }
        except Exception as exc:
            failures.append({"filter": filter_name, "error": f"{type(exc).__name__}: {exc}"})

    shared_ids: set[str] = set()
    center_delta: dict[str, float] = {}
    if len(solutions) == int(gates_config["exact_filter_count"]):
        shared_ids = set.intersection(*accepted_ids.values())
        assert image_shape is not None
        center_delta = center_separations(solutions, image_shape)
    global_gates = {
        "exact_filter_solutions": len(solutions) == int(gates_config["exact_filter_count"]),
        "all_filter_gates": bool(filter_results)
        and all(all(result["gates"].values()) for result in filter_results.values()),
        "minimum_shared_accepted_gaia_ids": len(shared_ids)
        >= int(gates_config["minimum_shared_accepted_gaia_ids_all_filters"]),
        "geometric_center_consistency": bool(center_delta)
        and max(center_delta.values())
        <= float(gates_config["maximum_geometric_center_disagreement_arcsec"]),
        "no_detection_rematching_or_prohibited_access": True,
    }
    all_pass = not failures and all(global_gates.values())

    diagnostic_path = output / "subpixel_astrometric_residuals.png"
    if filter_results:
        figure, axes = plt.subplots(1, len(filter_results), figsize=(4 * len(filter_results), 3.5), squeeze=False)
        for axis, filter_name in zip(axes.ravel(), sorted(filter_results), strict=True):
            result = filter_results[filter_name]
            bins = np.linspace(0, 4, 33)
            axis.hist(result["fitted_residual"], bins=bins, alpha=0.65, label="fitted")
            axis.hist(result["loo_residual"], bins=bins, alpha=0.55, label="leave-one-out")
            axis.axvline(1.0, color="#b22222", linestyle="--")
            axis.set_title(filter_name)
            axis.set_xlabel("Residual (pixel)")
        axes[0, 0].set_ylabel("Foreground stars")
        axes[0, 0].legend()
        figure.tight_layout()
        figure.savefig(diagnostic_path, dpi=180)
        plt.close(figure)

    report = {
        "report_version": config["protocol_version"],
        "generated_utc": datetime.now(UTC).isoformat(),
        "status": "passed_subpixel_astrometry" if all_pass else "failed_subpixel_astrometry",
        "input_hashes": input_hashes,
        "filters": {
            name: {
                "metrics": result["metrics"],
                "gates": result["gates"],
                "wcs_path": str(result["wcs_path"].relative_to(ROOT)).replace("\\", "/"),
                "wcs_sha256": sha256(result["wcs_path"]),
                "refined_matches_path": str(result["matches_path"].relative_to(ROOT)).replace("\\", "/"),
                "refined_matches_sha256": sha256(result["matches_path"]),
            }
            for name, result in sorted(filter_results.items())
        },
        "failures": failures,
        "shared_accepted_gaia_ids_all_filters": len(shared_ids),
        "geometric_center_separation_arcsec": center_delta,
        "global_gates": global_gates,
        "all_subpixel_astrometry_gates_pass": all_pass,
        "diagnostic_path": str(diagnostic_path.relative_to(ROOT)).replace("\\", "/")
        if diagnostic_path.exists()
        else None,
        "diagnostic_sha256": sha256(diagnostic_path) if diagnostic_path.exists() else None,
        "source_detection_or_rematching_run": False,
        "member_or_candidate_coordinate_or_cutout_opened": False,
        "photometry_or_deblending_fitted": False,
        "stellar_mass_or_current_inferred": False,
        "lensing_or_halo_payload_opened": False,
        "gravity_formula_or_parameter_changed": False,
        "claim_boundary": config["claim_boundary"],
    }
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
