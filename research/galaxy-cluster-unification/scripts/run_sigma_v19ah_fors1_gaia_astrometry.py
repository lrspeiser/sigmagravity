#!/usr/bin/env python3
"""Fit the frozen V19AH Gaia astrometry for the calibrated FORS1 triplet."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
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
from astropy.time import Time
from astropy.wcs import WCS


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from voidscreen.astrometric_registration import solve_foreground_star_wcs


DEFAULT_CONFIG = ROOT / "configs" / "sigma_v19ah_fors1_gaia_astrometry.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def prepare_detection_image(image: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
    values = np.asarray(image, dtype=np.float64)
    if values.ndim != 2:
        raise RuntimeError("V19AH science image is not two dimensional")
    finite = np.isfinite(values)
    if not np.any(finite):
        raise RuntimeError("V19AH science image has no finite pixels")
    median = float(np.median(values[finite]))
    prepared = values.copy()
    prepared[~finite] = median
    return prepared, {
        "finite_fraction": float(np.mean(finite)),
        "nonfinite_fill_value_adu": median,
    }


def select_and_propagate_gaia(
    catalog: pd.DataFrame, selection: dict[str, Any], target_epoch: float
) -> pd.DataFrame:
    frame = catalog.copy()
    numeric = [
        "ref_epoch",
        "ra",
        "dec",
        "pmra",
        "pmdec",
        "phot_g_mean_mag",
        "ruwe",
        "visibility_periods_used",
        "astrometric_params_solved",
    ]
    for column in numeric:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    duplicated = frame["duplicated_source"].astype(str).str.lower().eq("true")
    keep = (
        frame["astrometric_params_solved"].isin(selection["required_astrometric_params_solved"])
        & (frame["phot_g_mean_mag"] <= float(selection["maximum_g_magnitude"]))
        & (frame["ruwe"] < float(selection["maximum_ruwe"]))
        & (
            frame["visibility_periods_used"]
            >= int(selection["minimum_visibility_periods_used"])
        )
        & (~duplicated)
        & np.isfinite(frame["ra"])
        & np.isfinite(frame["dec"])
        & np.isfinite(frame["pmra"])
        & np.isfinite(frame["pmdec"])
        & np.isfinite(frame["ref_epoch"])
    )
    selected = frame.loc[keep].copy()
    dt = target_epoch - selected["ref_epoch"].to_numpy(dtype=float)
    dec_rad = np.deg2rad(selected["dec"].to_numpy(dtype=float))
    cos_dec = np.cos(dec_rad)
    if np.any(np.abs(cos_dec) < 1e-6):
        raise RuntimeError("Gaia propagation encountered a polar source")
    selected["target_epoch"] = target_epoch
    selected["ra_epoch"] = selected["ra"].to_numpy(dtype=float) + (
        selected["pmra"].to_numpy(dtype=float) * dt / cos_dec / 3_600_000.0
    )
    selected["dec_epoch"] = selected["dec"].to_numpy(dtype=float) + (
        selected["pmdec"].to_numpy(dtype=float) * dt / 3_600_000.0
    )
    selected = selected.sort_values("source_id", kind="stable").reset_index(drop=True)
    return selected


def json_wcs(wcs: WCS) -> dict[str, Any]:
    header = wcs.to_header(relax=True)
    result: dict[str, Any] = {}
    for card in header.cards:
        if card.keyword:
            value = card.value
            if isinstance(value, np.generic):
                value = value.item()
            result[card.keyword] = value
    return result


def validate_config(config_path: Path, config: dict[str, Any]) -> dict[str, str]:
    if (
        config["status"]
        != "frozen_after_equivalent_icrs_frame_metadata_correction_before_any_gaia_match_or_wcs_outcome"
    ):
        raise RuntimeError("V19AH protocol is not frozen")
    runner = ROOT / config["implementation"]["runner"]
    if sha256(runner) != config["implementation"]["runner_sha256"]:
        raise RuntimeError("frozen V19AH runner hash mismatch")
    hashes = {"config": sha256(config_path), "runner": sha256(runner)}
    for artifact in config["parent_artifacts"]:
        path = ROOT / artifact["path"]
        actual = sha256(path)
        if actual != artifact["sha256"]:
            raise RuntimeError(f"V19AH parent hash mismatch: {artifact['path']}")
        hashes[artifact["path"]] = actual
    if len(config["science_products"]) != int(config["gates"]["exact_filter_count"]):
        raise RuntimeError("V19AH science product count changed")
    for product in config["science_products"]:
        path = ROOT / product["path"]
        actual = sha256(path)
        if actual != product["sha256"]:
            raise RuntimeError(f"V19AH science hash mismatch: {product['filter']}")
        hashes[product["path"]] = actual
    authorization = config["authorization"]
    prohibited = [
        "inspect_member_or_candidate_coordinates_or_cutouts",
        "fit_source_photometry_or_deblending",
        "infer_stellar_mass_or_current",
        "read_chandra_source_match_shock_lensing_or_halo_payload",
        "change_gravity_physics_or_parameters",
        "open_holdout",
    ]
    if any(authorization[name] for name in prohibited):
        raise RuntimeError("V19AH authorizes a prohibited action")
    return hashes


def center_separations(solutions: dict[str, dict[str, Any]], shape: tuple[int, int]) -> dict[str, float]:
    y, x = (shape[0] - 1) / 2.0, (shape[1] - 1) / 2.0
    filters = sorted(solutions)
    centers = {
        name: solutions[name]["fit"].wcs.pixel_to_world(x, y) for name in filters
    }
    result: dict[str, float] = {}
    for i, first in enumerate(filters):
        for second in filters[i + 1 :]:
            result[f"{first}__{second}"] = float(
                centers[first].separation(centers[second]).to_value(u.arcsec)
            )
    return result


def run(config_path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    config_path = config_path.resolve()
    config = load_json(config_path)
    input_hashes = validate_config(config_path, config)
    catalog_path = ROOT / "data/raw/sigma_v19g_gaia_dr3/BULLET_gaia_dr3.csv"
    catalog = pd.read_csv(catalog_path, dtype={"source_id": str})
    output = ROOT / config["outputs"]["directory"]
    output.mkdir(parents=True, exist_ok=True)
    gates_config = config["gates"]
    expected_shape = tuple(int(value) for value in config["image_preparation"]["required_shape_yx"])
    solutions: dict[str, dict[str, Any]] = {}
    failures: list[dict[str, str]] = []

    for product in config["science_products"]:
        filter_name = product["filter"]
        try:
            with fits.open(ROOT / product["path"], memmap=False) as hdul:
                image = np.asarray(hdul[0].data, dtype=np.float64)
                header = hdul[0].header.copy()
            if image.shape != expected_shape:
                raise RuntimeError(f"shape changed: {image.shape}")
            prepared, preparation = prepare_detection_image(image)
            target_epoch = float(Time(str(header["DATE-OBS"]), format="isot", scale="utc").jyear)
            gaia = select_and_propagate_gaia(
                catalog, config["gaia_selection"], target_epoch
            )
            if len(gaia) < int(gates_config["minimum_selected_gaia_sources"]):
                raise RuntimeError(f"only {len(gaia)} Gaia sources survive quality selection")
            gaia_sky = SkyCoord(
                gaia["ra_epoch"].to_numpy(dtype=float) * u.deg,
                gaia["dec_epoch"].to_numpy(dtype=float) * u.deg,
                frame="icrs",
            )
            center = SkyCoord(float(header["CRVAL1"]) * u.deg, float(header["CRVAL2"]) * u.deg)
            pixel_scale = 3600.0 * float(
                np.mean([abs(float(header["CDELT1"])), abs(float(header["CDELT2"]))])
            )
            fit = solve_foreground_star_wcs(
                prepared,
                catalog_center=center,
                catalog_pixel_scale_arcsec=pixel_scale,
                gaia_sky=gaia_sky,
                settings=config["algorithm"],
            )
            nearest, separation, _ = fit.matched_sky.match_to_catalog_sky(gaia_sky)
            if float(np.max(separation.to_value(u.mas))) > 1e-3:
                raise RuntimeError("fitted Gaia identities are not exact catalog rows")
            matched = gaia.iloc[np.asarray(nearest, dtype=int)].copy().reset_index(drop=True)
            matched["image_x_pixel"] = fit.matched_pixel_xy[:, 0]
            matched["image_y_pixel"] = fit.matched_pixel_xy[:, 1]
            matched["residual_pixel"] = fit.residual_pixel
            matched["residual_arcsec"] = fit.residual_pixel * pixel_scale
            matched_path = output / f"matched_stars_{filter_name}.csv"
            matched.to_csv(matched_path, index=False)
            diagnostics = dict(fit.diagnostics)
            diagnostics.update(preparation)
            diagnostics["pixel_scale_arcsec"] = pixel_scale
            diagnostics["target_epoch_jyear"] = target_epoch
            diagnostics["selected_gaia_sources"] = int(len(gaia))
            diagnostics["p95_residual_arcsec"] = float(
                diagnostics["p95_residual_pixel"] * pixel_scale
            )
            diagnostics["fractional_similarity_scale_error"] = abs(
                float(diagnostics["similarity_scale"]) - 1.0
            )
            filter_gates = {
                "minimum_gaia_inliers": int(diagnostics["gaia_inliers"])
                >= int(gates_config["minimum_gaia_inliers_per_filter"]),
                "median_residual": float(diagnostics["median_residual_pixel"])
                <= float(gates_config["maximum_median_residual_pixel"]),
                "p95_residual_pixel": float(diagnostics["p95_residual_pixel"])
                <= float(gates_config["maximum_p95_residual_pixel"]),
                "p95_residual_arcsec": float(diagnostics["p95_residual_arcsec"])
                <= float(gates_config["maximum_p95_residual_arcsec"]),
                "similarity_scale": float(diagnostics["fractional_similarity_scale_error"])
                <= float(gates_config["maximum_fractional_similarity_scale_error"]),
                "orientation": diagnostics["orientation"] == gates_config["required_orientation"],
                "similarity_rotation": abs(float(diagnostics["similarity_rotation_deg"]))
                <= float(gates_config["maximum_absolute_similarity_rotation_deg"]),
            }
            wcs_payload = {
                "filter": filter_name,
                "wcs_header": json_wcs(fit.wcs),
                "diagnostics": diagnostics,
                "gates": filter_gates,
                "matched_stars": str(matched_path.relative_to(ROOT)).replace("\\", "/"),
                "matched_stars_sha256": sha256(matched_path),
                "member_or_candidate_coordinate_or_cutout_opened": False,
            }
            wcs_path = output / f"wcs_{filter_name}.json"
            wcs_path.write_text(json.dumps(wcs_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            solutions[filter_name] = {
                "fit": fit,
                "diagnostics": diagnostics,
                "gates": filter_gates,
                "source_ids": set(matched["source_id"].astype(str)),
                "wcs_path": wcs_path,
                "matched_path": matched_path,
            }
        except Exception as exc:
            failures.append({"filter": filter_name, "error": f"{type(exc).__name__}: {exc}"})

    center_delta: dict[str, float] = {}
    shared_ids: set[str] = set()
    if len(solutions) == int(gates_config["exact_filter_count"]):
        center_delta = center_separations(solutions, expected_shape)
        source_sets = [solution["source_ids"] for solution in solutions.values()]
        shared_ids = set.intersection(*source_sets)
    global_gates = {
        "exact_filter_solutions": len(solutions) == int(gates_config["exact_filter_count"]),
        "all_filter_gates": bool(solutions)
        and all(all(solution["gates"].values()) for solution in solutions.values()),
        "minimum_shared_gaia_inliers": len(shared_ids)
        >= int(gates_config["minimum_shared_gaia_inliers_all_filters"]),
        "geometric_center_consistency": bool(center_delta)
        and max(center_delta.values())
        <= float(gates_config["maximum_geometric_center_disagreement_arcsec"]),
        "no_prohibited_access": True,
    }
    all_pass = not failures and all(global_gates.values())

    figure_path = output / "astrometric_residuals.png"
    if solutions:
        figure, axes = plt.subplots(1, len(solutions), figsize=(4 * len(solutions), 3.5), squeeze=False)
        for axis, filter_name in zip(axes.ravel(), sorted(solutions), strict=True):
            residual = solutions[filter_name]["fit"].residual_pixel
            axis.hist(residual, bins=np.linspace(0, 3, 31), color="#285f8f")
            axis.axvline(float(gates_config["maximum_p95_residual_pixel"]), color="#b22222", linestyle="--")
            axis.set_title(filter_name)
            axis.set_xlabel("Residual (pixel)")
        axes[0, 0].set_ylabel("Gaia matches")
        figure.tight_layout()
        figure.savefig(figure_path, dpi=180)
        plt.close(figure)

    report = {
        "report_version": config["protocol_version"],
        "generated_utc": datetime.now(UTC).isoformat(),
        "status": "passed_gaia_astrometry" if all_pass else "failed_gaia_astrometry",
        "input_hashes": input_hashes,
        "filters": {
            name: {
                "diagnostics": solution["diagnostics"],
                "gates": solution["gates"],
                "wcs_path": str(solution["wcs_path"].relative_to(ROOT)).replace("\\", "/"),
                "wcs_sha256": sha256(solution["wcs_path"]),
                "matched_path": str(solution["matched_path"].relative_to(ROOT)).replace("\\", "/"),
                "matched_sha256": sha256(solution["matched_path"]),
            }
            for name, solution in sorted(solutions.items())
        },
        "failures": failures,
        "shared_gaia_inliers_all_filters": len(shared_ids),
        "geometric_center_separation_arcsec": center_delta,
        "global_gates": global_gates,
        "all_astrometry_gates_pass": all_pass,
        "diagnostic_path": str(figure_path.relative_to(ROOT)).replace("\\", "/")
        if figure_path.exists()
        else None,
        "diagnostic_sha256": sha256(figure_path) if figure_path.exists() else None,
        "member_or_candidate_coordinate_or_cutout_opened": False,
        "photometry_or_deblending_fitted": False,
        "stellar_mass_or_current_inferred": False,
        "chandra_source_match_shock_lensing_or_halo_payload_opened": False,
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
                "filters": {
                    name: values["diagnostics"] for name, values in report["filters"].items()
                },
                "global_gates": report["global_gates"],
                "failures": report["failures"],
            },
            indent=2,
        )
    )
    return 0 if report["all_astrometry_gates_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
