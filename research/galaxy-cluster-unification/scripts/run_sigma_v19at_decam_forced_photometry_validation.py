#!/usr/bin/env python3
"""Run the once-frozen V19AT DECam forced-photometry validation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
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

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "sigma_v19at_decam_forced_photometry_validation.json"
V19AS_PATH = ROOT / "scripts" / "run_sigma_v19as_decam_forced_photometry_development.py"


def load_v19as():
    spec = importlib.util.spec_from_file_location("sigma_v19as_frozen", V19AS_PATH)
    module = importlib.util.module_from_spec(spec)
    if spec.loader is None:
        raise RuntimeError("cannot load frozen V19AS implementation")
    spec.loader.exec_module(module)
    return module


V19AS = load_v19as()


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


def feature(row: dict[str, str]) -> np.ndarray:
    return np.asarray(
        [
            1.0,
            (float(row["B"]) - float(row["R"]) - 2.4) / 1.0,
            (float(row["R"]) - float(row["I"]) - 1.1) / 0.5,
        ]
    )


def fit_development_color_model(
    config: dict[str, Any],
    aggregates: list[dict[str, str]],
    bri: dict[str, dict[str, str]],
) -> tuple[dict[str, np.ndarray], dict[str, float], list[dict[str, Any]]]:
    selected = {
        (row["member_id"], row["filter"]): float(row["median_magnitude"])
        for row in aggregates
        if row["variant"] == config["frozen_measurement"]["variant"]
        and float(row["aperture_diameter_arcsec"])
        == float(config["frozen_measurement"]["aperture_diameter_arcsec"])
    }
    development_ids = config["split"]["development_ids"]
    outputs = [("g", "r"), ("r", "i"), ("i", "z")]
    parameters: dict[str, np.ndarray] = {}
    scales: dict[str, float] = {}
    fit_rows: list[dict[str, Any]] = []
    for first, second in outputs:
        name = f"{first}_minus_{second}"
        features = np.vstack([feature(bri[member]) for member in development_ids])
        values = np.asarray(
            [selected[(member, first)] - selected[(member, second)] for member in development_ids]
        )
        fitted = V19AS.affine_fit(
            features,
            values,
            float(config["color_model"]["ridge_penalty"]),
        )
        residuals = values - features @ fitted
        scale = max(
            float(config["color_model"]["predictive_scale_floor_mag"]),
            float(V19AS.robust_sigma(residuals)),
        )
        parameters[name] = fitted
        scales[name] = scale
        for member, observed, predicted, residual in zip(
            development_ids, values, features @ fitted, residuals
        ):
            fit_rows.append(
                {
                    "color": name,
                    "member_id": member,
                    "observed_color": observed,
                    "fitted_color": predicted,
                    "residual": residual,
                    "predictive_scale": scale,
                }
            )
    return parameters, scales, fit_rows


def aggregate_validation(
    measurements: list[dict[str, Any]],
    validation_ids: list[str],
) -> tuple[list[dict[str, Any]], dict[tuple[str, str], float]]:
    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in measurements:
        magnitude = float(row["magnitude"])
        if np.isfinite(magnitude):
            grouped[(row["member_id"], row["filter"])].append(magnitude)
    output: list[dict[str, Any]] = []
    medians: dict[tuple[str, str], float] = {}
    for member in validation_ids:
        for band in ("g", "r", "i", "z", "Y"):
            values = np.asarray(grouped.get((member, band), []))
            median = float(np.median(values)) if values.size else float("nan")
            scatter = V19AS.robust_sigma(values) if values.size >= 2 else float("nan")
            if np.isfinite(median):
                medians[(member, band)] = median
            output.append(
                {
                    "member_id": member,
                    "filter": band,
                    "valid_exposures": int(values.size),
                    "median_magnitude": median,
                    "robust_scatter_mag": scatter,
                }
            )
    return output, medians


def score_validation(
    config: dict[str, Any],
    medians: dict[tuple[str, str], float],
    bri: dict[str, dict[str, str]],
    parameters: dict[str, np.ndarray],
    scales: dict[str, float],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    validation_ids = config["split"]["validation_ids"]
    colors = [("g", "r"), ("r", "i"), ("i", "z")]
    predictions: list[dict[str, Any]] = []
    predicted_vectors: dict[str, np.ndarray] = {}
    observed_vectors: dict[str, np.ndarray] = {}
    for member in validation_ids:
        predicted = []
        observed = []
        row: dict[str, Any] = {"member_id": member}
        for first, second in colors:
            name = f"{first}_minus_{second}"
            predicted_color = float(feature(bri[member]) @ parameters[name])
            observed_color = float(medians.get((member, first), np.nan)) - float(
                medians.get((member, second), np.nan)
            )
            row[f"predicted_{name}"] = predicted_color
            row[f"observed_{name}"] = observed_color
            row[f"residual_{name}"] = observed_color - predicted_color
            predicted.append(predicted_color)
            observed.append(observed_color)
        predictions.append(row)
        predicted_vectors[member] = np.asarray(predicted)
        observed_vectors[member] = np.asarray(observed)

    scale_vector = np.asarray([scales["g_minus_r"], scales["r_minus_i"], scales["i_minus_z"]])
    retrieval: list[dict[str, Any]] = []
    ranks: dict[str, int] = {}
    for source in validation_ids:
        candidates = []
        for candidate in validation_ids:
            vector = observed_vectors[candidate]
            score = (
                float(np.sqrt(np.sum(((vector - predicted_vectors[source]) / scale_vector) ** 2)))
                if np.all(np.isfinite(vector))
                else float("inf")
            )
            candidates.append((score, candidate))
        candidates.sort(key=lambda item: (item[0], item[1]))
        for rank, (score, candidate) in enumerate(candidates, start=1):
            retrieval.append(
                {
                    "source_member_id": source,
                    "candidate_member_id": candidate,
                    "standardized_color_distance": score,
                    "rank": rank,
                    "is_true_pair": source == candidate,
                }
            )
            if source == candidate:
                ranks[source] = rank

    absolute_errors: dict[str, list[float]] = defaultdict(list)
    for row in predictions:
        for name in ("g_minus_r", "r_minus_i", "i_minus_z"):
            absolute_errors[name].append(abs(float(row[f"residual_{name}"])))
    metrics = {
        "median_absolute_error_mag": {
            name: float(np.median(values)) for name, values in absolute_errors.items()
        },
        "top1_retrievals": sum(rank == 1 for rank in ranks.values()),
        "mean_reciprocal_rank": float(np.mean([1.0 / rank for rank in ranks.values()])),
        "true_pair_ranks": ranks,
    }
    return predictions, retrieval, metrics


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
    development_aggregates = read_csv(ROOT / config["inputs"]["development_aggregates"])
    bri = {row["object_id"]: row for row in bri_rows}

    development_ids = set(config["split"]["development_ids"])
    validation_ids = set(config["split"]["validation_ids"])
    actual_development = {row["member_id"] for row in manifest if row["split"] == "development"}
    actual_validation = {row["member_id"] for row in manifest if row["split"] == "validation"}
    if development_ids != actual_development or validation_ids != actual_validation:
        raise RuntimeError("frozen split changed")
    validation_rows = [row for row in manifest if row["member_id"] in validation_ids]
    if any(row["split"] != "validation" for row in validation_rows):
        raise RuntimeError("development row reached validation measurement list")
    if len(validation_rows) != int(config["gates"]["exact_validation_measurements"]):
        raise RuntimeError("validation measurement count changed")

    plan_lookup = {(row["exposure"], row["sia_extension"]): row for row in plans}
    catalog_lookup = {row["measid"]: row for row in catalog_rows}
    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in validation_rows:
        grouped[(row["exposure"], row["sia_extension"])].append(row)
    if len(grouped) != int(config["gates"]["exact_validation_image_groups"]):
        raise RuntimeError("validation image-group count changed")

    measurement_rows: list[dict[str, Any]] = []
    group_audit: list[dict[str, Any]] = []
    diameter = float(config["frozen_measurement"]["aperture_diameter_arcsec"])
    for group_key in sorted(grouped):
        plan = plan_lookup[group_key]
        image_path = ROOT / plan["output_path"]
        if sha256(image_path) != plan["sha256"]:
            raise RuntimeError(f"image hash changed: {plan['output_path']}")
        with fits.open(image_path, memmap=True, checksum=False) as hdul:
            primary = hdul[0].header
            image_hdu = next(hdu for hdu in hdul if hdu.data is not None and hdu.data.ndim == 2)
            image = image_hdu.data
            image_wcs = WCS(image_hdu.header)
            pixel_scale = float(np.mean(proj_plane_pixel_scales(image_wcs.celestial) * 3600.0))
            fwhm_pixel = float(image_hdu.header.get("FWHM", np.nan))
            if not np.isfinite(fwhm_pixel) or fwhm_pixel <= 0:
                fwhm_pixel = float(primary.get("SEEING", np.nan)) / pixel_scale
            magzero = float(primary.get("MAGZERO", np.nan))
            if not np.isfinite(magzero):
                raise RuntimeError(f"missing MAGZERO in {image_path.name}")

            for row in grouped[group_key]:
                position = SkyCoord(float(row["ra_deg"]), float(row["dec_deg"]), unit="deg")
                cutout = Cutout2D(
                    image,
                    position,
                    size=float(config["frozen_measurement"]["cutout_size_arcsec"]) * u.arcsec,
                    wcs=image_wcs,
                    mode="partial",
                    fill_value=np.nan,
                    copy=True,
                )
                cx, cy = cutout.wcs.celestial.world_to_pixel(position)
                excluded = np.zeros(cutout.data.shape, dtype=bool)
                plane, noise, background_pixels = V19AS.fit_background_plane(
                    np.asarray(cutout.data, dtype=float),
                    (float(cx), float(cy)),
                    pixel_scale,
                    float(config["frozen_measurement"]["background_annulus_arcsec"][0]),
                    float(config["frozen_measurement"]["background_annulus_arcsec"][1]),
                    excluded,
                )
                signal = np.asarray(cutout.data, dtype=float) - plane
                neighbours, neighbour_count = V19AS.neighbour_mask(
                    signal,
                    (float(cx), float(cy)),
                    fwhm_pixel,
                    noise,
                    excluded,
                    float(config["frozen_measurement"]["detection_threshold_sigma"]),
                )
                flux, usable_pixels, total_pixels = V19AS.aperture_fluxes(
                    signal,
                    (float(cx), float(cy)),
                    0.5 * diameter / pixel_scale,
                    neighbours,
                )[config["frozen_measurement"]["variant"]]
                magnitude = (
                    magzero - 2.5 * math.log10(flux)
                    if np.isfinite(flux) and flux > 0
                    else float("nan")
                )
                catalog = catalog_lookup[row["measid"]]
                measurement_rows.append(
                    {
                        "member_id": row["member_id"],
                        "nsc_id": row["nsc_id"],
                        "measid": row["measid"],
                        "exposure": row["exposure"],
                        "sia_extension": row["sia_extension"],
                        "filter": row["filter"],
                        "variant": config["frozen_measurement"]["variant"],
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
                        "image_sha256": plan["sha256"],
                    }
                )
        group_audit.append(
            {
                "exposure": group_key[0],
                "sia_extension": group_key[1],
                "validation_members": ";".join(sorted(row["member_id"] for row in grouped[group_key])),
                "image_path": plan["output_path"],
                "image_sha256": plan["sha256"],
                "frozen_variant": config["frozen_measurement"]["variant"],
                "frozen_aperture_diameter_arcsec": diameter,
            }
        )

    aggregates, medians = aggregate_validation(
        measurement_rows,
        config["split"]["validation_ids"],
    )
    parameters, scales, fit_rows = fit_development_color_model(
        config,
        development_aggregates,
        bri,
    )
    predictions, retrieval, metrics = score_validation(
        config,
        medians,
        bri,
        parameters,
        scales,
    )

    gates = config["validation_gates"]
    complete_griz = sum(
        all((member, band) in medians for band in ("g", "r", "i", "z"))
        for member in config["split"]["validation_ids"]
    )
    member57_complete = all(("57", band) in medians for band in ("g", "r", "i", "z"))
    gate_results = {
        "all_validation_objects_complete_griz": complete_griz
        == int(gates["required_complete_griz_objects"]),
        "member57_complete_griz": member57_complete,
        "color_error": all(
            value <= float(gates["maximum_median_absolute_error_each_color_mag"])
            for value in metrics["median_absolute_error_mag"].values()
        ),
        "top1_retrieval": metrics["top1_retrievals"]
        >= int(gates["minimum_top1_retrievals"]),
        "mean_reciprocal_rank": metrics["mean_reciprocal_rank"]
        >= float(gates["minimum_mean_reciprocal_rank"]),
        "all_validation_measurements_retained": len(measurement_rows)
        == int(config["gates"]["exact_validation_measurements"]),
    }
    passed = all(gate_results.values())

    outputs = config["outputs"]
    products = {
        "measurements": (measurement_rows, list(measurement_rows[0])),
        "aggregates": (aggregates, list(aggregates[0])),
        "development_color_fit": (fit_rows, list(fit_rows[0])),
        "validation_predictions": (predictions, list(predictions[0])),
        "validation_retrieval": (retrieval, list(retrieval[0])),
        "group_audit": (group_audit, list(group_audit[0])),
    }
    output_metadata: dict[str, str] = {}
    for name, (rows_to_write, fields) in products.items():
        path = ROOT / outputs[name]
        write_csv(path, rows_to_write, fields)
        output_metadata[name] = path.relative_to(ROOT).as_posix()
        output_metadata[f"{name}_sha256"] = sha256(path)

    report = {
        "protocol_version": config["protocol_version"],
        "decision": "passed" if passed else "failed_closed",
        "counts": {
            "validation_anchors": len(validation_ids),
            "validation_measurements": len(measurement_rows),
            "validation_image_groups": len(grouped),
            "complete_griz_validation_objects": complete_griz,
        },
        "frozen_measurement": config["frozen_measurement"],
        "development_color_model": {
            "parameters": {name: value.tolist() for name, value in parameters.items()},
            "predictive_scales": scales,
        },
        "validation_metrics": metrics,
        "gate_results": gate_results,
        "calibration_boundary": config["calibration_boundary"],
        "outputs": output_metadata,
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
    print(json.dumps(run(args.config.resolve()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
