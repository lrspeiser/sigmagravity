#!/usr/bin/env python3
"""Measure every frozen V19AU candidate/exposure with the V19AT image rule."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
from collections import Counter, defaultdict
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
DEFAULT_CONFIG = ROOT / "configs" / "sigma_v19au_ambiguous_candidate_image_measurement.json"
V19AS_PATH = ROOT / "scripts" / "run_sigma_v19as_decam_forced_photometry_development.py"


def load_v19as():
    spec = importlib.util.spec_from_file_location("sigma_v19as_frozen_for_v19au", V19AS_PATH)
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


def failed_row(row: dict[str, str], error: str) -> dict[str, Any]:
    return {
        "candidate_id": row["candidate_id"],
        "group_id": row["group_id"],
        "exposure": row["exposure"],
        "sia_extension": row["sia_extension"],
        "filter": row["filter"],
        "flux": float("nan"),
        "flux_uncertainty": float("nan"),
        "magnitude": float("nan"),
        "magnitude_uncertainty": float("nan"),
        "magzero": float("nan"),
        "pixel_scale_arcsec": float("nan"),
        "fwhm_pixel": float("nan"),
        "background_noise": float("nan"),
        "background_pixels": 0,
        "detected_neighbours": 0,
        "usable_aperture_pixels": 0,
        "total_aperture_pixels": 0,
        "measurement_status": "failed_retained",
        "measurement_error": error,
        "image_path": row["image_path"],
        "image_sha256": row["image_sha256"],
    }


def aggregate(
    rows: list[dict[str, Any]],
    candidate_ids: list[str],
    filters: list[str],
) -> list[dict[str, Any]]:
    planned = Counter((row["candidate_id"], row["filter"]) for row in rows)
    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    grouped_flux: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in rows:
        magnitude = float(row["magnitude"])
        flux = float(row["flux"])
        if np.isfinite(magnitude):
            grouped[(row["candidate_id"], row["filter"])].append(magnitude)
        if np.isfinite(flux):
            grouped_flux[(row["candidate_id"], row["filter"])].append(flux)
    result: list[dict[str, Any]] = []
    for candidate_id in candidate_ids:
        for band in filters:
            values = np.asarray(grouped.get((candidate_id, band), []))
            fluxes = np.asarray(grouped_flux.get((candidate_id, band), []))
            median = float(np.median(values)) if values.size else float("nan")
            scatter = V19AS.robust_sigma(values) if values.size >= 2 else float("nan")
            median_flux = float(np.median(fluxes)) if fluxes.size else float("nan")
            flux_scatter = V19AS.robust_sigma(fluxes) if fluxes.size >= 2 else float("nan")
            count = planned[(candidate_id, band)]
            result.append(
                {
                    "candidate_id": candidate_id,
                    "filter": band,
                    "planned_exposures": count,
                    "finite_flux_exposures": int(fluxes.size),
                    "valid_exposures": int(values.size),
                    "valid_fraction": float(values.size / count) if count else 0.0,
                    "median_magnitude": median,
                    "robust_scatter_mag": scatter,
                    "median_flux": median_flux,
                    "robust_flux_scatter": flux_scatter,
                }
            )
    return result


def run(config_path: Path) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    for parent in config["parent_artifacts"]:
        path = ROOT / parent["path"]
        if sha256(path) != parent["sha256"]:
            raise RuntimeError(f"parent artifact hash changed: {parent['path']}")

    plan = read_csv(ROOT / config["inputs"]["candidate_measurement_plan"])
    if len(plan) != int(config["gates"]["exact_candidate_exposure_measurements"]):
        raise RuntimeError("candidate measurement plan count changed")
    candidate_ids = sorted({row["candidate_id"] for row in plan})
    if len(candidate_ids) != int(config["gates"]["exact_unique_candidates"]):
        raise RuntimeError("candidate count changed")
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in plan:
        grouped[row["group_id"]].append(row)
    if len(grouped) != int(config["gates"]["exact_image_groups"]):
        raise RuntimeError("image-group count changed")

    measurement = config["frozen_measurement"]
    diameter = float(measurement["aperture_diameter_arcsec"])
    measurement_rows: list[dict[str, Any]] = []
    group_audit: list[dict[str, Any]] = []
    for group_id in sorted(grouped):
        group_rows = grouped[group_id]
        image_path = ROOT / group_rows[0]["image_path"]
        expected_hash = group_rows[0]["image_sha256"]
        if any(
            row["image_path"] != group_rows[0]["image_path"]
            or row["image_sha256"] != expected_hash
            for row in group_rows
        ):
            raise RuntimeError(f"inconsistent image identity within {group_id}")
        if sha256(image_path) != expected_hash:
            raise RuntimeError(f"image hash changed: {group_rows[0]['image_path']}")

        group_failures = 0
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

            for row in group_rows:
                try:
                    position = SkyCoord(
                        float(row["candidate_ra_deg"]),
                        float(row["candidate_dec_deg"]),
                        unit="deg",
                    )
                    cutout = Cutout2D(
                        image,
                        position,
                        size=float(measurement["cutout_size_arcsec"]) * u.arcsec,
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
                        float(measurement["background_annulus_arcsec"][0]),
                        float(measurement["background_annulus_arcsec"][1]),
                        excluded,
                    )
                    signal = np.asarray(cutout.data, dtype=float) - plane
                    neighbours, neighbour_count = V19AS.neighbour_mask(
                        signal,
                        (float(cx), float(cy)),
                        fwhm_pixel,
                        noise,
                        excluded,
                        float(measurement["detection_threshold_sigma"]),
                    )
                    flux, usable_pixels, total_pixels = V19AS.aperture_fluxes(
                        signal,
                        (float(cx), float(cy)),
                        0.5 * diameter / pixel_scale,
                        neighbours,
                    )[measurement["variant"]]
                    magnitude = (
                        magzero - 2.5 * math.log10(flux)
                        if np.isfinite(flux) and flux > 0
                        else float("nan")
                    )
                    clean_fraction = usable_pixels / total_pixels if total_pixels else 0.0
                    flux_uncertainty = (
                        noise * math.sqrt(max(usable_pixels, 1)) / clean_fraction
                        if clean_fraction > 0
                        else float("nan")
                    )
                    magnitude_uncertainty = (
                        2.5 / math.log(10.0) * flux_uncertainty / flux
                        if np.isfinite(flux_uncertainty) and np.isfinite(flux) and flux > 0
                        else float("nan")
                    )
                    status = "valid" if np.isfinite(magnitude) else "nonpositive_retained"
                    measurement_rows.append(
                        {
                            "candidate_id": row["candidate_id"],
                            "group_id": group_id,
                            "exposure": row["exposure"],
                            "sia_extension": row["sia_extension"],
                            "filter": row["filter"],
                            "flux": flux,
                            "flux_uncertainty": flux_uncertainty,
                            "magnitude": magnitude,
                            "magnitude_uncertainty": magnitude_uncertainty,
                            "magzero": magzero,
                            "pixel_scale_arcsec": pixel_scale,
                            "fwhm_pixel": fwhm_pixel,
                            "background_noise": noise,
                            "background_pixels": background_pixels,
                            "detected_neighbours": neighbour_count,
                            "usable_aperture_pixels": usable_pixels,
                            "total_aperture_pixels": total_pixels,
                            "measurement_status": status,
                            "measurement_error": "",
                            "image_path": row["image_path"],
                            "image_sha256": expected_hash,
                        }
                    )
                    if status != "valid":
                        group_failures += 1
                except (RuntimeError, ValueError, ArithmeticError) as exc:
                    measurement_rows.append(failed_row(row, f"{type(exc).__name__}: {exc}"))
                    group_failures += 1
        group_audit.append(
            {
                "group_id": group_id,
                "image_path": group_rows[0]["image_path"],
                "image_sha256": expected_hash,
                "planned_candidates": len(group_rows),
                "failed_or_nonpositive_candidates": group_failures,
                "all_candidates_retained": True,
            }
        )

    filters = list(config["frozen_measurement"]["filters"])
    aggregates = aggregate(measurement_rows, candidate_ids, filters)
    valid_rows = sum(row["measurement_status"] == "valid" for row in measurement_rows)
    complete_griz_candidates = sum(
        all(
            any(
                row["candidate_id"] == candidate_id
                and row["filter"] == band
                and int(row["valid_exposures"]) > 0
                for row in aggregates
            )
            for band in ("g", "r", "i", "z")
        )
        for candidate_id in candidate_ids
    )
    complete_grizY_candidates = sum(
        all(
            any(
                row["candidate_id"] == candidate_id
                and row["filter"] == band
                and int(row["valid_exposures"]) > 0
                for row in aggregates
            )
            for band in filters
        )
        for candidate_id in candidate_ids
    )
    minimum_band_valid_fraction = min(float(row["valid_fraction"]) for row in aggregates)
    gate_results = {
        "all_measurement_memberships_retained": len(measurement_rows) == len(plan),
        "overall_valid_fraction": valid_rows / len(measurement_rows)
        >= float(config["quality_gates"]["minimum_overall_valid_fraction"]),
        "complete_griz_candidate_fraction": complete_griz_candidates / len(candidate_ids)
        >= float(config["quality_gates"]["minimum_complete_griz_candidate_fraction"]),
        "no_candidate_association_scored": True,
    }
    passed = all(gate_results.values())

    outputs = config["outputs"]
    measurement_path = ROOT / outputs["measurements"]
    aggregate_path = ROOT / outputs["aggregates"]
    group_path = ROOT / outputs["group_audit"]
    write_csv(measurement_path, measurement_rows, list(measurement_rows[0]))
    write_csv(aggregate_path, aggregates, list(aggregates[0]))
    write_csv(group_path, group_audit, list(group_audit[0]))
    report = {
        "protocol_version": config["protocol_version"],
        "decision": "passed" if passed else "failed_closed",
        "counts": {
            "unique_candidates": len(candidate_ids),
            "image_groups": len(grouped),
            "planned_and_retained_measurements": len(measurement_rows),
            "valid_measurements": valid_rows,
            "complete_griz_candidates": complete_griz_candidates,
            "complete_grizY_candidates": complete_grizY_candidates,
        },
        "quality": {
            "overall_valid_fraction": valid_rows / len(measurement_rows),
            "minimum_candidate_band_valid_fraction": minimum_band_valid_fraction,
            "measurement_status": dict(Counter(row["measurement_status"] for row in measurement_rows)),
        },
        "gate_results": gate_results,
        "association_or_bri_likelihood_computed": False,
        "outputs": {
            "measurements": measurement_path.relative_to(ROOT).as_posix(),
            "measurements_sha256": sha256(measurement_path),
            "aggregates": aggregate_path.relative_to(ROOT).as_posix(),
            "aggregates_sha256": sha256(aggregate_path),
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
    print(json.dumps(run(args.config.resolve()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
