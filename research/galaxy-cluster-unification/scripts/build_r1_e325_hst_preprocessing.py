#!/usr/bin/env python3
"""Build the frozen E325 common-grid HST products without arc selection."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales
from scipy.ndimage import map_coordinates


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/r1_e325_hst_preprocessing_protocol.json"
UPSTREAM_PATH = ROOT / "results/r1_e325_acquisition/report.json"
PROVENANCE_PATH = ROOT / "data/raw/r1_e325_science/provenance.json"


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def common_wcs(config: dict) -> WCS:
    grid = config["common_grid"]
    ny, nx = grid["shape_yx"]
    scale_deg = grid["pixel_scale_arcsec"] / 3600.0
    wcs = WCS(naxis=2)
    wcs.wcs.crpix = [(nx + 1) / 2.0, (ny + 1) / 2.0]
    wcs.wcs.crval = [grid["center_ra_deg"], grid["center_dec_deg"]]
    wcs.wcs.cdelt = [-scale_deg, scale_deg]
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    wcs.wcs.cunit = ["deg", "deg"]
    return wcs


def reproject_visit(path: Path, output_wcs: WCS, shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray, dict]:
    yy, xx = np.indices(shape, dtype=float)
    ra, dec = output_wcs.pixel_to_world_values(xx, yy)
    with fits.open(path, memmap=True, do_not_scale_image_data=True) as hdul:
        input_wcs = WCS(hdul["SCI"].header).celestial
        input_x, input_y = input_wcs.world_to_pixel_values(ra, dec)
        coordinates = np.vstack([input_y.ravel(), input_x.ravel()])
        science = map_coordinates(
            np.asarray(hdul["SCI"].data), coordinates, order=1, mode="constant", cval=np.nan
        ).reshape(shape)
        weight = map_coordinates(
            np.asarray(hdul["WHT"].data), coordinates, order=1, mode="constant", cval=0.0
        ).reshape(shape)
        weight[~np.isfinite(weight) | (weight <= 0)] = 0.0
        header = {
            "filter": str(hdul[0].header["FILTER"]),
            "exposure_seconds": float(hdul[0].header["EXPTIME"]),
            "bunit": str(hdul["SCI"].header.get("BUNIT", "")),
        }
    return science.astype(np.float32), weight.astype(np.float32), header


def moffat_psf(
    shape: tuple[int, int], pixel_scale: float, fwhm: float, beta: float, q: float, pa_deg: float
) -> np.ndarray:
    yy, xx = np.indices(shape, dtype=float)
    cy, cx = (np.array(shape) - 1.0) / 2.0
    dx = (xx - cx) * pixel_scale
    dy = (yy - cy) * pixel_scale
    angle = np.deg2rad(pa_deg)
    major = dx * np.cos(angle) + dy * np.sin(angle)
    minor = -dx * np.sin(angle) + dy * np.cos(angle)
    alpha = fwhm / (2.0 * np.sqrt(2.0 ** (1.0 / beta) - 1.0))
    radius2 = (major / alpha) ** 2 + (minor / (alpha * q)) ** 2
    psf = (1.0 + radius2) ** (-beta)
    psf /= psf.sum()
    return psf.astype(np.float64)


def main() -> None:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    upstream = json.loads(UPSTREAM_PATH.read_text(encoding="utf-8"))
    provenance = json.loads(PROVENANCE_PATH.read_text(encoding="utf-8"))
    if not upstream["authorization"]["construct_registered_observable_cutouts_and_empirical_psf"]:
        raise RuntimeError("Complete acquisition gate did not authorize HST preprocessing")
    if config["image_morphology_seen_at_freeze"]:
        raise RuntimeError("Preprocessing design was not residual-blind")

    grid = config["common_grid"]
    shape = tuple(int(value) for value in grid["shape_yx"])
    output_wcs = common_wcs(config)
    output_header = output_wcs.to_header()
    records = [record for record in provenance["records"] if record["archive"] == "HST_MAST"]
    records.sort(key=lambda item: (item["fits_headers"][0]["filter"], item["product_filename"]))
    if len(records) != 4:
        raise RuntimeError(f"Expected four frozen HST records, found {len(records)}")

    per_filter: dict[str, list[tuple[np.ndarray, np.ndarray, dict, dict]]] = {}
    output_hdus: list[fits.ImageHDU] = []
    visit_metrics: list[dict[str, object]] = []
    for record in records:
        path = ROOT / record["path"]
        science, weight, metadata = reproject_visit(path, output_wcs, shape)
        filter_name = metadata["filter"]
        visits = per_filter.setdefault(filter_name, [])
        visit_number = len(visits) + 1
        visits.append((science, weight, metadata, record))
        prefix = filter_name.replace("W", "")
        sci_name = f"{prefix}V{visit_number}S"
        wht_name = f"{prefix}V{visit_number}W"
        sci_header = output_header.copy()
        sci_header["FILTER"] = filter_name
        sci_header["EXPTIME"] = metadata["exposure_seconds"]
        sci_header["BUNIT"] = metadata["bunit"]
        sci_header["SRCFILE"] = record["product_filename"]
        weight_header = output_header.copy()
        weight_header["FILTER"] = filter_name
        weight_header["EXPTIME"] = metadata["exposure_seconds"]
        weight_header["WHTTYPE"] = "INPUT_DRC_WHT"
        weight_header["SRCFILE"] = record["product_filename"]
        output_hdus.extend(
            [
                fits.ImageHDU(data=science, header=sci_header, name=sci_name),
                fits.ImageHDU(data=weight, header=weight_header, name=wht_name),
            ]
        )
        visit_metrics.append(
            {
                "filter": filter_name,
                "visit": visit_number,
                "source_file": record["product_filename"],
                "science_finite_fraction": float(np.isfinite(science).mean()),
                "positive_weight_fraction": float((weight > 0).mean()),
                "exposure_seconds": metadata["exposure_seconds"],
            }
        )

    coadd_metrics: dict[str, dict[str, object]] = {}
    for filter_name, visits in sorted(per_filter.items()):
        if len(visits) != 2:
            raise RuntimeError(f"Expected two {filter_name} visits, found {len(visits)}")
        numerator = np.zeros(shape, dtype=np.float64)
        denominator = np.zeros(shape, dtype=np.float64)
        for science, weight, _, _ in visits:
            valid = np.isfinite(science) & (weight > 0)
            numerator[valid] += science[valid] * weight[valid]
            denominator[valid] += weight[valid]
        coadd = np.full(shape, np.nan, dtype=np.float32)
        valid = denominator > 0
        coadd[valid] = (numerator[valid] / denominator[valid]).astype(np.float32)
        coadd_weight = denominator.astype(np.float32)
        prefix = filter_name.replace("W", "")
        sci_header = output_header.copy()
        sci_header["FILTER"] = filter_name
        sci_header["EXPTIME"] = sum(item[2]["exposure_seconds"] for item in visits)
        sci_header["BUNIT"] = visits[0][2]["bunit"]
        wht_header = output_header.copy()
        wht_header["FILTER"] = filter_name
        wht_header["EXPTIME"] = sci_header["EXPTIME"]
        wht_header["WHTTYPE"] = "SUM_VISIT_WHT"
        output_hdus.extend(
            [
                fits.ImageHDU(data=coadd, header=sci_header, name=f"{prefix}COA"),
                fits.ImageHDU(data=coadd_weight, header=wht_header, name=f"{prefix}WHT"),
            ]
        )
        coadd_metrics[filter_name] = {
            "exposure_seconds": float(sci_header["EXPTIME"]),
            "science_finite_fraction": float(np.isfinite(coadd).mean()),
            "positive_weight_fraction": float((coadd_weight > 0).mean()),
        }

    output_path = ROOT / config["outputs"]["registered_cutouts"]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    primary_header = fits.Header()
    primary_header["PROTOVER"] = config["protocol_version"]
    primary_header["MORPHSEE"] = False
    primary_header["GRAVSEE"] = False
    fits.HDUList([fits.PrimaryHDU(header=primary_header), *output_hdus]).writeto(
        output_path, overwrite=True, checksum=True
    )

    psf_config = config["psf_family"]
    psf_arrays: dict[str, np.ndarray] = {}
    for filter_name, nominal in psf_config["nominal_fwhm_arcsec"].items():
        for multiplier in psf_config["frozen_fwhm_multipliers"]:
            for q in psf_config["frozen_axis_ratios"]:
                for pa in psf_config["frozen_position_angles_deg"]:
                    key = f"{filter_name}_m{multiplier:.1f}_q{q:.1f}_pa{int(pa):03d}"
                    psf_arrays[key] = moffat_psf(
                        tuple(psf_config["stamp_shape_yx"]),
                        grid["pixel_scale_arcsec"],
                        nominal * multiplier,
                        psf_config["beta"],
                        q,
                        pa,
                    )
    psf_path = ROOT / config["outputs"]["psf_family"]
    np.savez_compressed(psf_path, **psf_arrays)

    expected_exposures = config["gates"]["coadd_exposure_seconds"]
    minimum_weight = config["noise"]["minimum_positive_weight_fraction"]
    visits_pass = all(
        metric["science_finite_fraction"] == 1.0
        and metric["positive_weight_fraction"] >= minimum_weight
        for metric in visit_metrics
    )
    coadds_pass = all(
        coadd_metrics[name]["exposure_seconds"] == expected_exposures[name]
        and coadd_metrics[name]["science_finite_fraction"] == 1.0
        and coadd_metrics[name]["positive_weight_fraction"] >= minimum_weight
        for name in expected_exposures
    )
    scales = proj_plane_pixel_scales(output_wcs) * 3600.0
    center_x, center_y = output_wcs.world_to_pixel_values(
        grid["center_ra_deg"], grid["center_dec_deg"]
    )
    intended_center = (np.array(shape[::-1]) - 1.0) / 2.0
    center_offset_arcsec = float(
        np.hypot(center_x - intended_center[0], center_y - intended_center[1])
        * grid["pixel_scale_arcsec"]
    )
    scale_fractional_error = float(
        np.max(np.abs(scales - grid["pixel_scale_arcsec"]) / grid["pixel_scale_arcsec"])
    )
    psf_pass = bool(
        len(psf_arrays) == 36
        and all(array.shape == (41, 41) for array in psf_arrays.values())
        and all(abs(float(array.sum()) - 1.0) < 1e-12 for array in psf_arrays.values())
    )
    complete = bool(
        visits_pass
        and coadds_pass
        and center_offset_arcsec <= config["gates"]["all_common_grid_center_offsets_arcsec_maximum"]
        and scale_fractional_error <= config["gates"]["all_output_pixel_scale_fractional_error_maximum"]
        and psf_pass
    )
    report = {
        "report_version": config["protocol_version"],
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "selection_blind": True,
        "image_morphology_inspected": False,
        "gravity_residuals_inspected": False,
        "inputs": {
            "protocol": {"path": str(CONFIG_PATH.relative_to(ROOT)).replace("\\", "/"), "sha256": sha256(CONFIG_PATH)},
            "acquisition_report": {"path": str(UPSTREAM_PATH.relative_to(ROOT)).replace("\\", "/"), "sha256": sha256(UPSTREAM_PATH)},
            "raw_provenance": {"path": str(PROVENANCE_PATH.relative_to(ROOT)).replace("\\", "/"), "sha256": sha256(PROVENANCE_PATH)},
        },
        "common_grid": {
            "shape_yx": list(shape),
            "pixel_scale_arcsec": [float(value) for value in scales],
            "center_pixel_zero_based": [float(center_x), float(center_y)],
            "center_offset_arcsec": center_offset_arcsec,
            "scale_fractional_error": scale_fractional_error,
        },
        "visit_metrics": visit_metrics,
        "coadd_metrics": coadd_metrics,
        "psf_family": {
            "members": len(psf_arrays),
            "all_normalized": psf_pass,
            "empirical_star_selection_attempted": False,
        },
        "gates": {
            "visit_reprojection_integrity_passed": visits_pass,
            "coadd_integrity_passed": coadds_pass,
            "common_grid_astrometry_passed": center_offset_arcsec <= config["gates"]["all_common_grid_center_offsets_arcsec_maximum"],
            "common_grid_scale_passed": scale_fractional_error <= config["gates"]["all_output_pixel_scale_fractional_error_maximum"],
            "frozen_psf_family_passed": psf_pass,
            "complete_preprocessing_gate_passed": complete,
            "rank_three_candidate_admission_passed": False,
        },
        "outputs": {
            "registered_cutouts": str(output_path.relative_to(ROOT)).replace("\\", "/"),
            "registered_cutouts_bytes": output_path.stat().st_size,
            "registered_cutouts_sha256": sha256(output_path),
            "psf_family": str(psf_path.relative_to(ROOT)).replace("\\", "/"),
            "psf_family_bytes": psf_path.stat().st_size,
            "psf_family_sha256": sha256(psf_path),
        },
        "decision": "authorize_blind_arc_mask_and_noise_freeze" if complete else "stop_preprocessing_and_do_not_fit",
        "authorization": {
            "freeze_arc_and_negative_control_masks": complete,
            "estimate_drizzle_noise_inflation_after_mask_freeze": complete,
            "implement_frozen_image_level_jacobian": False,
            "count_toward_ten_system_target": False,
            "infer_gravity_response": False,
            "fit_new_force_or_action": False,
        },
    }
    report_path = ROOT / config["outputs"]["report"]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
