#!/usr/bin/env python3
"""Audit the complete E325 immutable receipt and minimal array integrity."""

from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import astropy.units as u
import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/r1_e325_acquisition_jacobian_protocol.json"
PROVENANCE_PATH = ROOT / "data/raw/r1_e325_science/provenance.json"
INVENTORY_PATH = ROOT / "data/derived/r1_e325_acquisition_integrity.csv"
REPORT_PATH = ROOT / "results/r1_e325_acquisition/report.json"


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def central_slices(x: float, y: float, radius_pixels: float, shape: tuple[int, int]) -> tuple[slice, slice]:
    x0 = max(0, int(math.floor(x - radius_pixels)))
    x1 = min(shape[1], int(math.ceil(x + radius_pixels + 1)))
    y0 = max(0, int(math.floor(y - radius_pixels)))
    y1 = min(shape[0], int(math.ceil(y + radius_pixels + 1)))
    return slice(y0, y1), slice(x0, x1)


def audit_hst(path: Path, product: dict, target: SkyCoord) -> dict[str, object]:
    with fits.open(path, memmap=True, do_not_scale_image_data=True) as hdul:
        names = [hdu.name for hdu in hdul]
        required_hdus = all(name in names for name in ("SCI", "WHT", "CTX"))
        primary = hdul[0].header
        sci_hdu = hdul["SCI"]
        wht_hdu = hdul["WHT"]
        wcs = WCS(sci_hdu.header).celestial
        x, y = wcs.world_to_pixel(target)
        scales = proj_plane_pixel_scales(wcs) * 3600.0
        pixel_scale = float(np.mean(scales))
        margin_pixels = 6.0 / pixel_scale
        shape = tuple(int(value) for value in sci_hdu.data.shape)
        ys, xs = central_slices(float(x), float(y), margin_pixels, shape)
        sci = np.asarray(sci_hdu.data[ys, xs])
        wht = np.asarray(wht_hdu.data[ys, xs])
        target_inside_with_margin = bool(
            x >= margin_pixels
            and y >= margin_pixels
            and x < shape[1] - margin_pixels
            and y < shape[0] - margin_pixels
        )
        return {
            "archive": product["archive"],
            "product_filename": product["product_filename"],
            "filter": str(primary.get("FILTER", "")),
            "exposure_seconds": float(primary.get("EXPTIME", 0.0)),
            "science_shape": "x".join(str(value) for value in shape),
            "required_hdus_pass": required_hdus,
            "pixel_scale_arcsec": pixel_scale,
            "target_x_zero_based": float(x),
            "target_y_zero_based": float(y),
            "target_inside_with_6arcsec_margin": target_inside_with_margin,
            "central_cutout_pixels": int(sci.size),
            "central_science_finite_fraction": float(np.isfinite(sci).mean()),
            "central_weight_finite_fraction": float(np.isfinite(wht).mean()),
            "central_positive_weight_fraction": float((np.isfinite(wht) & (wht > 0)).mean()),
            "spectral_min_angstrom": np.nan,
            "spectral_max_angstrom": np.nan,
            "stat_positive_fraction": np.nan,
        }


def audit_muse(path: Path, product: dict, target: SkyCoord) -> dict[str, object]:
    with fits.open(path, memmap=True, do_not_scale_image_data=True) as hdul:
        names = [hdu.name for hdu in hdul]
        required_hdus = all(name in names for name in ("DATA", "STAT"))
        data_hdu = hdul["DATA"]
        stat_hdu = hdul["STAT"]
        data = np.asarray(data_hdu.data)
        stat = np.asarray(stat_hdu.data)
        celestial = WCS(data_hdu.header).celestial
        x, y = celestial.world_to_pixel(target)
        scales = proj_plane_pixel_scales(celestial) * 3600.0
        pixel_scale = float(np.mean(scales))
        spatial_shape = data.shape[-2:]
        margin_pixels = 4.0 / pixel_scale
        target_inside = bool(
            x >= margin_pixels
            and y >= margin_pixels
            and x < spatial_shape[1] - margin_pixels
            and y < spatial_shape[0] - margin_pixels
        )
        header = data_hdu.header
        spectral_unit = u.Unit(str(header.get("CUNIT3", "Angstrom")))
        crval = float(header["CRVAL3"])
        crpix = float(header["CRPIX3"])
        cdelt = float(header.get("CDELT3", header.get("CD3_3")))
        first = (crval + (1.0 - crpix) * cdelt) * spectral_unit
        last = (crval + (data.shape[0] - crpix) * cdelt) * spectral_unit
        spectral_min = min(first.to_value(u.AA), last.to_value(u.AA))
        spectral_max = max(first.to_value(u.AA), last.to_value(u.AA))
        return {
            "archive": product["archive"],
            "product_filename": product["product_filename"],
            "filter": "",
            "exposure_seconds": float(hdul[0].header.get("EXPTIME", 0.0)),
            "science_shape": "x".join(str(value) for value in data.shape),
            "required_hdus_pass": required_hdus,
            "pixel_scale_arcsec": pixel_scale,
            "target_x_zero_based": float(x),
            "target_y_zero_based": float(y),
            "target_inside_with_6arcsec_margin": target_inside,
            "central_cutout_pixels": int(data.size),
            "central_science_finite_fraction": float(np.isfinite(data).mean()),
            "central_weight_finite_fraction": float(np.isfinite(stat).mean()),
            "central_positive_weight_fraction": np.nan,
            "spectral_min_angstrom": float(spectral_min),
            "spectral_max_angstrom": float(spectral_max),
            "stat_positive_fraction": float((np.isfinite(stat) & (stat > 0)).mean()),
        }


def main() -> None:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    provenance = json.loads(PROVENANCE_PATH.read_text(encoding="utf-8"))
    products = config["acquisition"]["products"]
    product_by_name = {product["product_filename"]: product for product in products}
    records_by_name = {record["product_filename"]: record for record in provenance["records"]}
    expected_names = set(product_by_name)
    receipt_names = set(records_by_name)
    exact_product_set = expected_names == receipt_names and len(receipt_names) == 5
    target = SkyCoord(config["center"]["ra_deg"] * u.deg, config["center"]["dec_deg"] * u.deg)

    integrity: list[dict[str, object]] = []
    for name in sorted(expected_names):
        product = product_by_name[name]
        record = records_by_name[name]
        path = ROOT / record["path"]
        size_pass = path.exists() and path.stat().st_size == int(product["expected_bytes"])
        hash_pass = bool(size_pass and sha256(path) == record["sha256"])
        if not (size_pass and hash_pass):
            raise RuntimeError(f"Immutable receipt failed before array inspection: {name}")
        if product["archive"] == "HST_MAST":
            measurement = audit_hst(path, product, target)
        else:
            measurement = audit_muse(path, product, target)
        measurement.update(
            {
                "bytes": path.stat().st_size,
                "sha256": record["sha256"],
                "size_pass": size_pass,
                "hash_pass": hash_pass,
            }
        )
        integrity.append(measurement)

    table = pd.DataFrame(integrity).sort_values("product_filename", kind="stable")
    INVENTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(INVENTORY_PATH, index=False, lineterminator="\n")
    hst = table.loc[table["archive"] == "HST_MAST"]
    muse = table.loc[table["archive"] == "ESO_SODA"].iloc[0]
    exposure_by_filter = hst.groupby("filter")["exposure_seconds"].sum().to_dict()
    hst_integrity = bool(
        len(hst) == 4
        and hst["required_hdus_pass"].all()
        and hst["target_inside_with_6arcsec_margin"].all()
        and (hst["central_science_finite_fraction"] == 1.0).all()
        and (hst["central_weight_finite_fraction"] == 1.0).all()
        and (hst["central_positive_weight_fraction"] >= 0.99).all()
        and 0.045 <= hst["pixel_scale_arcsec"].min()
        and hst["pixel_scale_arcsec"].max() <= 0.055
        and exposure_by_filter.get("F814W") == 18882.0
        and exposure_by_filter.get("F475W") == 4800.0
    )
    muse_integrity = bool(
        muse["required_hdus_pass"]
        and muse["target_inside_with_6arcsec_margin"]
        and muse["central_science_finite_fraction"] >= 0.99
        and muse["central_weight_finite_fraction"] >= 0.99
        and muse["stat_positive_fraction"] >= 0.99
        and 0.19 <= muse["pixel_scale_arcsec"] <= 0.21
        and muse["spectral_min_angstrom"] <= 4750.5
        and muse["spectral_max_angstrom"] >= 5599.5
    )
    immutable_receipt = bool(
        exact_product_set
        and table["size_pass"].all()
        and table["hash_pass"].all()
        and provenance["science_arrays_seen_before_protocol_freeze"] is False
    )
    complete_gate = immutable_receipt and hst_integrity and muse_integrity
    report = {
        "report_version": config["protocol_version"],
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "system": config["system"],
        "selection_blind": True,
        "gravity_residuals_inspected": False,
        "science_arrays_inspected_only_after_complete_hash_receipt": True,
        "inputs": {
            "protocol": {
                "path": str(CONFIG_PATH.relative_to(ROOT)).replace("\\", "/"),
                "sha256": sha256(CONFIG_PATH),
            },
            "provenance": {
                "path": str(PROVENANCE_PATH.relative_to(ROOT)).replace("\\", "/"),
                "sha256": sha256(PROVENANCE_PATH),
            },
        },
        "receipt": {
            "exact_product_set_passed": exact_product_set,
            "expected_products": 5,
            "received_products": int(len(table)),
            "received_bytes": int(table["bytes"].sum()),
            "all_sizes_passed": bool(table["size_pass"].all()),
            "all_hashes_passed": bool(table["hash_pass"].all()),
            "output": str(INVENTORY_PATH.relative_to(ROOT)).replace("\\", "/"),
            "output_sha256": sha256(INVENTORY_PATH),
        },
        "hst_integrity": {
            "passed": hst_integrity,
            "products": int(len(hst)),
            "exposure_seconds_by_filter": {key: float(value) for key, value in exposure_by_filter.items()},
            "pixel_scale_arcsec_range": [
                float(hst["pixel_scale_arcsec"].min()),
                float(hst["pixel_scale_arcsec"].max()),
            ],
            "minimum_central_positive_weight_fraction": float(hst["central_positive_weight_fraction"].min()),
            "all_target_footprints_cover_six_arcseconds": bool(hst["target_inside_with_6arcsec_margin"].all()),
        },
        "muse_integrity": {
            "passed": muse_integrity,
            "shape": muse["science_shape"],
            "exposure_seconds": float(muse["exposure_seconds"]),
            "pixel_scale_arcsec": float(muse["pixel_scale_arcsec"]),
            "spectral_range_angstrom": [
                float(muse["spectral_min_angstrom"]),
                float(muse["spectral_max_angstrom"]),
            ],
            "science_finite_fraction": float(muse["central_science_finite_fraction"]),
            "variance_finite_fraction": float(muse["central_weight_finite_fraction"]),
            "variance_positive_fraction": float(muse["stat_positive_fraction"]),
            "target_footprint_covers_four_arcseconds": bool(muse["target_inside_with_6arcsec_margin"]),
        },
        "gates": {
            "immutable_receipt_passed": immutable_receipt,
            "hst_array_integrity_passed": hst_integrity,
            "muse_array_integrity_passed": muse_integrity,
            "complete_acquisition_gate_passed": complete_gate,
            "rank_three_candidate_admission_passed": False,
        },
        "decision": (
            "authorize_observable_level_cutout_psf_and_jacobian_implementation"
            if complete_gate
            else "stop_E325_and_document_receipt_or_array_integrity_failure"
        ),
        "authorization": {
            "construct_registered_observable_cutouts_and_empirical_psf": complete_gate,
            "implement_frozen_image_level_jacobian": complete_gate,
            "reconstruct_MUSE_numerical_kinematics": complete_gate,
            "count_toward_ten_system_target": False,
            "infer_gravity_response": False,
            "fit_new_force_or_action": False,
            "authorize_R2": False,
        },
        "ten_system_effect": {
            "previous_structural_ceiling": 3,
            "updated_structural_ceiling": 3,
            "minimum_new_rank_three_systems_still_required": 7,
        },
        "next_action": "Construct the preregistered registered HST cutouts and PSF/noise products, then run the frozen source-light-marginalized image Jacobian and an independent MUSE numerical-kinematics reconstruction. Do not inspect a gravity residual or count E325 before both gates pass.",
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
