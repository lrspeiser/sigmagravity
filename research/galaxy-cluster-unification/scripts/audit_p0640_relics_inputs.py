#!/usr/bin/env python3
"""Validate open RELICS baryons while keeping lensing payloads opaque."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.utils.exceptions import AstropyWarning
from astropy.wcs import WCS

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from voidscreen.gravity_arc_tomography import (
    combine_f160_photometry,
    photometric_membership_weights,
    read_relics_catalog,
)

DEFAULT_CONFIG = ROOT / "configs" / "p0640_relics_baryon_and_sealed_lensing_acquisition.json"
DEFAULT_PROVENANCE = ROOT / "results" / "p0640_relics_input_acquisition" / "provenance.json"
DEFAULT_OUTPUT = ROOT / "results" / "p0640_relics_input_audit"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def one(directory: Path, pattern: str) -> Path:
    matches = list(directory.glob(pattern))
    if len(matches) != 1:
        raise RuntimeError(f"expected one {pattern} in {directory}, found {len(matches)}")
    return matches[0]


def audit_target(target: dict, raw: Path) -> tuple[dict, list[dict]]:
    directory = raw / target["id"]
    hst = directory / "hst"
    catalog_path = one(hst, "*_cat.txt")
    image_path = one(hst, "*_f160w_v1_drz.fits")
    segmentation_path = one(hst, "*_segm.fits")
    catalog = read_relics_catalog(catalog_path)
    flux, significance = combine_f160_photometry(catalog)
    hard, soft = photometric_membership_weights(catalog, float(target["redshift"]))
    with fits.open(image_path, memmap=True) as image_hdul, fits.open(
        segmentation_path, memmap=True
    ) as segmentation_hdul:
        image = np.asarray(image_hdul[0].data)
        segmentation = np.asarray(segmentation_hdul[0].data)
        image_header = image_hdul[0].header
        segmentation_header = segmentation_hdul[0].header
        if image.shape != segmentation.shape or image.ndim != 2:
            raise RuntimeError(f"{target['id']}: HST image and segmentation geometry differ")
        if not WCS(image_header).has_celestial or not WCS(segmentation_header).has_celestial:
            raise RuntimeError(f"{target['id']}: HST product lacks celestial WCS")
        x = np.rint(catalog["x"].to_numpy(float)).astype(int) - 1
        y = np.rint(catalog["y"].to_numpy(float)).astype(int) - 1
        inside = (
            (x >= 0)
            & (x < segmentation.shape[1])
            & (y >= 0)
            & (y < segmentation.shape[0])
        )
        sampled = segmentation[y[inside], x[inside]]
        segmentation_match = float(
            np.mean(sampled == catalog["id"].to_numpy(int)[inside])
        )
        finite_fraction = float(np.mean(np.isfinite(image)))
        infinite_pixels = int(np.count_nonzero(np.isinf(image)))
        positive_segmentation_pixels = int(np.count_nonzero(segmentation))

    chandra_rows = []
    expected_by_name = {
        filename: (int(obsid), int(expected), float(exposure))
        for obsid, filename, expected, exposure in target["chandra"]
    }
    paths = sorted((directory / "chandra").glob("*.fits.gz"))
    if {path.name for path in paths} != set(expected_by_name):
        raise RuntimeError(f"{target['id']}: Chandra file inventory differs from frozen config")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", AstropyWarning)
        for path in paths:
            obsid, _expected, expected_exposure = expected_by_name[path.name]
            with fits.open(path, memmap=True) as hdul:
                data = np.asarray(hdul[0].data)
                header = hdul[0].header
                exposure = float(header["EXPOSURE"])
                valid = bool(
                    data.ndim == 2
                    and np.issubdtype(data.dtype, np.integer)
                    and np.min(data) >= 0
                    and np.sum(data) > 0
                    and WCS(header).has_celestial
                    and abs(exposure / expected_exposure - 1.0) < 1.0e-10
                )
                chandra_rows.append(
                    {
                        "system": target["id"],
                        "obsid": obsid,
                        "filename": path.name,
                        "shape": "x".join(str(value) for value in data.shape),
                        "exposure_s": exposure,
                        "counts": int(np.sum(data)),
                        "positive_pixels": int(np.count_nonzero(data)),
                        "valid": valid,
                    }
                )
    row = {
        "system": target["id"],
        "redshift": float(target["redshift"]),
        "catalog_rows": len(catalog),
        "f160w_5sigma_detections": int(np.sum(np.isfinite(flux) & (significance >= 5.0))),
        "hard_photoz_members": int(np.sum(hard)),
        "hard_photoz_f160w_5sigma_members": int(np.sum(hard & (significance >= 5.0))),
        "soft_membership_sum": float(np.sum(soft)),
        "hst_shape": "x".join(str(value) for value in image.shape),
        "hst_finite_fraction": finite_fraction,
        "hst_infinite_pixels": infinite_pixels,
        "segmentation_catalog_match_fraction": segmentation_match,
        "segmentation_positive_pixels": positive_segmentation_pixels,
        "chandra_observations": len(chandra_rows),
        "chandra_exposure_ks": float(sum(item["exposure_s"] for item in chandra_rows) / 1000.0),
        "chandra_counts": int(sum(item["counts"] for item in chandra_rows)),
        "all_chandra_images_valid": bool(all(item["valid"] for item in chandra_rows)),
    }
    return row, chandra_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--provenance", type=Path, default=DEFAULT_PROVENANCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    config_path = args.config.resolve()
    provenance_path = args.provenance.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    if provenance["config_sha256"] != sha256(config_path):
        raise RuntimeError("P0640 config changed after acquisition")
    records = provenance["records"]
    errors = []
    for record in records:
        path = ROOT / record["relative_path"]
        if not path.is_file() or path.stat().st_size != int(record["bytes"]):
            errors.append(f"missing or size-changed file: {record['relative_path']}")
        elif sha256(path) != record["sha256"]:
            errors.append(f"hash-changed file: {record['relative_path']}")

    raw = ROOT / config["raw_directory"]
    raw_names = " ".join(path.name.lower() for path in raw.rglob("*") if path.is_file())
    for fragment in config["forbidden_raw_baryon_fragments"]:
        if fragment.lower() in raw_names:
            errors.append(f"forbidden lensing artifact in open baryon directory: {fragment}")

    target_rows = []
    chandra_rows = []
    for target in config["targets"]:
        row, xray = audit_target(target, raw)
        target_rows.append(row)
        chandra_rows.extend(xray)
    targets = pd.DataFrame(target_rows)
    xray = pd.DataFrame(chandra_rows)
    sealed_records = [row for row in records if row["kind"] == "sealed_constraint_container"]
    sealed_directory = ROOT / config["sealed_directory"]
    expected_sealed = {Path(row["relative_path"]).name for row in sealed_records}
    actual_sealed = {path.name for path in sealed_directory.iterdir() if path.is_file()}
    if actual_sealed != expected_sealed:
        errors.append("sealed directory contains an unexpected or missing artifact")
    if any(config["sealed_state"].values()):
        errors.append("sealed-state declaration is not fully false")

    gates = {
        "four_preregistered_clusters": len(targets) == 4,
        "all_34_artifacts_hashed": len(records) == 34 and not errors,
        "real_f160w_footprint_at_least_15_percent": bool(
            (targets["hst_finite_fraction"] >= 0.15).all()
            and (targets["hst_infinite_pixels"] == 0).all()
        ),
        "segmentation_matches_catalog": bool(
            (targets["segmentation_catalog_match_fraction"] >= 0.995).all()
        ),
        "minimum_200_hard_photoz_members": bool((targets["hard_photoz_members"] >= 200).all()),
        "minimum_90_f160w_detected_hard_members": bool(
            (targets["hard_photoz_f160w_5sigma_members"] >= 90).all()
        ),
        "all_19_chandra_maps_valid": len(xray) == 19 and bool(xray["valid"].all()),
        "minimum_70ks_chandra_per_cluster": bool(
            (targets["chandra_exposure_ks"] >= 70.0).all()
        ),
        "two_lensing_sources_sealed_opaque": len(sealed_records) == 2
        and not any(config["sealed_state"].values()),
        "no_derived_lens_map_in_baryon_inputs": not any(
            fragment.lower() in raw_names
            for fragment in config["forbidden_raw_baryon_fragments"]
        ),
    }
    status = "ready" if all(gates.values()) and not errors else "input_failure"
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    targets.to_csv(output / "systems.csv", index=False)
    xray.to_csv(output / "chandra_observations.csv", index=False)
    report = {
        "report_version": config["protocol_version"],
        "status": status,
        "config_sha256": sha256(config_path),
        "provenance_sha256": sha256(provenance_path),
        "gates": gates,
        "errors": errors,
        "totals": {
            "systems": len(targets),
            "artifacts": len(records),
            "bytes": int(sum(row["bytes"] for row in records)),
            "catalog_objects": int(targets["catalog_rows"].sum()),
            "hard_photoz_members": int(targets["hard_photoz_members"].sum()),
            "chandra_observations": len(xray),
            "chandra_exposure_ks": float(targets["chandra_exposure_ks"].sum()),
            "sealed_constraint_containers": len(sealed_records),
        },
        "sealed_state": config["sealed_state"],
        "systems": target_rows,
    }
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    summary = f"""# P0640 RELICS baryon and sealed-lensing input audit

- Status: **{status.upper()}**
- Untouched clusters: {len(targets)}
- Hashed artifacts: {len(records)} ({report['totals']['bytes']:,} bytes)
- HST catalog objects: {report['totals']['catalog_objects']:,}
- Strict photometric members: {report['totals']['hard_photoz_members']:,}
- Chandra observations: {len(xray)} ({report['totals']['chandra_exposure_ks']:.1f} ks)
- Opaque sealed constraint containers: {len(sealed_records)}
- Derived lens maps downloaded: `false`
- Sealed constraint contents opened: `false`

The open inputs are real HST F160W pixels, their matched source segmentation,
photometric member catalogs, and Chandra level-2 count maps. The raw
multiple-image tables are present only as byte-counted, SHA-256-hashed sealed
artifacts; this audit never parses or extracts them.
"""
    (output / "SUMMARY.md").write_text(summary, encoding="utf-8")
    print(json.dumps({"status": status, "gates": gates, "totals": report["totals"]}, indent=2))
    if status != "ready":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
