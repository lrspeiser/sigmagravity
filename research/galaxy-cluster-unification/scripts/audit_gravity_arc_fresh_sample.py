#!/usr/bin/env python3
"""Audit fresh RELICS catalogs and map geometry without viewing kappa morphology."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
from astropy.cosmology import Planck18
from astropy.io import fits
from astropy.wcs import WCS
import astropy.units as u


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from voidscreen.gravity_arc_tomography import (  # noqa: E402
    combine_f160_photometry,
    photometric_membership_weights,
    read_relics_catalog,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def aperture_coverage(
    header: fits.Header,
    center: SkyCoord,
    *,
    kpc_per_arcsec: float,
    radius_kpc: float,
    spacing_kpc: float,
) -> float:
    axis = np.arange(-radius_kpc, radius_kpc + spacing_kpc, spacing_kpc)
    x_grid, y_grid = np.meshgrid(axis, axis, indexing="xy")
    aperture = np.hypot(x_grid, y_grid) <= radius_kpc
    world = center.spherical_offsets_by(
        (x_grid / kpc_per_arcsec) * u.arcsec,
        (y_grid / kpc_per_arcsec) * u.arcsec,
    )
    pixel_x, pixel_y = WCS(header).world_to_pixel(world)
    inside = (
        np.isfinite(pixel_x)
        & np.isfinite(pixel_y)
        & (pixel_x >= 0.0)
        & (pixel_y >= 0.0)
        & (pixel_x <= float(header["NAXIS1"] - 1))
        & (pixel_y <= float(header["NAXIS2"] - 1))
    )
    return float(np.mean(inside[aperture]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="configs/gravity_arc_fresh_sample_protocol.json"
    )
    args = parser.parse_args()
    config_path = ROOT / args.config
    protocol = json.loads(config_path.read_text(encoding="utf-8"))
    if protocol["status"] != "frozen_before_download_or_fresh_map_spatial_inspection":
        raise RuntimeError("fresh-sample protocol is not frozen")
    acquisition = protocol["acquisition"]
    provenance_path = ROOT / acquisition["provenance"]
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    if provenance["protocol_sha256"] != sha256(config_path):
        raise RuntimeError("protocol changed after fresh-sample acquisition")
    manifest_path = ROOT / acquisition["manifest"]
    if provenance["manifest_sha256"] != sha256(manifest_path):
        raise RuntimeError("fresh-sample manifest changed after acquisition")
    manifest = load_manifest(manifest_path)
    manifest_by_path = {row["path"]: row for row in manifest}

    output = ROOT / protocol["outputs"]["input_audit_directory"]
    output.mkdir(parents=True, exist_ok=True)
    raw = ROOT / acquisition["output_directory"]
    settings = protocol["spatial_preprocessing"]
    radius_limit = float(settings["common_radius_kpc"])
    rows = []
    source_rows = []
    columns = min(5, len(protocol["systems"]))
    rows_count = int(math.ceil(len(protocol["systems"]) / columns))
    figure, axes = plt.subplots(
        rows_count,
        columns,
        figsize=(4 * columns, 4 * rows_count),
        constrained_layout=True,
        squeeze=False,
    )
    flat_axes = axes.ravel()
    for axis, system in zip(flat_axes, protocol["systems"], strict=False):
        catalog_path = raw / "catalogs" / system["catalog_filename"]
        relative_catalog = str(catalog_path.relative_to(ROOT)).replace("\\", "/")
        if sha256(catalog_path) != manifest_by_path[relative_catalog]["sha256"]:
            raise RuntimeError(f"catalog hash changed for {system['label']}")
        catalog = read_relics_catalog(catalog_path)
        flux, significance = combine_f160_photometry(catalog)
        hard, soft = photometric_membership_weights(catalog, float(system["cluster_redshift"]))

        lenstool = next(item for item in system["models"] if item["method"] == "lenstool")
        reference_path = raw / "models" / system["slug"] / "lenstool" / lenstool["best_filename"]
        reference_header = fits.getheader(reference_path)
        center = SkyCoord(
            float(reference_header["CRVAL1"]) * u.deg,
            float(reference_header["CRVAL2"]) * u.deg,
            frame="icrs",
        )
        coordinates = SkyCoord(
            catalog["RA"].to_numpy(float) * u.deg,
            catalog["Dec"].to_numpy(float) * u.deg,
            frame="icrs",
        )
        offset_x, offset_y = center.spherical_offsets_to(coordinates)
        kpc_per_arcsec = float(
            Planck18.kpc_proper_per_arcmin(float(system["cluster_redshift"])).value / 60.0
        )
        x_kpc = offset_x.to_value(u.arcsec) * kpc_per_arcsec
        y_kpc = offset_y.to_value(u.arcsec) * kpc_per_arcsec
        radius_kpc = np.hypot(x_kpc, y_kpc)
        galaxy = catalog["stel"].to_numpy(float) < 0.8
        detected = np.isfinite(flux) & np.isfinite(significance) & (significance >= 5.0)
        usable = galaxy & detected & (radius_kpc <= radius_limit)
        hard_usable = usable & hard
        if np.sum(hard_usable) < 3:
            raise RuntimeError(f"{system['label']}: fewer than three hard-member sources")

        for index in np.flatnonzero(usable):
            source_rows.append(
                {
                    "system": system["label"],
                    "id": int(catalog.iloc[index]["id"]),
                    "ra_deg": float(catalog.iloc[index]["RA"]),
                    "dec_deg": float(catalog.iloc[index]["Dec"]),
                    "x_kpc": float(x_kpc[index]),
                    "y_kpc": float(y_kpc[index]),
                    "radius_kpc": float(radius_kpc[index]),
                    "f160w_flux_nJy": float(flux[index]),
                    "f160w_significance": float(significance[index]),
                    "stellarity": float(catalog.iloc[index]["stel"]),
                    "zb": float(catalog.iloc[index]["zb"]),
                    "zbmin": float(catalog.iloc[index]["zbmin"]),
                    "zbmax": float(catalog.iloc[index]["zbmax"]),
                    "odds": float(catalog.iloc[index]["odds"]),
                    "hard_member": bool(hard[index]),
                    "soft_membership_weight": float(soft[index]),
                }
            )

        method_coverage = {}
        for model in system["models"]:
            best_path = raw / "models" / system["slug"] / model["method"] / model["best_filename"]
            header = fits.getheader(best_path)
            method_coverage[model["method"]] = aperture_coverage(
                header,
                center,
                kpc_per_arcsec=kpc_per_arcsec,
                radius_kpc=radius_limit,
                spacing_kpc=float(settings["grid_spacing_kpc"]),
            )
            range_paths = sorted((best_path.parent / "range").glob("*_kappa.fits"))
            if len(range_paths) != int(model["range_count"]):
                raise RuntimeError(
                    f"{system['label']} {model['method']}: expected {model['range_count']} "
                    f"range maps, found {len(range_paths)}"
                )
        rows.append(
            {
                "system": system["label"],
                "cluster_redshift": float(system["cluster_redshift"]),
                "reference_ra_deg": float(center.ra.deg),
                "reference_dec_deg": float(center.dec.deg),
                "catalog_rows": int(len(catalog)),
                "usable_f160_galaxies_300kpc": int(np.sum(usable)),
                "hard_photoz_members_300kpc": int(np.sum(hard_usable)),
                "hard_effective_source_count": float(
                    np.square(np.sum(flux[hard_usable]))
                    / max(np.sum(np.square(flux[hard_usable])), np.finfo(float).tiny)
                ),
                "lenstool_aperture_coverage": method_coverage["lenstool"],
                "glafic_aperture_coverage": method_coverage["glafic"],
            }
        )
        size = 8.0 + 80.0 * np.sqrt(flux[hard_usable] / np.max(flux[hard_usable]))
        axis.scatter(x_kpc[usable], y_kpc[usable], s=3, color="0.75", alpha=0.5)
        axis.scatter(x_kpc[hard_usable], y_kpc[hard_usable], s=size, color="tab:blue", alpha=0.7)
        axis.set(
            title=f"{system['label']}\n{np.sum(hard_usable)} strict members",
            xlim=(-radius_limit, radius_limit),
            ylim=(-radius_limit, radius_limit),
            aspect="equal",
            xlabel="east offset (kpc)",
            ylabel="north offset (kpc)",
        )
    for axis in flat_axes[len(protocol["systems"]) :]:
        axis.set_visible(False)

    systems = pd.DataFrame(rows)
    sources = pd.DataFrame(source_rows)
    systems.to_csv(output / "systems.csv", index=False)
    sources.to_csv(output / "sources.csv", index=False)
    figure.savefig(output / "source_footprints.png", dpi=160)
    plt.close(figure)
    minimum_coverage = float(settings["minimum_finite_aperture_fraction"])
    report = {
        "report_version": protocol["protocol_version"],
        "status": "completed_without_inspecting_fresh_kappa_pixel_values",
        "protocol_sha256": sha256(config_path),
        "provenance_sha256": sha256(provenance_path),
        "coverage_gate_passed": bool(
            (systems.lenstool_aperture_coverage >= minimum_coverage).all()
            and (systems.glafic_aperture_coverage >= minimum_coverage).all()
        ),
        "systems": rows,
        "totals": {
            "systems": int(len(systems)),
            "catalog_rows": int(systems.catalog_rows.sum()),
            "usable_f160_galaxies_300kpc": int(systems.usable_f160_galaxies_300kpc.sum()),
            "hard_photoz_members_300kpc": int(systems.hard_photoz_members_300kpc.sum()),
            "lenstool_range_maps": int(
                sum(
                    model["range_count"]
                    for system in protocol["systems"]
                    for model in system["models"]
                    if model["method"] == "lenstool"
                )
            ),
            "glafic_best_maps": int(
                sum(
                    any(model["method"] == "glafic" for model in system["models"])
                    for system in protocol["systems"]
                )
            ),
        },
        "source_rule": settings["source_rule"],
        "reference_coordinate_rule": settings["reference_coordinate"],
    }
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
