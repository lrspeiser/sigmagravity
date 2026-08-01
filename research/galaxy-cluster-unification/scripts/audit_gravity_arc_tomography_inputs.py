#!/usr/bin/env python3
"""Audit RELICS galaxy catalogs against the existing lens-map footprints."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.wcs import WCS


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from voidscreen.gravity_arc_tomography import (
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


def unique_columns(catalog: pd.DataFrame, prefix: str) -> int:
    return sum(name == prefix or name.startswith(prefix + "__") for name in catalog)


def main() -> None:
    acquisition_path = ROOT / "configs/gravity_arc_tomography_acquisition.json"
    acquisition = json.loads(acquisition_path.read_text(encoding="utf-8"))
    provenance_path = ROOT / acquisition["outputs"]["provenance"]
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    if provenance["protocol_sha256"] != sha256(acquisition_path):
        raise RuntimeError("acquisition protocol changed after download")
    provenance_by_label = {row["label"]: row for row in provenance["files"]}
    output = ROOT / "results/gravity_arc_tomography_input_audit"
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    source_rows = []
    figure, axes = plt.subplots(1, 3, figsize=(17, 5.5), constrained_layout=True)
    directory = ROOT / acquisition["outputs"]["directory"]
    for axis, system in zip(axes, acquisition["systems"], strict=True):
        catalog_path = directory / system["catalog_filename"]
        if sha256(catalog_path) != provenance_by_label[system["label"]]["sha256"]:
            raise RuntimeError(f"catalog hash changed for {system['label']}")
        catalog = read_relics_catalog(catalog_path)
        flux, significance = combine_f160_photometry(catalog)
        hard, soft = photometric_membership_weights(catalog, system["cluster_redshift"])
        range_paths = sorted((ROOT / system["lensing_directory"] / "range").glob("*_kappa.fits"))
        if len(range_paths) != 100:
            raise RuntimeError(f"{system['label']}: expected 100 range maps")
        header = fits.getheader(range_paths[0])
        wcs = WCS(header)
        x, y = wcs.world_to_pixel_values(
            catalog["RA"].to_numpy(float), catalog["Dec"].to_numpy(float)
        )
        inside = (
            (x >= 0.0)
            & (x <= float(header["NAXIS1"] - 1))
            & (y >= 0.0)
            & (y <= float(header["NAXIS2"] - 1))
        )
        galaxy = catalog["stel"].to_numpy(float) < 0.8
        detected = np.isfinite(flux) & np.isfinite(significance) & (significance >= 5.0)
        usable = inside & galaxy & detected
        hard_usable = usable & hard
        soft_flux = np.where(usable, flux * soft, 0.0)
        for index in np.flatnonzero(usable):
            source_rows.append(
                {
                    "system": system["label"],
                    "id": int(catalog.iloc[index]["id"]),
                    "ra_deg": float(catalog.iloc[index]["RA"]),
                    "dec_deg": float(catalog.iloc[index]["Dec"]),
                    "map_x_pixel": float(x[index]),
                    "map_y_pixel": float(y[index]),
                    "f160w_flux_nJy": float(flux[index]),
                    "f160w_significance": float(significance[index]),
                    "stellarity": float(catalog.iloc[index]["stel"]),
                    "zb": float(catalog.iloc[index]["zb"]),
                    "zbmin": float(catalog.iloc[index]["zbmin"]),
                    "zbmax": float(catalog.iloc[index]["zbmax"]),
                    "odds": float(catalog.iloc[index]["odds"]),
                    "hard_member": bool(hard[index]),
                    "soft_membership_weight": float(soft[index]),
                    "soft_f160w_weight": float(soft_flux[index]),
                }
            )
        rows.append(
            {
                "system": system["label"],
                "catalog_rows": int(len(catalog)),
                "catalog_columns": int(len(catalog.columns)),
                "f160_measurements": unique_columns(catalog, "f160w_fluxnJy"),
                "inside_lens_map": int(np.sum(inside)),
                "usable_f160_galaxies": int(np.sum(usable)),
                "hard_photoz_members": int(np.sum(hard_usable)),
                "soft_effective_source_count": float(
                    np.square(np.sum(soft_flux))
                    / max(np.sum(np.square(soft_flux)), np.finfo(float).tiny)
                ),
                "soft_member_light_fraction": float(
                    np.sum(soft_flux) / max(np.sum(np.where(usable, flux, 0.0)), np.finfo(float).tiny)
                ),
                "range_maps": len(range_paths),
                "map_shape_y": int(header["NAXIS2"]),
                "map_shape_x": int(header["NAXIS1"]),
                "map_pixel_scale_arcsec": abs(float(header["CDELT1"])) * 3600.0,
            }
        )
        point_size = 4.0 + 35.0 * np.sqrt(flux[usable] / np.nanmax(flux[usable]))
        axis.scatter(x[usable], y[usable], s=point_size, c=soft[usable], cmap="viridis", alpha=0.65)
        axis.scatter(x[hard_usable], y[hard_usable], s=12, facecolors="none", edgecolors="red", linewidths=0.5)
        axis.set(
            title=f"{system['label']}\nusable F160W sources; red = hard photo-z",
            xlabel="lens-map x pixel",
            ylabel="lens-map y pixel",
            xlim=(0, int(header["NAXIS1"]) - 1),
            ylim=(0, int(header["NAXIS2"]) - 1),
            aspect="equal",
        )
    systems = pd.DataFrame(rows)
    sources = pd.DataFrame(source_rows)
    systems.to_csv(output / "systems.csv", index=False)
    sources.to_csv(output / "sources.csv", index=False)
    figure.savefig(output / "source_footprints.png", dpi=180)
    plt.close(figure)
    report = {
        "status": "completed_without_inspecting_catalog_to_kappa_spatial_correlation",
        "protocol_sha256": sha256(acquisition_path),
        "provenance_sha256": sha256(provenance_path),
        "systems": rows,
        "totals": {
            "catalog_rows": int(systems.catalog_rows.sum()),
            "usable_f160_galaxies": int(systems.usable_f160_galaxies.sum()),
            "hard_photoz_members": int(systems.hard_photoz_members.sum()),
            "lensing_realizations": int(systems.range_maps.sum()),
        },
        "source_rule": {
            "base": "inside range-map WCS, stellarity < 0.8, positive F160W flux at >=5 sigma",
            "hard_member": "cluster redshift inside BPZ 95% interval and ODDS >= 0.5",
            "soft_weight": "F160W flux times ODDS times a Gaussian redshift-overlap weight; this is a deterministic tracer weight, not a calibrated membership probability"
        }
    }
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
