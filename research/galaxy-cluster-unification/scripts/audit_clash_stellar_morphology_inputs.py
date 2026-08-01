#!/usr/bin/env python3
"""Residual-blind WCS, coverage, and signal audit for CLASH F160W maps."""

from __future__ import annotations

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
from astropy.cosmology import FlatLambdaCDM
from astropy.io import fits
from astropy.wcs import WCS


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/clash_stellar_morphology_acquisition_protocol.json"
OUTPUT = ROOT / "results/clash_stellar_morphology_input_audit"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def member_table(path: Path, system: dict) -> pd.DataFrame:
    rows = []
    cosine = math.cos(math.radians(float(system["center_dec_deg"])))
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        fields = line.split()
        if len(fields) < 7:
            continue
        ra, dec, magnitude = float(fields[1]), float(fields[2]), float(fields[6])
        rows.append(
            {
                "member_id": fields[0],
                "ra_deg": ra,
                "dec_deg": dec,
                "magnitude": magnitude,
                "x_arcsec": (ra - float(system["center_ra_deg"])) * 3600.0 * cosine,
                "y_arcsec": (dec - float(system["center_dec_deg"])) * 3600.0,
            }
        )
    return pd.DataFrame(rows)


def image_table(catalog: pd.DataFrame, system: dict) -> pd.DataFrame:
    rows = catalog[
        catalog.system.eq(system["system"])
        & catalog.metric_neutral_likelihood_row.astype(bool)
    ].copy()
    cosine = math.cos(math.radians(float(system["center_dec_deg"])))
    rows["x_arcsec"] = (
        rows.ra_deg.astype(float) - float(system["center_ra_deg"])
    ) * 3600.0 * cosine
    rows["y_arcsec"] = (
        rows.dec_deg.astype(float) - float(system["center_dec_deg"])
    ) * 3600.0
    return rows


def world_to_pixel(header, ra, dec) -> tuple[np.ndarray, np.ndarray]:
    # CLASH mosaics are already distortion-corrected.  The archive explicitly
    # instructs users to call the low-level linear WCS and not re-apply SIP.
    wcs = WCS(header)
    x, y = wcs.wcs_world2pix(np.asarray(ra), np.asarray(dec), 0)
    return np.asarray(x, dtype=float), np.asarray(y, dtype=float)


def main() -> None:
    protocol = json.loads(CONFIG.read_text(encoding="utf-8"))
    if protocol["status"] != "frozen_before_f160w_download_or_pixel_inspection":
        raise RuntimeError("acquisition protocol is not frozen")
    provenance_path = ROOT / protocol["outputs"]["provenance"]
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    if provenance["protocol_sha256"] != sha256(CONFIG):
        raise RuntimeError("acquisition protocol hash changed")
    file_records = {Path(row["path"]).name: row for row in provenance["files"]}
    catalog = pd.read_csv(ROOT / protocol["shared_inputs"]["image_catalog"])
    cosmo = FlatLambdaCDM(H0=70.0, Om0=0.27)

    OUTPUT.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(2, 2, figsize=(13, 12), constrained_layout=True)
    report_rows = []
    input_hashes = {
        "protocol": sha256(CONFIG),
        "provenance": sha256(provenance_path),
        "image_catalog": sha256(ROOT / protocol["shared_inputs"]["image_catalog"]),
    }
    for axis, system in zip(axes.flat, protocol["systems"]):
        science_path = ROOT / protocol["outputs"]["directory"] / system["science"]["filename"]
        weight_path = ROOT / protocol["outputs"]["directory"] / system["weight"]["filename"]
        for path in (science_path, weight_path):
            record = file_records[path.name]
            if path.stat().st_size != int(record["bytes"]) or sha256(path) != record["sha256"]:
                raise RuntimeError(f"download integrity failure: {path}")
            input_hashes[path.name] = record["sha256"]

        with fits.open(science_path, memmap=True) as science_hdul, fits.open(
            weight_path, memmap=True
        ) as weight_hdul:
            header = science_hdul[0].header
            data = science_hdul[0].data
            weight = weight_hdul[0].data
            if data.shape != weight.shape or data.shape != (5000, 5000):
                raise RuntimeError(f"unexpected mosaic shape for {system['label']}")
            center_x, center_y = world_to_pixel(
                header, [system["center_ra_deg"]], [system["center_dec_deg"]]
            )
            center_x, center_y = float(center_x[0]), float(center_y[0])
            cd = np.asarray(
                [
                    [float(header["CD1_1"]), float(header["CD1_2"])],
                    [float(header["CD2_1"]), float(header["CD2_2"])],
                ]
            )
            pixel_scale_arcsec = float(np.sqrt(abs(np.linalg.det(cd))) * 3600.0)
            half_width_arcsec = 90.0
            half_width_pixel = int(np.ceil(half_width_arcsec / pixel_scale_arcsec))
            x0, x1 = int(round(center_x)) - half_width_pixel, int(round(center_x)) + half_width_pixel + 1
            y0, y1 = int(round(center_y)) - half_width_pixel, int(round(center_y)) + half_width_pixel + 1
            cut = np.asarray(data[y0:y1, x0:x1], dtype=np.float32)
            cut_weight = np.asarray(weight[y0:y1, x0:x1], dtype=np.float32)
            yy, xx = np.indices(cut.shape, dtype=float)
            x_arcsec = (xx + x0 - center_x) * pixel_scale_arcsec
            y_arcsec = (yy + y0 - center_y) * pixel_scale_arcsec
            radius = np.hypot(x_arcsec, y_arcsec)
            outer = (radius >= 75.0) & (radius <= 90.0) & (cut_weight > 0.0) & np.isfinite(cut)
            background = float(np.median(cut[outer]))
            outer_mad = float(1.4826 * np.median(np.abs(cut[outer] - background)))
            valid = (radius <= 90.0) & (cut_weight > 0.0) & np.isfinite(cut)

            images = image_table(catalog, system)
            members = member_table(ROOT / system["member_catalog"], system)
            image_px, image_py = world_to_pixel(
                header, images.ra_deg.to_numpy(float), images.dec_deg.to_numpy(float)
            )
            member_px, member_py = world_to_pixel(
                header, members.ra_deg.to_numpy(float), members.dec_deg.to_numpy(float)
            )
            image_ix = np.rint(image_px).astype(int)
            image_iy = np.rint(image_py).astype(int)
            member_ix = np.rint(member_px).astype(int)
            member_iy = np.rint(member_py).astype(int)
            image_covered = (
                (image_ix >= 0)
                & (image_ix < data.shape[1])
                & (image_iy >= 0)
                & (image_iy < data.shape[0])
            )
            member_covered = (
                (member_ix >= 0)
                & (member_ix < data.shape[1])
                & (member_iy >= 0)
                & (member_iy < data.shape[0])
            )
            image_weight_positive = image_covered.copy()
            member_weight_positive = member_covered.copy()
            image_weight_positive[image_covered] = weight[
                image_iy[image_covered], image_ix[image_covered]
            ] > 0.0
            member_weight_positive[member_covered] = weight[
                member_iy[member_covered], member_ix[member_covered]
            ] > 0.0

            display = np.arcsinh(np.maximum(cut - background, 0.0) / max(outer_mad, 1.0e-12))
            finite_display = display[np.isfinite(display) & valid]
            vmax = float(np.quantile(finite_display, 0.997))
            axis.imshow(
                display,
                origin="lower",
                cmap="gray",
                vmin=0.0,
                vmax=vmax,
                extent=[-half_width_arcsec, half_width_arcsec, -half_width_arcsec, half_width_arcsec],
            )
            axis.scatter(
                members.x_arcsec,
                members.y_arcsec,
                s=13,
                facecolors="none",
                edgecolors="#3ddc97",
                linewidths=0.55,
                label="Caminha members",
            )
            axis.scatter(
                images.x_arcsec,
                images.y_arcsec,
                marker="x",
                s=28,
                c="#ff5c5c",
                linewidths=0.9,
                label="all lens images (to mask)",
            )
            axis.set(
                title=f"{system['label']}  F160W",
                xlabel="RA offset (arcsec)",
                ylabel="Dec offset (arcsec)",
                xlim=(90, -90),
                ylim=(-90, 90),
            )
            axis.legend(loc="upper right", fontsize=7)
            scale = float(cosmo.kpc_proper_per_arcmin(system["lens_redshift"]).value / 60.0)
            report_rows.append(
                {
                    "label": system["label"],
                    "shape": list(data.shape),
                    "pixel_scale_arcsec": pixel_scale_arcsec,
                    "center_x_pixel_zero_based": center_x,
                    "center_y_pixel_zero_based": center_y,
                    "center_margin_arcsec": min(
                        center_x,
                        center_y,
                        data.shape[1] - 1 - center_x,
                        data.shape[0] - 1 - center_y,
                    )
                    * pixel_scale_arcsec,
                    "angular_scale_kpc_per_arcsec": scale,
                    "covered_fraction_within_90_arcsec": float(np.mean(valid[radius <= 90.0])),
                    "covered_fraction_within_60_arcsec": float(
                        np.mean((cut_weight[radius <= 60.0] > 0.0) & np.isfinite(cut[radius <= 60.0]))
                    ),
                    "background_electrons_per_s": background,
                    "outer_annulus_robust_sigma_electrons_per_s": outer_mad,
                    "known_images": len(images),
                    "known_images_positive_weight": int(np.sum(image_weight_positive)),
                    "maximum_image_radius_arcsec": float(np.max(np.hypot(images.x_arcsec, images.y_arcsec))),
                    "members": len(members),
                    "members_positive_weight": int(np.sum(member_weight_positive)),
                    "maximum_member_radius_arcsec": float(np.max(np.hypot(members.x_arcsec, members.y_arcsec))),
                }
            )

    figure_path = OUTPUT / "f160w_input_audit.png"
    figure.savefig(figure_path, dpi=170)
    plt.close(figure)
    all_pass = all(
        row["known_images_positive_weight"] == row["known_images"]
        and row["members_positive_weight"] == row["members"]
        and row["covered_fraction_within_60_arcsec"] >= 0.999
        for row in report_rows
    )
    report = {
        "report_version": "CLASH-STELLAR-MORPHOLOGY-INPUT-AUDIT-0.1.0",
        "selection_blind": True,
        "lens_residuals_inspected": False,
        "gravity_scores_computed": False,
        "input_hashes": input_hashes,
        "systems": report_rows,
        "common_geometry_passed": all_pass,
        "common_supported_radius_arcsec": 60.0 if all_pass else None,
        "outer_footprint_note": "The WFC3/IR footprint becomes incomplete outside about 65 arcsec; the declared 60-arcsec common support contains every raw-lensing image.",
        "figure": str(figure_path.relative_to(ROOT)).replace("\\", "/"),
    }
    (OUTPUT / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    pd.DataFrame(report_rows).to_csv(OUTPUT / "systems.csv", index=False)
    print(json.dumps({"common_geometry_passed": all_pass, "systems": report_rows}, indent=2))


if __name__ == "__main__":
    main()
