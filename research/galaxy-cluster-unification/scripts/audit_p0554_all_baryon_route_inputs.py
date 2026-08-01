#!/usr/bin/env python3
"""Audit registered HST light and Chandra event coverage without scoring gravity."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.io.fits import Header
from astropy.wcs import WCS


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/p0554_all_baryon_route_input_audit.json"
LABEL_TO_TABLE = {
    "RXJ2129": "RX J2129",
    "MACS0329": "MACS J0329",
    "MACS0429": "MACS J0429",
    "MACS1115": "MACS J1115",
    "MACS1931": "MACS J1931",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().lower()


def event_wcs(header) -> WCS:
    result = Header()
    result["NAXIS"] = 2
    for output, source in {
        "CTYPE1": "TCTYP11",
        "CTYPE2": "TCTYP12",
        "CRVAL1": "TCRVL11",
        "CRVAL2": "TCRVL12",
        "CRPIX1": "TCRPX11",
        "CRPIX2": "TCRPX12",
        "CDELT1": "TCDLT11",
        "CDELT2": "TCDLT12",
    }.items():
        result[output] = header[source]
    result["RADESYS"] = header.get("RADESYS", "ICRS")
    return WCS(result)


def hst_paths(acquisition, reused):
    directory = ROOT / reused["outputs"]["directory"]
    for label in acquisition["hst"]["reused_labels"]:
        system = next(row for row in reused["systems"] if row["label"] == label)
        yield label, directory / system["science"]["filename"], directory / system["weight"]["filename"]
    system = acquisition["hst"]["new_system"]
    directory = ROOT / acquisition["outputs"]["hst_directory"]
    yield system["label"], directory / system["science"]["filename"], directory / system["weight"]["filename"]


def chandra_evt2_paths(acquisition, label):
    system = next(row for row in acquisition["chandra"]["systems"] if row["label"] == label)
    if system.get("source", "").startswith("reused:"):
        directory = ROOT / system["source"].split(":", 1)[1]
        paths = [next((directory / str(obsid) / "primary").glob("*evt2.fits.gz")) for obsid in system["obsids"]]
    else:
        directory = ROOT / acquisition["outputs"]["chandra_directory"] / label
        paths = [next((directory / str(obsid)).glob("*evt2.fits.gz")) for obsid in system["obsids"]]
    return system, paths


def main():
    protocol = json.loads(CONFIG.read_text(encoding="utf-8"))
    if not protocol["status"].startswith("frozen_"):
        raise RuntimeError("all-baryon input audit is not frozen")
    acquisition_path = ROOT / protocol["inputs"]["acquisition_protocol"]
    acquisition = json.loads(acquisition_path.read_text(encoding="utf-8"))
    provenance_path = ROOT / protocol["inputs"]["acquisition_provenance"]
    provenance = json.loads(provenance_path.read_text(encoding="utf-8-sig"))
    reused = json.loads((ROOT / protocol["inputs"]["reused_hst_protocol"]).read_text(encoding="utf-8"))
    images = pd.read_csv(ROOT / protocol["inputs"]["image_catalog"])
    centers = {row["label"]: row for row in acquisition["chandra"]["systems"]}
    recorded = {
        row["local_path"]: row
        for key in ("downloaded_records", "reused_records")
        for row in provenance[key]
    }

    hst_rows = []
    hash_checks = []
    radius_limit = float(protocol["hst_common_coverage_radius_arcsec"])
    for label, science_path, weight_path in hst_paths(acquisition, reused):
        for path in (science_path, weight_path):
            relative = path.relative_to(ROOT).as_posix()
            record = recorded[relative]
            hash_checks.append(
                path.stat().st_size == int(record["size_bytes"])
                and sha256(path) == record["sha256"].lower()
            )
        with fits.open(science_path, memmap=True) as science_hdul, fits.open(weight_path, memmap=True) as weight_hdul:
            science = science_hdul[0].data
            weight = weight_hdul[0].data
            wcs = WCS(science_hdul[0].header)
            wcs.sip = None
            center = centers[label]
            center_x, center_y = wcs.wcs_world2pix([[center["center_ra_deg"], center["center_dec_deg"]]], 0)[0]
            header = science_hdul[0].header
            matrix = np.asarray([[header["CD1_1"], header["CD1_2"]], [header["CD2_1"], header["CD2_2"]]], dtype=float)
            pixel_scale = float(np.sqrt(abs(np.linalg.det(matrix))) * 3600.0)
            cut_pixels = int(np.ceil(radius_limit / pixel_scale))
            x0, x1 = int(center_x) - cut_pixels, int(center_x) + cut_pixels + 1
            y0, y1 = int(center_y) - cut_pixels, int(center_y) + cut_pixels + 1
            yy, xx = np.indices((y1 - y0, x1 - x0), dtype=float)
            rr = np.hypot(xx + x0 - center_x, yy + y0 - center_y) * pixel_scale
            local_weight = np.asarray(weight[y0:y1, x0:x1], dtype=float)
            aperture = rr <= radius_limit
            covered = float(np.mean(np.isfinite(local_weight[aperture]) & (local_weight[aperture] > 0)))
            system_images = images[images.table_cluster.eq(LABEL_TO_TABLE[label])]
            px, py = wcs.wcs_world2pix(system_images[["ra_deg", "dec_deg"]].to_numpy(float), 0).T
            ix, iy = np.rint(px).astype(int), np.rint(py).astype(int)
            in_bounds = (ix >= 0) & (iy >= 0) & (ix < weight.shape[1]) & (iy < weight.shape[0])
            positive = np.zeros(len(ix), dtype=bool)
            positive[in_bounds] = weight[iy[in_bounds], ix[in_bounds]] > 0
            hst_rows.append({
                "system_label": label,
                "shape_y": int(science.shape[0]),
                "shape_x": int(science.shape[1]),
                "pixel_scale_arcsec": pixel_scale,
                "covered_fraction_within_60_arcsec": covered,
                "known_images": int(len(system_images)),
                "known_images_positive_weight": int(positive.sum()),
                "finite_science_pixels": int(np.isfinite(science).sum()),
            })

    chandra_rows = []
    for label in LABEL_TO_TABLE:
        system, paths = chandra_evt2_paths(acquisition, label)
        for path in paths:
            relative = path.relative_to(ROOT).as_posix()
            if label == "RXJ2129":
                rx_provenance = json.loads((ROOT / "data/raw/r1_rxj2129_chandra/provenance.json").read_text(encoding="utf-8-sig"))
                record = next(row for row in rx_provenance["records"] if row["local_path"] == relative)
            else:
                record = recorded[relative]
            hash_checks.append(
                path.stat().st_size == int(record["size_bytes"])
                and sha256(path) == record["sha256"].lower()
            )
            with fits.open(path, memmap=False) as hdul:
                events = hdul["EVENTS"]
                header = events.header
                wcs = event_wcs(header)
                center_x, center_y = wcs.all_world2pix([[system["center_ra_deg"], system["center_dec_deg"]]], 1)[0]
                pixel_arcsec = abs(float(header["TCDLT11"])) * 3600.0
                radius = np.hypot(
                    (events.data["x"].astype(float) - center_x) * pixel_arcsec,
                    (events.data["y"].astype(float) - center_y) * pixel_arcsec,
                )
                energy = events.data["energy"].astype(float) / 1000.0
                lo, hi = protocol["bands_keV"]["soft_morphology"]
                soft = (energy >= lo) & (energy <= hi) & (radius <= float(protocol["audit_aperture_arcsec"]))
                chandra_rows.append({
                    "system_label": label,
                    "obsid": int(header["OBS_ID"]),
                    "exposure_ks": float(header["EXPOSURE"]) / 1000.0,
                    "events": int(len(events.data)),
                    "soft_events_inside_100_arcsec": int(soft.sum()),
                    "pixel_scale_arcsec": pixel_arcsec,
                    "valid_event_columns": {"x", "y", "energy"}.issubset(set(events.columns.names)),
                })

    hst_frame = pd.DataFrame(hst_rows).sort_values("system_label")
    chandra_frame = pd.DataFrame(chandra_rows).sort_values(["system_label", "obsid"])
    gates = protocol["gates"]
    totals = chandra_frame.groupby("system_label").agg(
        exposure_ks=("exposure_ks", "sum"),
        soft_events_inside_100_arcsec=("soft_events_inside_100_arcsec", "sum"),
    )
    checks = {
        "expected_systems": len(hst_frame) == int(gates["expected_systems"]) == len(totals),
        "expected_hst_files": 2 * len(hst_frame) == int(gates["expected_hst_files"]),
        "expected_chandra_evt2_observations": len(chandra_frame) == int(gates["expected_chandra_evt2_observations"]),
        "minimum_total_chandra_exposure": chandra_frame.exposure_ks.sum() >= float(gates["minimum_total_chandra_exposure_ks"]),
        "minimum_each_system_chandra_exposure": bool((totals.exposure_ks >= float(gates["minimum_each_system_chandra_exposure_ks"])).all()),
        "minimum_each_system_soft_events": bool((totals.soft_events_inside_100_arcsec >= int(gates["minimum_each_system_soft_events_inside_aperture"])).all()),
        "hst_common_coverage": bool((hst_frame.covered_fraction_within_60_arcsec >= float(gates["minimum_hst_covered_fraction_inside_common_radius"])).all()),
        "every_known_image_positive_hst_weight": bool((hst_frame.known_images == hst_frame.known_images_positive_weight).all()),
        "all_hashes_verified": bool(all(hash_checks)),
        "valid_event_columns": bool(chandra_frame.valid_event_columns.all()),
        "gas_mass_inferred": False,
        "gravity_or_lens_residual_scored": False,
    }
    checks = {key: bool(value) for key, value in checks.items()}
    positive = [key for key in checks if key not in {"gas_mass_inferred", "gravity_or_lens_residual_scored"}]
    passed = bool(all(checks[key] for key in positive))

    output = ROOT / protocol["outputs"]["directory"]
    output.mkdir(parents=True, exist_ok=True)
    hst_frame.to_csv(output / protocol["outputs"]["hst_ledger"], index=False)
    chandra_frame.to_csv(output / protocol["outputs"]["chandra_ledger"], index=False)
    report = {
        "report_version": "P0554-ALL-BARYON-ROUTE-INPUT-AUDIT-RESULTS-0.1.0",
        "status": "input_adequacy_passed" if passed else "input_adequacy_failed",
        "coverage": {
            "systems": len(hst_frame),
            "hst_files": 2 * len(hst_frame),
            "chandra_evt2_observations": len(chandra_frame),
            "total_chandra_exposure_ks": float(chandra_frame.exposure_ks.sum()),
            "known_lens_images": int(hst_frame.known_images.sum()),
        },
        "system_chandra_totals": totals.reset_index().to_dict("records"),
        "checks": checks,
        "input_adequacy_pass": passed,
        "interpretation": "The acquired maps are adequate for a separately frozen registered morphology test; X-ray brightness remains a gas-location proxy, not a gas-mass measurement.",
        "next_action": "Freeze point-source masking, HST diffuse/member separation, map registration, route construction, and held-out scoring before reading any new gravity score.",
    }
    (output / protocol["outputs"]["report"]).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    axes[0].bar(hst_frame.system_label, 100 * hst_frame.covered_fraction_within_60_arcsec)
    axes[0].set(ylabel="HST positive-weight coverage (%)", ylim=(95, 100.2), title="Registered F160W support")
    axes[0].tick_params(axis="x", rotation=30)
    axes[1].bar(totals.index, totals.soft_events_inside_100_arcsec)
    axes[1].set(ylabel="0.7-2 keV events inside 100 arcsec", title="Chandra morphology counts")
    axes[1].tick_params(axis="x", rotation=30)
    fig.savefig(output / protocol["outputs"]["figure"], dpi=180)
    plt.close(fig)

    summary = f"""# P0554 all-baryon input audit

Input adequacy: **{'pass' if passed else 'fail'}**.

The five systems have registered F160W support at every one of the
{int(hst_frame.known_images.sum())} published lens-image coordinates and
{float(chandra_frame.exposure_ks.sum()):.1f} ks of public Chandra event exposure
across {len(chandra_frame)} observations. No gas mass or gravity score was computed.
"""
    (output / protocol["outputs"]["summary"]).write_text(summary, encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
