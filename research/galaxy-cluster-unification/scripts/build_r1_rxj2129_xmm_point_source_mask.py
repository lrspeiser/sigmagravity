#!/usr/bin/env python3
"""Generate calibrated PSF r80 values and freeze the RX J2129 source mask."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from astropy.io import fits


PROJECT = Path(__file__).resolve().parents[1]
DERIVED = PROJECT / "data/derived/r1_rxj2129_xmm_x2"
PRE_PSF = DERIVED / "point_source_catalog_pre_psf.csv"
FINAL_CATALOG = DERIVED / "point_source_catalog.csv"
PSF_LEDGER = DERIVED / "point_source_psf_r80.csv"
REGIONS = DERIVED / "point_source_mask_fk5.reg"
MANIFEST_PATH = DERIVED / "point_source_mask_manifest.json"
X2B = Path("/home/henry/.local/share/sigmagravity-xmm/work/rxj2129/0093030201/x2b")
PSF_ROOT = X2B / "psf"

ENERGIES = {
    1: 774.5966692414834,
    2: 1549.1933384829667,
    3: 3741.657386773941,
}
INSTRUMENTS = {
    "EMOS1": "mos1S001-fovimt.fits",
    "EMOS2": "mos2S002-fovimt.fits",
    "EPN": "pnS003-fovimt.fits",
}
BAND_DIRS = {
    1: "detect_band1_500_1200eV",
    2: "detect_band2_1200_2000eV",
    3: "detect_band3_2000_7000eV",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def measure_r80(path: Path) -> dict[str, float]:
    with fits.open(path, memmap=True) as hdus:
        data = np.asarray(hdus[0].data, dtype=float)
        header = hdus[0].header
    finite = np.isfinite(data) & (data >= 0)
    weights = np.where(finite, data, 0.0)
    total = float(weights.sum())
    if data.ndim != 2 or total <= 0:
        raise RuntimeError(f"invalid PSF image {path}")
    yy, xx = np.indices(weights.shape)
    centroid_x = float(np.sum(weights * xx) / total)
    centroid_y = float(np.sum(weights * yy) / total)
    radius_pixels = np.hypot(xx - centroid_x, yy - centroid_y).ravel()
    flat_weights = weights.ravel()
    order = np.argsort(radius_pixels)
    cumulative = np.cumsum(flat_weights[order]) / total
    r80_pixels = float(radius_pixels[order][np.searchsorted(cumulative, 0.8)])
    pixel_x_arcsec = abs(float(header["CDELT1"])) * 3600.0
    pixel_y_arcsec = abs(float(header["CDELT2"])) * 3600.0
    pixel_scale_arcsec = math.sqrt(pixel_x_arcsec * pixel_y_arcsec)
    return {
        "PSF_sum_in_199x199_image": total,
        "intensity_centroid_x_pixel_zero_based": centroid_x,
        "intensity_centroid_y_pixel_zero_based": centroid_y,
        "pixel_scale_arcsec": pixel_scale_arcsec,
        "r80_pixels": r80_pixels,
        "r80_arcsec": r80_pixels * pixel_scale_arcsec,
    }


def main() -> None:
    PSF_ROOT.mkdir(parents=True, exist_ok=True)
    with PRE_PSF.open() as handle:
        sources = list(csv.DictReader(handle))
    psf_rows: list[dict[str, object]] = []
    log_path = PSF_ROOT / "psfgen.log"
    with log_path.open("a") as log:
        for source in sources:
            source_id = int(source["source_id"])
            ra = float(source["ra_deg"])
            dec = float(source["dec_deg"])
            source_root = PSF_ROOT / f"source_{source_id:04d}"
            source_root.mkdir(parents=True, exist_ok=True)
            for band_index, energy in ENERGIES.items():
                for instrument, image_name in INSTRUMENTS.items():
                    image = X2B / BAND_DIRS[band_index] / image_name
                    output = source_root / f"{instrument}_band{band_index}_psf.fits"
                    if not output.is_file():
                        part = output.with_suffix(".fits.part")
                        command = [
                            "psfgen",
                            "withimage=yes",
                            f"image={image}",
                            "withinstrument=yes",
                            f"instrument={instrument}",
                            f"x={ra:.12f}",
                            f"y={dec:.12f}",
                            "coordtype=EQPOS",
                            f"energy={energy:.12f}",
                            "weight=1",
                            "level=ELLBETA",
                            "xsize=199",
                            "ysize=199",
                            f"output={part}",
                        ]
                        log.write("COMMAND " + " ".join(command) + "\n")
                        result = subprocess.run(
                            command,
                            env=os.environ.copy(),
                            text=True,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT,
                            check=False,
                        )
                        log.write(result.stdout)
                        log.write(f"RETURN_CODE {result.returncode}\n")
                        log.flush()
                        if result.returncode != 0 or not part.is_file():
                            raise RuntimeError(
                                f"psfgen failed for source {source_id}, {instrument}, band {band_index}"
                            )
                        part.replace(output)
                    measurement = measure_r80(output)
                    psf_rows.append(
                        {
                            "source_id": source_id,
                            "ra_deg": ra,
                            "dec_deg": dec,
                            "instrument": instrument,
                            "band_index": band_index,
                            "energy_eV": energy,
                            "PSF_path": str(output),
                            "PSF_sha256": sha256(output),
                            **measurement,
                        }
                    )

    with PSF_LEDGER.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(psf_rows[0]))
        writer.writeheader()
        writer.writerows(psf_rows)

    rows_by_source: dict[int, list[dict[str, object]]] = {}
    for row in psf_rows:
        rows_by_source.setdefault(int(row["source_id"]), []).append(row)
    final_rows: list[dict[str, object]] = []
    for source in sources:
        source_id = int(source["source_id"])
        rows = rows_by_source[source_id]
        maximum_r80 = max(float(row["r80_arcsec"]) for row in rows)
        radius = max(15.0, min(60.0, 1.5 * maximum_r80))
        final_rows.append(
            {
                **source,
                "maximum_local_r80_arcsec": maximum_r80,
                "mask_radius_arcsec": radius,
                "PSF_evaluations": len(rows),
                "PSF_mask_status": "frozen",
            }
        )
    with FINAL_CATALOG.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(final_rows[0]))
        writer.writeheader()
        writer.writerows(final_rows)

    with REGIONS.open("w") as handle:
        handle.write("# Region file format: DS9 version 4.1\n")
        handle.write(
            'global color=green dashlist=8 3 width=1 font="helvetica 10 normal roman" select=1 highlite=1 dash=0 fixed=0 edit=0 move=0 delete=0 include=0 source=1\n'
        )
        handle.write("fk5\n")
        for row in final_rows:
            handle.write(
                f'-circle({float(row["ra_deg"]):.10f},{float(row["dec_deg"]):.10f},{float(row["mask_radius_arcsec"]):.6f}\") # text={{XMMPS {row["source_id"]}}}\n'
            )

    manifest = json.loads(MANIFEST_PATH.read_text())
    all_finite = all(
        math.isfinite(float(row["r80_arcsec"])) and float(row["r80_arcsec"]) > 0
        for row in psf_rows
    )
    all_nine = all(len(rows_by_source[int(source["source_id"])]) == 9 for source in sources)
    all_radii = all(15.0 <= float(row["mask_radius_arcsec"]) <= 60.0 for row in final_rows)
    x2b1_pass = (
        manifest["gates"]["all_three_emldetect_catalog_gates_passed"]
        and all_finite
        and all_nine
        and all_radii
    )
    manifest["version"] = "R1B3-RXJ2129-XMM-X2b1-mask-0.3"
    manifest["generated_utc"] = datetime.now(timezone.utc).isoformat()
    manifest["PSF"] = {
        "model": "ELLBETA",
        "source_count": len(sources),
        "evaluations": len(psf_rows),
        "evaluations_per_source": 9,
        "ledger": str(PSF_LEDGER.relative_to(PROJECT)),
        "ledger_sha256": sha256(PSF_LEDGER),
        "minimum_r80_arcsec": min(float(row["r80_arcsec"]) for row in psf_rows),
        "maximum_r80_arcsec": max(float(row["r80_arcsec"]) for row in psf_rows),
    }
    manifest["final_catalog"] = str(FINAL_CATALOG.relative_to(PROJECT))
    manifest["final_catalog_sha256"] = sha256(FINAL_CATALOG)
    manifest["region_mask"] = str(REGIONS.relative_to(PROJECT))
    manifest["region_mask_sha256"] = sha256(REGIONS)
    manifest["mask_radius_arcsec_range"] = [
        min(float(row["mask_radius_arcsec"]) for row in final_rows),
        max(float(row["mask_radius_arcsec"]) for row in final_rows),
    ]
    manifest["gates"].update(
        {
            "all_PSF_radii_completed": all_finite and all_nine,
            "immutable_point_source_mask_frozen": all_radii,
            "X2b1_gate_passed": x2b1_pass,
            "full_X2_gate_passed": False,
        }
    )
    manifest["authorization"] = {
        "run_frozen_PSF_radius_stage": False,
        "run_background_after_mask_gate": x2b1_pass,
        "fit_temperature_or_density": False,
        "fit_new_force_or_action": False,
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))
    if not x2b1_pass:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
