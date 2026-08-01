#!/usr/bin/env python3
"""Download frozen Gaia DR3 foreground-star catalogs for P0638."""

from __future__ import annotations

import argparse
import hashlib
import json
import urllib.parse
import urllib.request
from pathlib import Path

import astropy.units as u
import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
from astropy.io import fits

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "p0638_gaia_astrometric_registration.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.config.resolve().read_text(encoding="utf-8"))
    maps = json.loads((ROOT / config["parent_maps"]).read_text(encoding="utf-8"))
    metadata = pd.read_csv(
        ROOT / "results" / "p0637_little_things_photometric_metadata" / "photometric_inputs.csv"
    ).set_index("galaxy")
    output = ROOT / config["raw_directory"]
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    catalog = config["catalog"]
    for target in maps["targets"]:
        galaxy = target["id"]
        row = metadata.loc[galaxy]
        v_path = (
            ROOT
            / maps["raw_directory"]
            / galaxy
            / f"{target['optical_prefix']}v.fits"
        )
        shape = np.squeeze(fits.getdata(v_path, memmap=True)).shape
        radius_arcmin = (
            np.hypot(shape[1], shape[0])
            * float(row["optical_pixel_scale_arcsec"])
            / 120.0
            + float(catalog["query_margin_arcmin"])
        )
        center = SkyCoord(
            str(row["photometric_center_ra_j2000"]),
            str(row["photometric_center_dec_j2000"]),
            unit=(u.hourangle, u.deg),
        )
        parameters = {
            "-source": catalog["vizier_id"],
            "-c": f"{center.ra.deg:.10f} {center.dec.deg:.10f}",
            "-c.rm": f"{radius_arcmin:.5f}",
            "-out": ",".join(catalog["columns"]),
            "Gmag": f"<{float(catalog['maximum_g_magnitude']):g}",
            "-sort": "Gmag",
            "-out.max": str(catalog["maximum_rows"]),
        }
        url = f"{catalog['base_url']}?{urllib.parse.urlencode(parameters)}"
        destination = output / f"{galaxy}_gaia_dr3.tsv"
        if args.force or not destination.exists():
            with urllib.request.urlopen(url, timeout=120) as response:
                payload = response.read()
            if payload.lstrip().lower().startswith(b"<!doctype html"):
                raise RuntimeError(f"VizieR returned HTML for {galaxy}")
            destination.write_bytes(payload)
        rows.append(
            {
                "galaxy": galaxy,
                "url": url,
                "query_radius_arcmin": radius_arcmin,
                "relative_path": destination.relative_to(ROOT).as_posix(),
                "bytes": destination.stat().st_size,
                "sha256": sha256(destination),
            }
        )
        print(f"{galaxy}: Gaia field frozen ({destination.stat().st_size} bytes)")
    (output / "provenance.json").write_text(
        json.dumps(
            {
                "protocol_version": config["protocol_version"],
                "catalog": catalog,
                "targets": rows,
                "sealed_target_observables_opened": False,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
