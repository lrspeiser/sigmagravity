#!/usr/bin/env python3
"""Download and hash the frozen HI4PI Galactic-column queries for Sigma v17C."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import re
import urllib.parse
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REGIONS = ROOT / "results" / "sigma_v17b_temperature_regions" / "report.json"
DEFAULT_RAW = ROOT / "data" / "raw" / "sigma_v17_hi4pi"
DEFAULT_OUTPUT = ROOT / "results" / "sigma_v17c_hi4pi_acquisition"
ENDPOINT = "https://heasarc.gsfc.nasa.gov/cgi-bin/Tools/w3nh/w3nh.pl"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_value(text: str, label: str) -> float:
    match = re.search(rf"{label} nH \(cm\*\*-2\)\s+([0-9.]+E[+-][0-9]+)", text, re.IGNORECASE)
    if match is None:
        raise RuntimeError(f"could not parse {label} nH from HEASARC response")
    return float(match.group(1))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--regions", type=Path, default=DEFAULT_REGIONS)
    parser.add_argument("--raw", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    regions_path = args.regions.resolve()
    raw = args.raw.resolve()
    output = args.output.resolve()
    raw.mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=True, exist_ok=True)
    regions = json.loads(regions_path.read_text(encoding="utf-8"))
    if regions["status"] != "both_clusters_passed_frozen_temperature_region_gate":
        raise RuntimeError("frozen region gate has not passed")

    records = []
    for cluster in regions["clusters"]:
        center = cluster["final_center"]
        params = {
            "Entry": f"{center['ra']:.14f},{center['dec']:.14f}",
            "NR": "GRB/SIMBAD+Sesame/NED",
            "CoordSys": "Equatorial",
            "equinox": "2000",
            "radius": "0.1",
            "usemap": "0",
        }
        url = ENDPOINT + "?" + urllib.parse.urlencode(params)
        destination = raw / f"{cluster['cluster']}_HI4PI.html"
        if destination.exists():
            payload = destination.read_bytes()
            reused = True
        else:
            request = urllib.request.Request(url, headers={"User-Agent": "SigmaGravity/17C"})
            with urllib.request.urlopen(request, timeout=60) as response:
                payload = response.read()
            destination.write_bytes(payload)
            reused = False
        text = html.unescape(payload.decode("utf-8"))
        if "Using map h1_nh_HI4PI.fits" not in text:
            raise RuntimeError(f"{cluster['cluster']} response is not an HI4PI result")
        average = parse_value(text, "Average")
        weighted = parse_value(text, "Weighted average")
        records.append(
            {
                "cluster": cluster["cluster"],
                "query_url": url,
                "center_ra_deg": center["ra"],
                "center_dec_deg": center["dec"],
                "cone_radius_deg": 0.1,
                "map": "h1_nh_HI4PI.fits",
                "average_nh_cm2": average,
                "weighted_average_nh_cm2": weighted,
                "relative_path": destination.relative_to(ROOT).as_posix(),
                "bytes": destination.stat().st_size,
                "sha256": sha256(destination),
                "reused": reused,
            }
        )

    provenance = {
        "status": "frozen_HI4PI_columns_downloaded_and_hashed",
        "generated_utc": datetime.now(UTC).isoformat(),
        "source": "NASA_HEASARC_nH_version_3_HI4PI_web_interface",
        "endpoint": ENDPOINT,
        "temperature_region_report_sha256": sha256(regions_path),
        "selection": "inverse-distance weighted average within the frozen 0.1-degree HI4PI cone",
        "records": records,
        "lensing_target_opened": False,
        "spectrum_extracted": False,
        "temperature_fit_run": False,
    }
    destination = output / "provenance.json"
    destination.write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(destination)


if __name__ == "__main__":
    main()
