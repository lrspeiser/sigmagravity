#!/usr/bin/env python3
"""Download and hash the frozen V19S HI4PI Galactic columns."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
import re
import urllib.parse
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "sigma_v19s_hi4pi_acquisition.json"
DEFAULT_RAW = ROOT / "data" / "raw" / "sigma_v19s_hi4pi"
DEFAULT_OUTPUT = ROOT / "results" / "sigma_v19s_hi4pi_acquisition"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_value(text: str, label: str) -> float:
    match = re.search(
        rf"{label} nH \(cm\*\*-2\)\s+([0-9.]+E[+-][0-9]+)",
        text,
        re.IGNORECASE,
    )
    if match is None:
        raise RuntimeError(f"could not parse {label} nH from HEASARC response")
    value = float(match.group(1))
    if not math.isfinite(value) or value <= 0.0:
        raise RuntimeError(f"invalid {label} nH value: {value}")
    return value


def validate(config: dict[str, Any]) -> dict[str, Any]:
    for key, value in config["parents"].items():
        if key.endswith("_sha256"):
            continue
        expected = config["parents"].get(f"{key}_sha256")
        if expected is not None and sha256(ROOT / value) != expected:
            raise RuntimeError(f"V19S parent hash mismatch: {value}")
    source = json.loads(
        (ROOT / config["parents"]["source_map_report"]).read_text(encoding="utf-8")
    )
    by_cluster = {row["cluster"]: row for row in source["clusters"]}
    for cluster, target in config["targets"].items():
        center = by_cluster[cluster]["final_center"]
        if (
            float(target["center_ra_deg"]) != float(center["ra"])
            or float(target["center_dec_deg"]) != float(center["dec"])
        ):
            raise RuntimeError(f"V19S target center changed: {cluster}")
    return source


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--raw", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    validate(config)
    raw = args.raw.resolve()
    output = args.output.resolve()
    raw.mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=True, exist_ok=True)
    source = config["source"]
    records = []
    for cluster, target in config["targets"].items():
        params = {
            "Entry": f"{target['center_ra_deg']:.14f},{target['center_dec_deg']:.14f}",
            "NR": "GRB/SIMBAD+Sesame/NED",
            "CoordSys": source["coordinate_system"],
            "equinox": str(source["equinox"]),
            "radius": str(source["cone_radius_deg"]),
            "usemap": str(source["usemap"]),
        }
        url = source["endpoint"] + "?" + urllib.parse.urlencode(params)
        destination = raw / f"{cluster}_HI4PI.html"
        if destination.exists():
            payload = destination.read_bytes()
            reused = True
        else:
            request = urllib.request.Request(url, headers={"User-Agent": "SigmaGravity/19S"})
            with urllib.request.urlopen(request, timeout=60) as response:
                payload = response.read()
            destination.write_bytes(payload)
            reused = False
        text = html.unescape(payload.decode("utf-8"))
        expected_map = f"Using map {source['map']}"
        if expected_map not in text:
            raise RuntimeError(f"{cluster} response is not an HI4PI result")
        records.append(
            {
                "cluster": cluster,
                "query_url": url,
                "center_ra_deg": target["center_ra_deg"],
                "center_dec_deg": target["center_dec_deg"],
                "cone_radius_deg": source["cone_radius_deg"],
                "map": source["map"],
                "average_nh_cm2": parse_value(text, "Average"),
                "weighted_average_nh_cm2": parse_value(text, "Weighted average"),
                "relative_path": destination.relative_to(ROOT).as_posix(),
                "bytes": destination.stat().st_size,
                "sha256": sha256(destination),
                "reused": reused,
            }
        )
    report = {
        "status": "both_frozen_hi4pi_columns_acquired_and_hashed",
        "protocol_version": config["protocol_version"],
        "generated_utc": datetime.now(UTC).isoformat(),
        "config_sha256": sha256(config_path),
        "runner_sha256": sha256(Path(__file__).resolve()),
        "source": source,
        "records": records,
        "temperature_fit_authorized": len(records) == len(config["targets"]),
        "spectrum_or_response_constructed": False,
        "temperature_or_abundance_fit_run": False,
        "lensing_target_opened": False,
        "gravity_formula_or_parameter_changed": False,
    }
    report_path = output / "provenance.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(report_path)
    for record in records:
        print(f"{record['cluster']}: {record['weighted_average_nh_cm2']:.6g} cm^-2")


if __name__ == "__main__":
    main()
