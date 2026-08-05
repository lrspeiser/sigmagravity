#!/usr/bin/env python3
"""Download and hash the target-blind Sigma v19G Gaia DR3 reference cones."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import requests
import sigma_v19f_chandra_common as common

ROOT = common.ROOT
DEFAULT_CONFIG = ROOT / "configs" / "sigma_v19g_gaia_hierarchical_astrometry.json"
DEFAULT_OUTPUT = ROOT / "results" / "sigma_v19g_gaia_acquisition"
FROZEN_STATUS = (
    "frozen before any v19 Gaia query, X-ray/Gaia cross-match, relative X-ray "
    "cross-match, transform application, registered science-image inspection, shock "
    "fit, source construction, or replacement-cluster lensing access"
)


def validate(config_path: Path) -> dict:
    config = common.load_json(config_path)
    if config["status"] != FROZEN_STATUS:
        raise RuntimeError("v19G Gaia protocol is not frozen")
    common.validate_parent_hashes(config)
    if set(config["clusters"]) != {"BULLET", "ABELL2146"}:
        raise RuntimeError("v19G changed the development pair")
    if config["matching"]["method"] != "trans":
        raise RuntimeError("v19G absolute astrometry is not translation only")
    if config["relative_matching"]["method"] != "trans":
        raise RuntimeError("v19G relative astrometry is not translation only")
    for section in ("matching", "relative_matching"):
        if any(
            float(config[section][key]) != expected
            for key, expected in (
                ("rotation_deg", 0.0),
                ("scale", 1.0),
                ("shear", 0.0),
            )
        ):
            raise RuntimeError(f"v19G {section} can change scale, rotation, or shear")
    return config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    config_path = args.config.resolve()
    config = validate(config_path)
    shared = common.load_module(
        ROOT / config["parents"]["shared_gaia_downloader"],
        "sigma_v17a_gaia_download_shared",
    )
    raw = ROOT / config["archive"]["raw_directory"]
    raw.mkdir(parents=True, exist_ok=True)
    records = []
    for cluster, values in config["clusters"].items():
        query = shared.adql(config, values)
        response = requests.post(
            config["archive"]["tap_sync_url"],
            data={
                "REQUEST": "doQuery",
                "LANG": "ADQL",
                "FORMAT": config["archive"]["format"],
                "QUERY": query,
            },
            timeout=(30, 300),
        )
        response.raise_for_status()
        rows = shared.validate_csv(response.content, config["archive"]["columns"])
        path = raw / f"{cluster}_gaia_dr3.csv"
        if path.exists() and path.read_bytes() != response.content:
            raise RuntimeError(
                f"immutable v19G Gaia response differs from the current response: {path}"
            )
        if not path.exists():
            path.write_bytes(response.content)
        records.append(
            {
                "cluster": cluster,
                "query": query,
                "query_sha256": hashlib.sha256(query.encode("utf-8")).hexdigest(),
                "relative_path": path.relative_to(ROOT).as_posix(),
                "rows": rows,
                "bytes": path.stat().st_size,
                "sha256": common.sha256(path),
            }
        )
        print(f"{cluster}: {rows} Gaia DR3 sources", flush=True)

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    report = {
        "status": "frozen_v19g_Gaia_DR3_reference_cones_downloaded_and_hashed",
        "protocol_version": config["protocol_version"],
        "generated_utc": datetime.now(UTC).isoformat(),
        "config_sha256": common.sha256(config_path),
        "records": records,
        "files": len(records),
        "rows": sum(row["rows"] for row in records),
        "bytes": sum(row["bytes"] for row in records),
        "xray_source_crossmatch_run": False,
        "astrometric_offset_fit": False,
        "registered_science_image_inspected": False,
        "shock_front_fitted": False,
        "source_constructed": False,
        "lensing_target_opened": False,
    }
    report_path = output / "provenance.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(report_path)


if __name__ == "__main__":
    main()
