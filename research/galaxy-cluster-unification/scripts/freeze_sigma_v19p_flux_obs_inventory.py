#!/usr/bin/env python3
"""Hash the immutable V19P event and per-observation flux_obs inputs.

This deliberately records bytes and hashes only.  It does not open a FITS
array, count an event, construct a region task, or evaluate a V19P gate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_REPORT = ROOT / "results" / "sigma_v19h_source_maps" / "report.json"
DEFAULT_OUTPUT = (
    ROOT
    / "results"
    / "sigma_v19p_exact_flux_obs_support"
    / "input_inventory.json"
)
CLUSTERS = ("BULLET", "ABELL2146")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def record(path: Path, role: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    print(f"hashing {role}: {path}", flush=True)
    return {
        "role": role,
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-report", type=Path, default=DEFAULT_SOURCE_REPORT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    source_report = args.source_report.resolve()
    report = json.loads(source_report.read_text(encoding="utf-8"))
    by_cluster = {row["cluster"]: row for row in report["clusters"]}
    clusters = []
    for cluster in CLUSTERS:
        source = by_cluster[cluster]
        broad_counts = Path(source["products"]["broad_counts"])
        broad_dir = broad_counts.parent
        observations = []
        for observation in sorted(source["observations"], key=lambda row: int(row["obsid"])):
            obsid = int(observation["obsid"])
            observations.append(
                {
                    "obsid": obsid,
                    "products": [
                        record(Path(observation["science_reprojected"]), "science_event"),
                        record(Path(observation["blanksky_reprojected"]), "blanksky_event"),
                        record(
                            broad_dir / f"{obsid}_0.5-7.0_thresh.img",
                            "flux_obs_broad_counts",
                        ),
                        record(
                            broad_dir / f"{obsid}_0.5-7.0_thresh.expmap",
                            "flux_obs_broad_exposure",
                        ),
                        record(broad_dir / f"{obsid}.fov", "flux_obs_support_fov"),
                    ],
                }
            )
        clusters.append({"cluster": cluster, "observations": observations})
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    inventory = {
        "status": "hash_only_inventory_frozen_before_v19p_array_access",
        "generated_utc": datetime.now(UTC).isoformat(),
        "source_report": str(source_report),
        "source_report_sha256": sha256(source_report),
        "clusters": clusters,
        "integrity": {
            "fits_array_opened": False,
            "event_row_read": False,
            "corrected_count_or_task_outcome_known": False,
            "spectrum_or_response_constructed": False,
            "gravity_formula_or_parameter_changed": False,
        },
    }
    output.write_text(json.dumps(inventory, indent=2) + "\n", encoding="utf-8")
    print(f"inventory: {output}")
    print(f"sha256: {sha256(output)}")


if __name__ == "__main__":
    main()
