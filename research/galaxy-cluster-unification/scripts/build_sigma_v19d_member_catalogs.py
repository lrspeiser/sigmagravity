#!/usr/bin/env python3
"""Extract the frozen v19D member catalogs from primary-paper TeX tables."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import tarfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "sigma_v19d_member_catalog_extraction.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def table_text(archive_path: Path, member_name: str) -> str:
    with tarfile.open(archive_path, mode="r:*") as archive:
        handle = archive.extractfile(member_name)
        if handle is None:
            raise RuntimeError(f"missing archive member {member_name}")
        return handle.read().decode("latin-1")


def ra_hms_to_deg(value: str) -> float:
    hours, minutes, seconds = (float(item) for item in value.split(":"))
    return 15.0 * (hours + minutes / 60.0 + seconds / 3600.0)


def dec_dms_to_deg(value: str) -> float:
    sign = -1.0 if value.startswith("-") else 1.0
    degrees, minutes, seconds = (
        float(item) for item in value.lstrip("+-").split(":")
    )
    return sign * (degrees + minutes / 60.0 + seconds / 3600.0)


def clean_tex_cell(value: str) -> str:
    return value.strip().rstrip("\\").strip()


def bullet_rows(text: str, source_arxiv_id: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in text.splitlines():
        if not re.match(r"^\s*\d{2}\s*&", line):
            continue
        cells = [clean_tex_cell(cell) for cell in line.split("&")]
        if len(cells) != 11:
            raise RuntimeError(f"unexpected Bullet table row: {line}")
        spectral_type = cells[10] if cells[10] in {"E", "L"} else ""
        rows.append(
            {
                "cluster": "BULLET",
                "object_id": cells[0],
                "ra_deg": ra_hms_to_deg(cells[1]),
                "dec_deg": dec_dms_to_deg(cells[2]),
                "heliocentric_cz_km_s": int(cells[6]),
                "cz_uncertainty_km_s": int(cells[7]),
                "subcluster_label": "",
                "is_bcg": False,
                "spectral_type": spectral_type,
                "source_arxiv_id": source_arxiv_id,
            }
        )
    return rows


def abell2146_rows(text: str, source_arxiv_id: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in text.splitlines():
        if not re.match(r"^\s*\d+\s*&\s*\d{3}\.\d+\s*&", line):
            continue
        cells = [clean_tex_cell(cell) for cell in line.split("&")]
        if len(cells) != 7:
            raise RuntimeError(f"unexpected Abell 2146 table row: {line}")
        subgroup = "A" if cells[5].startswith("A") else "B"
        rows.append(
            {
                "cluster": "ABELL2146",
                "object_id": cells[0],
                "ra_deg": float(cells[1]),
                "dec_deg": float(cells[2]),
                "heliocentric_cz_km_s": int(cells[3]),
                "cz_uncertainty_km_s": int(cells[4]),
                "subcluster_label": subgroup,
                "is_bcg": "BCG" in cells[5],
                "spectral_type": cells[6],
                "source_arxiv_id": source_arxiv_id,
            }
        )
    return rows


def validate_rows(rows: list[dict[str, Any]], expected_rows: int) -> None:
    if len(rows) != expected_rows:
        raise RuntimeError(f"parsed {len(rows)} rows, expected {expected_rows}")
    identifiers = [row["object_id"] for row in rows]
    if len(identifiers) != len(set(identifiers)):
        raise RuntimeError("member identifiers are not unique")
    for row in rows:
        if not 0.0 <= float(row["ra_deg"]) < 360.0:
            raise RuntimeError("right ascension is outside [0,360)")
        if not -90.0 <= float(row["dec_deg"]) <= 90.0:
            raise RuntimeError("declination is outside [-90,90]")
        if int(row["heliocentric_cz_km_s"]) <= 0:
            raise RuntimeError("heliocentric cz is not positive")
        if int(row["cz_uncertainty_km_s"]) <= 0:
            raise RuntimeError("quoted cz uncertainty is not positive")


def write_catalog(path: Path, columns: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def catalog_summary(rows: list[dict[str, Any]], path: Path) -> dict[str, Any]:
    velocities = [float(row["heliocentric_cz_km_s"]) for row in rows]
    errors = [float(row["cz_uncertainty_km_s"]) for row in rows]
    subgroup_counts: dict[str, int] = {}
    for row in rows:
        label = str(row["subcluster_label"] or "unassigned")
        subgroup_counts[label] = subgroup_counts.get(label, 0) + 1
    return {
        "rows": len(rows),
        "output": path.relative_to(ROOT).as_posix(),
        "output_sha256": sha256(path),
        "mean_heliocentric_cz_km_s": sum(velocities) / len(velocities),
        "minimum_cz_uncertainty_km_s": min(errors),
        "maximum_cz_uncertainty_km_s": max(errors),
        "subcluster_counts": subgroup_counts,
        "bcg_rows": sum(bool(row["is_bcg"]) for row in rows),
    }


def build(config_path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    config_path = config_path.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if not config["status"].startswith("frozen before parsing"):
        raise RuntimeError("v19D member extraction protocol is not frozen")
    authorization = config["authorization"]
    if authorization["use_lensing_or_halo_payload"]:
        raise RuntimeError("member extraction cannot use lensing or halo payloads")
    if authorization["infer_missing_measurement_uncertainties"]:
        raise RuntimeError("member extraction cannot invent uncertainties")

    hashes = {"config": sha256(config_path)}
    for key in ("source_acquisition_config", "source_acquisition_manifest"):
        path = ROOT / config["parents"][key]
        actual = sha256(path)
        if actual != config["parents"][f"{key}_sha256"]:
            raise RuntimeError(f"frozen {key} changed")
        hashes[key] = actual

    columns = list(config["required_output_columns"])
    summaries: dict[str, Any] = {}
    for cluster, definition in config["catalogs"].items():
        archive_path = ROOT / definition["archive"]
        actual_archive_hash = sha256(archive_path)
        if actual_archive_hash != definition["archive_sha256"]:
            raise RuntimeError(f"frozen {cluster} archive changed")
        hashes[f"{cluster}_archive"] = actual_archive_hash
        text = table_text(archive_path, definition["member_name"])
        parser = bullet_rows if cluster == "BULLET" else abell2146_rows
        rows = parser(text, definition["source_arxiv_id"])
        validate_rows(rows, int(definition["expected_rows"]))
        output = ROOT / definition["output"]
        write_catalog(output, columns, rows)
        summaries[cluster] = catalog_summary(rows, output)

    report = {
        "status": "completed Sigma v19D lossless member-table extraction",
        "protocol_version": config["protocol_version"],
        "generated_utc": datetime.now(UTC).isoformat(),
        "input_hashes": hashes,
        "catalogs": summaries,
        "total_rows": sum(item["rows"] for item in summaries.values()),
        "all_rows_retain_quoted_cz_uncertainty": True,
        "missing_uncertainties_inferred": False,
        "lensing_or_halo_payload_used": False,
        "source_construction_performed": False,
        "gravity_parameters_fit": 0,
        "holdout_opened": False,
        "next_gate": "construct registered shock-front measurements and an assumption-aware projection ensemble before defining any causal Sigma source",
    }
    report_path = ROOT / config["output_report"]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    print(json.dumps(build(args.config), indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
