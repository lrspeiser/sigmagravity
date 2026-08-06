#!/usr/bin/env python3
"""Inventory A2319 FITS headers and schemas without reading HDU data values."""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import re
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import astropy
from astropy.io import fits

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "sigma_v19cy_a2319_fits_metadata_inventory.json"
BLOCK_BYTES = 4 * 1024 * 1024


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(BLOCK_BYTES), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def scalar(value: Any) -> str | int | float | bool | None:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def validate_inputs(config_path: Path) -> tuple[dict[str, Any], dict[str, Any], Path]:
    config = load_json(config_path)
    if config.get("protocol_version") != "SIGMA-V19CY-A2319-FITS-METADATA-1.0.0":
        raise RuntimeError("unexpected A2319 FITS metadata protocol")
    if config.get("status") != (
        "frozen after the A2319 environment passed but before opening any XRISM FITS header or HDU schema"
    ):
        raise RuntimeError("A2319 FITS metadata protocol is not frozen")
    parents = config["parents"]
    manifest = ROOT / parents["acquisition_manifest"]
    provenance_path = ROOT / parents["download_provenance"]
    environment_path = ROOT / parents["environment_report"]
    for path, expected in (
        (manifest, parents["acquisition_manifest_sha256"]),
        (provenance_path, parents["download_provenance_sha256"]),
        (environment_path, parents["environment_report_sha256"]),
    ):
        if not path.is_file() or sha256(path) != expected:
            raise RuntimeError(f"frozen parent changed: {path}")
    provenance = load_json(provenance_path)
    environment = load_json(environment_path)
    if environment.get("status") != parents["required_environment_status"]:
        raise RuntimeError("A2319 environment did not pass")
    if astropy.__version__ != config["runtime"]["astropy_version"]:
        raise RuntimeError("Astropy version changed")
    authorization = config["authorization"]
    for key in (
        "read_any_table_or_image_value",
        "calculate_gain_solution",
        "fit_spectrum_or_velocity",
        "access_validation_or_holdout_assets",
        "open_lensing_halo_or_gravity_targets",
        "change_gravity_formula_or_parameters",
        "derive_or_select_action",
    ):
        if authorization[key]:
            raise RuntimeError(f"sealed FITS metadata boundary is open: {key}")
    return config, provenance, manifest


def selected_rows(config: dict[str, Any], manifest: Path) -> list[dict[str, str]]:
    pattern = re.compile(config["selection"]["path_regex"], flags=re.IGNORECASE)
    with manifest.open(encoding="utf-8", newline="") as stream:
        rows = [
            row
            for row in csv.DictReader(stream)
            if row["asset_group"].startswith(config["selection"]["asset_group_prefix"])
            and pattern.search(row["download_path"])
        ]
    if len(rows) != config["selection"]["expected_files"]:
        raise RuntimeError(f"expected 87 metadata files, selected {len(rows)}")
    if sorted({row["obsid"] for row in rows}) != sorted(config["selection"]["obsids"]):
        raise RuntimeError("metadata selection opened an unexpected ObsID")
    return sorted(rows, key=lambda row: row["download_path"])


@contextmanager
def fits_source(path: Path):
    if path.suffix.lower() == ".gz":
        with gzip.open(path, "rb") as stream:
            yield stream
    else:
        yield path


def inspect_header_only(path: Path, config: dict[str, Any]) -> tuple[list[dict[str, Any]], bool]:
    records: list[dict[str, Any]] = []
    all_unloaded = True
    with fits_source(path) as source, fits.open(
        source,
        mode="readonly",
        memmap=False,
        lazy_load_hdus=True,
        do_not_scale_image_data=True,
    ) as hdus:
        for index, hdu in enumerate(hdus):
            header = hdu.header
            was_loaded = bool(getattr(hdu, "_data_loaded", False))
            values = {
                key: scalar(header[key])
                for key in config["header_keys"]
                if key in header
            }
            columns: list[dict[str, Any]] = []
            fields = int(header.get("TFIELDS", 0) or 0)
            for column_index in range(1, fields + 1):
                item: dict[str, Any] = {"index": column_index}
                for output_key, prefix_key in (
                    ("name", "column_name_prefix"),
                    ("format", "column_format_prefix"),
                    ("unit", "column_unit_prefix"),
                ):
                    card = f"{config['schema_keys'][prefix_key]}{column_index}"
                    if card in header:
                        item[output_key] = scalar(header[card])
                columns.append(item)
            after_loaded = bool(getattr(hdu, "_data_loaded", False))
            all_unloaded = all_unloaded and not was_loaded and not after_loaded
            records.append(
                {
                    "index": index,
                    "class": type(hdu).__name__,
                    "header": values,
                    "columns": columns,
                    "data_loaded_before_or_after_header_inspection": was_loaded or after_loaded,
                }
            )
    return records, all_unloaded


def build_report(config_path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    config, provenance, manifest = validate_inputs(config_path)
    rows = selected_rows(config, manifest)
    provenance_by_path = {record["download_path"]: record for record in provenance["records"]}
    raw_root = (ROOT / config["paths"]["raw_root"]).resolve()
    files: list[dict[str, Any]] = []
    every_unloaded = True
    for row in rows:
        path = (raw_root / row["download_path"]).resolve()
        if not path.is_relative_to(raw_root):
            raise RuntimeError(f"metadata path escapes raw root: {row['download_path']}")
        terminal = provenance_by_path.get(row["download_path"])
        if terminal is None:
            raise RuntimeError(f"metadata file absent from terminal provenance: {row['download_path']}")
        digest = sha256(path)
        if path.stat().st_size != terminal["bytes"] or digest != terminal["sha256"]:
            raise RuntimeError(f"metadata file changed: {row['download_path']}")
        hdus, unloaded = inspect_header_only(path, config)
        every_unloaded = every_unloaded and unloaded
        files.append(
            {
                "asset_group": row["asset_group"],
                "role": row["role"],
                "obsid": row["obsid"],
                "download_path": row["download_path"],
                "bytes": terminal["bytes"],
                "sha256": digest,
                "hdus": hdus,
            }
        )
    report = {
        "protocol_version": config["protocol_version"],
        "status": (
            "a2319_development_fits_metadata_and_schemas_inventoried_without_loading_data"
            if len(files) == config["acceptance"]["exact_file_count"] and every_unloaded
            else "a2319_fits_metadata_inventory_failed_closed"
        ),
        "generated_utc": datetime.now(UTC).isoformat(),
        "config_sha256": sha256(config_path),
        "files": len(files),
        "compressed_bytes": sum(item["bytes"] for item in files),
        "hdus": sum(len(item["hdus"]) for item in files),
        "every_hdu_data_object_remained_unloaded": every_unloaded,
        "table_or_image_value_read": False,
        "scientific_fit_performed": False,
        "validation_or_holdout_accessed": False,
        "files_metadata": files,
        "authorization": {
            "freeze_gain_reconstruction_protocol": (
                len(files) == config["acceptance"]["exact_file_count"] and every_unloaded
            ),
            "read_any_table_or_image_value": False,
            "calculate_gain_solution": False,
            "fit_spectrum_or_velocity": False,
            "access_validation_or_holdout_assets": False,
            "open_lensing_halo_or_gravity_targets": False,
            "derive_or_select_action": False,
        },
    }
    output = ROOT / config["paths"]["report"]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    summary = build_report()
    print(
        json.dumps(
            {
                key: summary[key]
                for key in (
                    "status",
                    "files",
                    "compressed_bytes",
                    "hdus",
                    "every_hdu_data_object_remained_unloaded",
                )
            },
            indent=2,
            sort_keys=True,
        )
    )
