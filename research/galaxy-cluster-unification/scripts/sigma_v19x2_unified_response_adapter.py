"""Independent V19W4 unified-response adapter for the future V19X successor.

This module contains no spectral fit, gravity equation, or data-selection rule.
It only validates that a terminal V19W4 report and mixed base/recovery index can
be consumed without assuming every cell lives under the original V19W archive.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
PRODUCT_ROLES = ("source_pha", "background_pha", "arf", "rmf")
AUTHORIZED_STATUS = (
    "hardened_unified_5082_response_archive_passed_and_v19x_successor_may_be_frozen"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def task_key(row: dict[str, Any]) -> tuple[str, int, int, int]:
    return (
        str(row["cluster"]),
        int(row["bin_id"]),
        int(row["obsid"]),
        int(row["ccd_id"]),
    )


def cell_name(row: dict[str, Any]) -> str:
    cluster, bin_id, obsid, ccd_id = task_key(row)
    return f"{cluster}_bin{bin_id}_obs{obsid}_ccd{ccd_id}"


def authorize_unified_index(
    report_path: Path,
    *,
    expected_config_sha256: str,
    expected_runner_sha256: str,
    expected_cells: int = 5082,
    expected_products: int = 20328,
    root: Path = ROOT,
) -> tuple[dict[str, Any], Path]:
    """Validate the terminal V19W4 authority and return its immutable index."""
    if not report_path.is_file():
        raise RuntimeError("V19W4 terminal authorization report is absent")
    report = load_json(report_path)
    if report.get("status") != AUTHORIZED_STATUS:
        raise RuntimeError(f"V19W4 status does not authorize V19X2: {report.get('status')}")
    if report.get("config_sha256") != expected_config_sha256:
        raise RuntimeError("V19W4 terminal report names another config")
    if report.get("runner_sha256") != expected_runner_sha256:
        raise RuntimeError("V19W4 terminal report names another runner")
    if not report.get("gates") or not all(report["gates"].values()):
        raise RuntimeError("V19W4 terminal report contains a failed gate")
    if int(report.get("unified_cells", -1)) != expected_cells:
        raise RuntimeError("V19W4 unified-cell count does not authorize V19X2")
    if int(report.get("unified_product_files", -1)) != expected_products:
        raise RuntimeError("V19W4 product count does not authorize V19X2")
    if report.get("base_v19w_archive_modified") is not False:
        raise RuntimeError("V19W4 did not prove the base archive remained immutable")
    if report.get("original_v19x_authorized") is not False:
        raise RuntimeError("V19W4 unexpectedly authorized the obsolete V19X protocol")
    if report.get("v19x_successor_configuration_may_be_frozen") is not True:
        raise RuntimeError("V19W4 withheld authority to freeze a V19X successor")
    item = report.get("unified_product_index", {})
    if int(item.get("rows", -1)) != expected_cells:
        raise RuntimeError("V19W4 unified index row count changed")
    index_path = root / str(item.get("path", ""))
    if not index_path.is_file():
        raise RuntimeError("V19W4 unified product index is absent")
    if index_path.stat().st_size != int(item.get("bytes", -1)):
        raise RuntimeError("V19W4 unified product index size changed")
    if sha256(index_path) != item.get("sha256"):
        raise RuntimeError("V19W4 unified product index hash changed")
    return report, index_path


def required_index_fields() -> set[str]:
    fields = {
        "production_index",
        "batch_id",
        "cluster",
        "bin_id",
        "obsid",
        "ccd_id",
        "cell_name",
        "archive",
        "cell_directory",
        "cell_report_sha256",
        "four_product_bytes",
    }
    for role in PRODUCT_ROLES:
        fields.update(
            {
                f"{role}_name",
                f"{role}_bytes",
                f"{role}_sha256",
            }
        )
    return fields


def load_unified_index(
    path: Path, expected_rows: int = 5082
) -> dict[tuple[str, int, int, int], dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None or not required_index_fields().issubset(
            reader.fieldnames
        ):
            raise RuntimeError("V19W4 unified index lacks required fields")
        rows = list(reader)
    indexed = {task_key(row): row for row in rows}
    if len(rows) != expected_rows or len(indexed) != expected_rows:
        raise RuntimeError("V19W4 unified index is incomplete or has duplicate tasks")
    return indexed


def path_is_within(path: Path, root: Path) -> bool:
    resolved = path.resolve()
    frozen_root = root.resolve()
    return resolved == frozen_root or resolved.is_relative_to(frozen_root)


def validate_unified_cell(
    manifest_row: dict[str, str],
    index_row: dict[str, str],
    allowed_archive_roots: dict[str, Path],
) -> dict[str, Any]:
    """Independently validate one base or recovery checkpoint from its index row."""
    key = task_key(manifest_row)
    if task_key(index_row) != key:
        raise RuntimeError(f"V19X2 unified index identity mismatch: {key}")
    archive = index_row["archive"]
    if archive not in {"base_v19w", "v19w4_recovery"}:
        raise RuntimeError(f"V19X2 unknown response archive: {index_row['archive']}")
    if set(allowed_archive_roots) != {"base_v19w", "v19w4_recovery"}:
        raise RuntimeError("V19X2 allowed archive-root mapping is incomplete")
    cell_directory = Path(index_row["cell_directory"])
    if not cell_directory.is_absolute() or not path_is_within(
        cell_directory, allowed_archive_roots[archive]
    ):
        raise RuntimeError(f"V19X2 cell path is outside frozen archive roots: {key}")
    report_path = cell_directory / "cell_report.json"
    if not report_path.is_file():
        raise RuntimeError(f"V19X2 missing cell report: {key}")
    if sha256(report_path) != index_row["cell_report_sha256"]:
        raise RuntimeError(f"V19X2 changed cell report hash: {key}")
    report = load_json(report_path)
    expected_name = cell_name(manifest_row)
    if (
        index_row["cell_name"] != expected_name
        or task_key(report) != key
        or report.get("cell_name") != expected_name
    ):
        raise RuntimeError(f"V19X2 cell report identity mismatch: {key}")
    if not report.get("gates") or not all(report["gates"].values()):
        raise RuntimeError(f"V19X2 cell contains a failed response gate: {key}")
    preflight = report["preflight"]
    if int(preflight["source_band_events"]) != int(
        manifest_row["source_band_events"]
    ):
        raise RuntimeError(f"V19X2 source event count mismatch: {key}")
    if int(preflight["background_band_events"]) != int(
        manifest_row["background_band_events"]
    ):
        raise RuntimeError(f"V19X2 background event count mismatch: {key}")

    paths: dict[str, Path] = {}
    product_bytes = 0
    for role in PRODUCT_ROLES:
        item = report["products"][role]
        if item["name"] != index_row[f"{role}_name"]:
            raise RuntimeError(f"V19X2 changed {role} name: {key}")
        path = cell_directory / "products" / item["name"]
        expected_bytes = int(index_row[f"{role}_bytes"])
        if (
            not path.is_file()
            or path.stat().st_size != expected_bytes
            or int(item["bytes"]) != expected_bytes
        ):
            raise RuntimeError(f"V19X2 missing or resized {role}: {key}")
        digest = sha256(path)
        if digest != item["sha256"] or digest != index_row[f"{role}_sha256"]:
            raise RuntimeError(f"V19X2 changed {role} hash: {key}")
        paths[role] = path
        product_bytes += expected_bytes
    if product_bytes != int(index_row["four_product_bytes"]):
        raise RuntimeError(f"V19X2 four-product byte total mismatch: {key}")

    pha_audit = report["source_pha_channel_audit"]
    pha_total = int(pha_audit["pha_total_counts"])
    if not pha_audit["exact"] or pha_total <= 0:
        raise RuntimeError(f"V19X2 source PHA count audit failed: {key}")
    return {
        "cluster": key[0],
        "bin_id": key[1],
        "obsid": key[2],
        "ccd_id": key[3],
        "cell_name": index_row["cell_name"],
        "archive": index_row["archive"],
        "cell_directory": cell_directory,
        "source_band_events": int(manifest_row["source_band_events"]),
        "background_band_events": int(manifest_row["background_band_events"]),
        "source_pha_total_counts": pha_total,
        "source_pha": paths["source_pha"],
        "source_pha_sha256": index_row["source_pha_sha256"],
    }


def validate_unified_archive(
    manifest: list[dict[str, str]],
    index_path: Path,
    allowed_archive_roots: dict[str, Path],
) -> dict[tuple[str, int, int, int], dict[str, Any]]:
    indexed = load_unified_index(index_path, len(manifest))
    validated: dict[tuple[str, int, int, int], dict[str, Any]] = {}
    for manifest_row in manifest:
        key = task_key(manifest_row)
        if key not in indexed:
            raise RuntimeError(f"V19X2 unified index lacks manifest task: {key}")
        validated[key] = validate_unified_cell(
            manifest_row, indexed[key], allowed_archive_roots
        )
    if len(validated) != len(manifest):
        raise RuntimeError("V19X2 did not validate every manifest task")
    return validated
