from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = ROOT / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import sigma_v19x2_unified_response_adapter as adapter


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def make_cell(
    archive_root: Path,
    archive: str,
    cluster: str = "TEST",
    bin_id: int = 7,
    obsid: int = 42,
    ccd_id: int = 3,
) -> tuple[dict[str, str], dict[str, str]]:
    name = f"{cluster}_bin{bin_id}_obs{obsid}_ccd{ccd_id}"
    cell = archive_root / "completed" / name
    products = {}
    row: dict[str, str] = {
        "production_index": "1",
        "batch_id": "1",
        "cluster": cluster,
        "bin_id": str(bin_id),
        "obsid": str(obsid),
        "ccd_id": str(ccd_id),
        "cell_name": name,
        "archive": archive,
        "cell_directory": str(cell.resolve()),
    }
    total_bytes = 0
    names = {
        "source_pha": "source.pi",
        "background_pha": "background.pi",
        "arf": "source.arf",
        "rmf": "source.rmf",
    }
    for role, filename in names.items():
        path = cell / "products" / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"{archive}-{role}".encode())
        digest = sha256(path)
        size = path.stat().st_size
        products[role] = {"name": filename, "bytes": size, "sha256": digest}
        row[f"{role}_name"] = filename
        row[f"{role}_bytes"] = str(size)
        row[f"{role}_sha256"] = digest
        total_bytes += size
    report = {
        "cluster": cluster,
        "bin_id": bin_id,
        "obsid": obsid,
        "ccd_id": ccd_id,
        "cell_name": name,
        "gates": {"fixture": True},
        "preflight": {"source_band_events": 11, "background_band_events": 5},
        "products": products,
        "source_pha_channel_audit": {"exact": True, "pha_total_counts": 13},
    }
    report_path = cell / "cell_report.json"
    write_json(report_path, report)
    row["cell_report_sha256"] = sha256(report_path)
    row["four_product_bytes"] = str(total_bytes)
    manifest = {
        "cluster": cluster,
        "bin_id": str(bin_id),
        "obsid": str(obsid),
        "ccd_id": str(ccd_id),
        "source_band_events": "11",
        "background_band_events": "5",
    }
    return manifest, row


def write_index(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_adapter_accepts_base_and_recovery_cells(tmp_path: Path) -> None:
    base = tmp_path / "base"
    recovery = tmp_path / "recovery"
    manifest_base, row_base = make_cell(base, "base_v19w", obsid=41)
    manifest_recovery, row_recovery = make_cell(
        recovery, "v19w4_recovery", obsid=42
    )
    index = tmp_path / "index.csv"
    write_index(index, [row_base, row_recovery])
    validated = adapter.validate_unified_archive(
        [manifest_base, manifest_recovery],
        index,
        {"base_v19w": base, "v19w4_recovery": recovery},
    )
    assert len(validated) == 2
    assert {row["archive"] for row in validated.values()} == {
        "base_v19w",
        "v19w4_recovery",
    }
    assert all(row["source_pha"].is_file() for row in validated.values())


def test_adapter_accepts_v19w5_recovery_only_when_declared(tmp_path: Path) -> None:
    base = tmp_path / "base"
    recovery = tmp_path / "recovery"
    manifest_base, row_base = make_cell(base, "base_v19w", obsid=41)
    manifest_recovery, row_recovery = make_cell(
        recovery, "v19w5_recovery", obsid=42, ccd_id=7
    )
    index = tmp_path / "index.csv"
    write_index(index, [row_base, row_recovery])
    roots = {"base_v19w": base, "v19w5_recovery": recovery}
    validated = adapter.validate_unified_archive(
        [manifest_base, manifest_recovery],
        index,
        roots,
        recovery_archive="v19w5_recovery",
    )
    assert {row["archive"] for row in validated.values()} == {
        "base_v19w",
        "v19w5_recovery",
    }
    with pytest.raises(RuntimeError, match="archive-root mapping|unknown response"):
        adapter.validate_unified_archive(
            [manifest_base, manifest_recovery], index, roots
        )


def test_adapter_refuses_paths_outside_frozen_roots(tmp_path: Path) -> None:
    archive = tmp_path / "archive"
    manifest, row = make_cell(archive, "base_v19w")
    with pytest.raises(RuntimeError, match="outside frozen archive roots"):
        adapter.validate_unified_cell(
            manifest,
            row,
            {
                "base_v19w": tmp_path / "another",
                "v19w4_recovery": tmp_path / "recovery",
            },
        )


def test_adapter_enforces_archive_label_to_root_mapping(tmp_path: Path) -> None:
    base = tmp_path / "base"
    recovery = tmp_path / "recovery"
    manifest, row = make_cell(base, "base_v19w")
    row["archive"] = "v19w4_recovery"
    with pytest.raises(RuntimeError, match="outside frozen archive roots"):
        adapter.validate_unified_cell(
            manifest,
            row,
            {"base_v19w": base, "v19w4_recovery": recovery},
        )


def test_adapter_detects_mutated_product_after_index(tmp_path: Path) -> None:
    archive = tmp_path / "archive"
    manifest, row = make_cell(archive, "base_v19w")
    source = Path(row["cell_directory"]) / "products" / row["source_pha_name"]
    source.write_bytes(b"mutated")
    with pytest.raises(RuntimeError, match="resized source_pha|changed source_pha hash"):
        adapter.validate_unified_cell(
            manifest,
            row,
            {"base_v19w": archive, "v19w4_recovery": tmp_path / "recovery"},
        )


def test_adapter_detects_changed_cell_report(tmp_path: Path) -> None:
    archive = tmp_path / "archive"
    manifest, row = make_cell(archive, "base_v19w")
    report = Path(row["cell_directory"]) / "cell_report.json"
    report.write_text(report.read_text(encoding="utf-8") + " ", encoding="utf-8")
    with pytest.raises(RuntimeError, match="changed cell report hash"):
        adapter.validate_unified_cell(
            manifest,
            row,
            {"base_v19w": archive, "v19w4_recovery": tmp_path / "recovery"},
        )


def test_terminal_authorization_checks_report_and_index(tmp_path: Path) -> None:
    root = tmp_path / "root"
    index = root / "results" / "index.csv"
    index.parent.mkdir(parents=True)
    index.write_text("fixture\n", encoding="utf-8")
    report = {
        "status": adapter.AUTHORIZED_STATUS,
        "config_sha256": "a" * 64,
        "runner_sha256": "b" * 64,
        "gates": {"all": True},
        "unified_cells": 1,
        "unified_product_files": 4,
        "base_v19w_archive_modified": False,
        "original_v19x_authorized": False,
        "v19x_successor_configuration_may_be_frozen": True,
        "unified_product_index": {
            "path": "results/index.csv",
            "rows": 1,
            "bytes": index.stat().st_size,
            "sha256": sha256(index),
        },
    }
    report_path = root / "report.json"
    write_json(report_path, report)
    loaded, authorized_index = adapter.authorize_unified_index(
        report_path,
        expected_config_sha256="a" * 64,
        expected_runner_sha256="b" * 64,
        expected_cells=1,
        expected_products=4,
        root=root,
    )
    assert loaded == report
    assert authorized_index == index
    report["original_v19x_authorized"] = True
    write_json(report_path, report)
    with pytest.raises(RuntimeError, match="obsolete V19X"):
        adapter.authorize_unified_index(
            report_path,
            expected_config_sha256="a" * 64,
            expected_runner_sha256="b" * 64,
            expected_cells=1,
            expected_products=4,
            root=root,
        )


def test_terminal_authorization_accepts_v19w5_status_only_when_declared(
    tmp_path: Path,
) -> None:
    root = tmp_path / "root"
    index = root / "results" / "index.csv"
    index.parent.mkdir(parents=True)
    index.write_text("fixture\n", encoding="utf-8")
    report = {
        "status": adapter.V19W5_AUTHORIZED_STATUS,
        "config_sha256": "c" * 64,
        "runner_sha256": "d" * 64,
        "gates": {"all": True},
        "unified_cells": 1,
        "unified_product_files": 4,
        "base_v19w_archive_modified": False,
        "original_v19x_authorized": False,
        "v19x_successor_configuration_may_be_frozen": True,
        "unified_product_index": {
            "path": "results/index.csv",
            "rows": 1,
            "bytes": index.stat().st_size,
            "sha256": sha256(index),
        },
    }
    report_path = root / "report.json"
    write_json(report_path, report)
    with pytest.raises(RuntimeError, match="status does not authorize"):
        adapter.authorize_unified_index(
            report_path,
            expected_config_sha256="c" * 64,
            expected_runner_sha256="d" * 64,
            expected_cells=1,
            expected_products=4,
            root=root,
        )
    loaded, authorized_index = adapter.authorize_unified_index(
        report_path,
        expected_config_sha256="c" * 64,
        expected_runner_sha256="d" * 64,
        expected_cells=1,
        expected_products=4,
        root=root,
        expected_status=adapter.V19W5_AUTHORIZED_STATUS,
        authority_label="V19W5",
    )
    assert loaded == report
    assert authorized_index == index
