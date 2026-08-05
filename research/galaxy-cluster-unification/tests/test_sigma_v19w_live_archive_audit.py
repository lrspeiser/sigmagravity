from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "audit_sigma_v19w_live_archive.py"
SPEC = importlib.util.spec_from_file_location("sigma_v19w_live_audit", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fixture(tmp_path: Path) -> tuple[Path, dict[str, str]]:
    manifest = {
        "cluster": "TEST",
        "bin_id": "7",
        "obsid": "42",
        "ccd_id": "3",
        "source_band_events": "11",
        "background_band_events": "5",
    }
    name = MODULE.cell_name(manifest)
    completed = tmp_path / "completed" / name
    products_dir = completed / "products"
    products_dir.mkdir(parents=True)
    products = {}
    filenames = {
        "source_pha": "source.pi",
        "background_pha": "background.pi",
        "arf": "source.arf",
        "rmf": "source.rmf",
    }
    total = 0
    for role, filename in filenames.items():
        path = products_dir / filename
        path.write_bytes((role + "-fixture").encode("utf-8"))
        products[role] = {
            "name": filename,
            "bytes": path.stat().st_size,
            "sha256": digest(path),
        }
        total += path.stat().st_size
    report = {
        "cell_name": name,
        "cluster": "TEST",
        "bin_id": 7,
        "obsid": 42,
        "ccd_id": 3,
        "attempt": 1,
        "preflight": {"source_band_events": 11, "background_band_events": 5},
        "source_pha_channel_audit": {"exact": True},
        "background_pha_channel_audit": {"exact": True},
        "response_audit": {
            "arf_finite": True,
            "arf_positive_bins": 10,
            "rmf_finite": True,
            "rmf_nonzero_elements": 20,
        },
        "source_pha_links": {
            "BACKFILE": "background.pi",
            "ANCRFILE": "source.arf",
            "RESPFILE": "source.rmf",
        },
        "four_product_bytes": total,
        "products": products,
        "gates": {"first": True, "second": True},
    }
    report_path = completed / "cell_report.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")
    return report_path, manifest


def test_valid_checkpoint_passes_without_mutation(tmp_path: Path):
    report_path, manifest = fixture(tmp_path)
    before = {path: digest(path) for path in report_path.parent.rglob("*") if path.is_file()}
    record = MODULE.validate_checkpoint(report_path, manifest)
    after = {path: digest(path) for path in report_path.parent.rglob("*") if path.is_file()}
    assert record["task_key"] == ("TEST", 7, 42, 3)
    assert record["attempt"] == 1
    assert before == after


def test_hash_or_manifest_count_change_fails_closed(tmp_path: Path):
    report_path, manifest = fixture(tmp_path)
    product = report_path.parent / "products" / "source.pi"
    product.write_bytes(b"changed")
    with pytest.raises(RuntimeError, match="resized|hash changed"):
        MODULE.validate_checkpoint(report_path, manifest)
    report_path, manifest = fixture(tmp_path / "second")
    manifest["source_band_events"] = "12"
    with pytest.raises(RuntimeError, match="source count changed"):
        MODULE.validate_checkpoint(report_path, manifest)


def test_failed_cell_gate_or_out_of_range_attempt_fails_closed(tmp_path: Path):
    report_path, manifest = fixture(tmp_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["gates"]["second"] = False
    report_path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(RuntimeError, match="gate failed"):
        MODULE.validate_checkpoint(report_path, manifest)
    report_path, manifest = fixture(tmp_path / "second")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["attempt"] = 3
    report_path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(RuntimeError, match="outside frozen range"):
        MODULE.validate_checkpoint(report_path, manifest)


def test_cell_name_round_trip_and_snapshot_digest_are_deterministic(tmp_path: Path):
    first, first_manifest = fixture(tmp_path / "first")
    second, second_manifest = fixture(tmp_path / "second")
    assert MODULE.cell_name(first_manifest) == "TEST_bin7_obs42_ccd3"
    assert MODULE.cell_name(second_manifest) == "TEST_bin7_obs42_ccd3"
    first_paths = [first, second]
    second_paths = [first, second]
    assert MODULE.snapshot_digest(first_paths) == MODULE.snapshot_digest(second_paths)
