from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "acquire_sigma_v19ar_current_archive_decam_cutouts.py"
CONFIG = ROOT / "configs" / "sigma_v19ar_current_archive_decam_cutouts.json"
REPORT = ROOT / "results" / "sigma_v19ar_current_archive_decam_cutouts" / "report.json"
SPEC = importlib.util.spec_from_file_location("sigma_v19ar_results", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def test_completed_report_and_manifest_pass_every_gate() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    assert report["status"] == "completed_current_archive_group_acquisition"
    assert report["counts"]["groups"] == 139
    assert report["counts"]["measurement_memberships"] == 1032
    assert report["counts"]["unique_exposures"] == 82
    assert report["counts"]["unique_archive_files"] == 82
    assert report["counts"]["download_bytes"] == 733527360
    assert report["counts"]["checksum_keyword_present"] == 0
    assert report["counts"]["datasum_keyword_present"] == 0
    assert report["minimum_finite_pixel_fraction"] == 1.0
    assert report["minimum_anchor_edge_margin_pixel"] == 22.444585
    assert all(report["gates"].values())
    manifest = ROOT / report["download_manifest"]
    assert digest(manifest) == report["download_manifest_sha256"]
    assert digest(ROOT / config["frozen_resolved_plan"]["path"]) == config[
        "frozen_resolved_plan"
    ]["sha256"]


def test_every_raw_payload_matches_the_completed_manifest() -> None:
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    rows = read_rows(ROOT / report["download_manifest"])
    assert len(rows) == 139
    assert sum(int(row["measurement_rows"]) for row in rows) == 1032
    assert len({row["output_path"] for row in rows}) == 139
    assert len({row["sha256"] for row in rows}) == 139
    assert sum(int(row["download_bytes"]) for row in rows) == 733527360
    for row in rows:
        payload = ROOT / row["output_path"]
        assert payload.stat().st_size == int(row["download_bytes"])
        assert digest(payload) == row["sha256"]
        assert row["fits_structure_passed"] == "True"
        assert row["wcs_celestial"] == "True"
        assert row["anchors_contained"] == "True"
        assert row["returned_extname"] == row["header_extname"]
        assert row["returned_ccdnum"] == row["header_ccdnum"]


def test_every_payload_reopens_and_reproduces_structural_observables() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    rows = read_rows(ROOT / report["download_manifest"])
    groups = MODULE.AQ.AP.AO.parent_groups(config)
    for row in rows:
        inspection = MODULE.AQ.inspect_payload(
            (ROOT / row["output_path"]).read_bytes(),
            groups[(row["exposure"], row["sia_extension"])],
            row,
            float(config["retrieval"]["wcs_containment_tolerance_pixel"]),
        )
        assert inspection["fits_structure_passed"] is True
        assert inspection["anchors_contained"] is True
        assert inspection["returned_extname"] == row["returned_extname"]
        assert inspection["returned_ccdnum"] == row["returned_ccdnum"]
        assert inspection["finite_pixel_fraction"] == row["finite_pixel_fraction"]
        assert inspection["minimum_anchor_edge_margin_pixel"] == row[
            "minimum_anchor_edge_margin_pixel"
        ]
