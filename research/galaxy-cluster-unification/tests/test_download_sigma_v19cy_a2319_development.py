import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = ROOT / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import download_sigma_v19cy_a2319_development as download


def test_frozen_download_inputs_match_inventory() -> None:
    config, report, manifest = download.validate_frozen_inputs(
        download.DEFAULT_CONFIG, download.DEFAULT_REPORT
    )
    assert config["maximum_workers"] == 4
    assert report["manifest"]["rows"] == 197
    assert download.sha256(manifest) == report["manifest"]["sha256"]


def test_manifest_resolves_all_destinations_below_raw_root(tmp_path: Path) -> None:
    _, report, manifest = download.validate_frozen_inputs(
        download.DEFAULT_CONFIG, download.DEFAULT_REPORT
    )
    jobs = download.read_jobs(manifest, tmp_path / "raw")
    assert len(jobs) == report["manifest"]["rows"] == 197
    assert sum(job["expected_bytes"] for job in jobs) == 12_742_865_194
    resolved_root = (tmp_path / "raw").resolve()
    assert all(job["path"].is_relative_to(resolved_root) for job in jobs)


def test_manifest_path_traversal_is_rejected(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.csv"
    fields = [
        "asset_group",
        "role",
        "obsid",
        "relative_path",
        "download_path",
        "url",
        "bytes",
        "last_modified",
        "etag",
        "content_type",
    ]
    with manifest.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerow(
            {
                "asset_group": "test",
                "role": "test",
                "obsid": "",
                "relative_path": "escape",
                "download_path": "../escape",
                "url": "https://example.test/escape",
                "bytes": 1,
                "last_modified": "",
                "etag": "",
                "content_type": "application/octet-stream",
            }
        )
    try:
        download.read_jobs(manifest, tmp_path / "raw")
    except RuntimeError as error:
        assert "escapes raw root" in str(error)
    else:
        raise AssertionError("path traversal was accepted")


def test_existing_exact_size_file_is_reused_and_hashed(tmp_path: Path) -> None:
    path = tmp_path / "asset.bin"
    path.write_bytes(b"sigma")
    record = download.download_one(
        {
            "asset_group": "test",
            "role": "test",
            "obsid": "",
            "relative_path": "asset.bin",
            "download_path": "asset.bin",
            "url": "https://example.test/asset.bin",
            "expected_bytes": 5,
            "expected_etag": "",
            "path": path,
        }
    )
    assert record["reused"]
    assert not record["resumed"]
    assert record["sha256"] == "38de90475bb334fb3dea5d54f250500aba60fe2c6158115d342b06bcb46e39bf"


def test_disk_preflight_preserves_a_large_reserve(tmp_path: Path) -> None:
    preflight = download.ensure_free_space(tmp_path, 1)
    assert preflight["needed_bytes"] == 1
    assert preflight["reserve_bytes"] == 5 * 1024**3


def test_terminal_development_download_provenance_is_exact() -> None:
    provenance_path = (
        ROOT
        / "results"
        / "sigma_v19cy_direct_icm_velocity_evidence"
        / "development_download_provenance.json"
    )
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    assert provenance["status"] == (
        "all_frozen_a2319_development_payloads_downloaded_size_verified_and_sha256_hashed"
    )
    assert provenance["files"] == 197
    assert provenance["bytes"] == 12_742_865_194
    assert provenance["by_asset_group"] == {
        "caldb": {"bytes": 1_780_998_985, "files": 3},
        "chandra_ssm": {"bytes": 265_771_249, "files": 62},
        "official_gain_report": {"bytes": 4_183_922, "files": 1},
        "xrism_calibration_predecessor": {"bytes": 1_567_139_641, "files": 15},
        "xrism_science": {"bytes": 9_124_771_397, "files": 116},
    }
    assert len(provenance["records"]) == 197
    assert all(len(record["sha256"]) == 64 for record in provenance["records"])
    assert not provenance["validation_or_holdout_asset_accessed"]
    assert not provenance["lensing_halo_or_gravity_payload_opened"]
    assert not provenance["scientific_velocity_fit_performed"]
    assert provenance["validation_and_holdout_outcome_seals_preserved"]
