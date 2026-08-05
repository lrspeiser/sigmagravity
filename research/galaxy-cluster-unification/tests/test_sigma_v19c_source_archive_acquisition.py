from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "sigma_v19c_source_archive_acquisition.json"
MANIFEST = ROOT / "data" / "raw" / "sigma_v19c_assembly_sources" / "provenance.json"
SCRIPT = ROOT / "scripts" / "download_sigma_v19c_source_archives.py"


def load_module():
    spec = importlib.util.spec_from_file_location("sigma_v19c_download", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_acquisition_is_exactly_the_v19b_source_gate_pair() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    screen = json.loads(
        (ROOT / config["parents"]["replacement_screen_report"]).read_text(
            encoding="utf-8"
        )
    )
    assert sorted(config["selected_clusters"]) == sorted(
        screen["selected_development_pair"]
    )
    assert set(config["selected_clusters"]) == {"BULLET", "ABELL2146"}


def test_acquisition_excludes_every_lensing_payload() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    forbidden = set(config["explicitly_forbidden_arxiv_ids"])
    assert forbidden == {"1209.0384", "1609.06765"}
    assert not any(asset["arxiv_id"] in forbidden for asset in config["assets"])
    assert all(not asset["contains_lensing_target_payload"] for asset in config["assets"])
    authorization = config["authorization"]
    assert authorization["download_lensing_paper_source_archives"] is False
    assert authorization["download_multiple_image_coordinates"] is False
    assert authorization["read_lens_models_or_inferred_halo_products"] is False


def test_frozen_assets_cover_members_and_shocks_for_both_clusters() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    for cluster in config["selected_clusters"]:
        assets = [asset for asset in config["assets"] if asset["cluster"] == cluster]
        roles = " ".join(asset["role"].lower() for asset in assets)
        assert len(assets) == 3
        assert "member" in roles
        assert "shock" in roles


def test_manifest_verifies_every_frozen_archive_and_parent() -> None:
    module = load_module()
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert manifest["asset_count"] == len(config["assets"])
    assert manifest["input_hashes"]["config"] == digest(CONFIG)
    for key in ("replacement_screen_config", "replacement_screen_report"):
        assert manifest["input_hashes"][key] == digest(ROOT / config["parents"][key])
    by_name = {row["filename"]: row for row in manifest["assets"]}
    for asset in config["assets"]:
        path = ROOT / config["output_root"] / asset["filename"]
        row = by_name[asset["filename"]]
        assert row["sha256"] == digest(path)
        assert row["bytes"] == path.stat().st_size
        assert row["archive_file_count"] > 0
        assert module.validate_download(path, config["minimum_archive_bytes"])
    assert manifest["all_replacement_lensing_targets_remained_sealed"] is True
    assert manifest["lensing_or_halo_payload_downloaded"] is False
    assert manifest["gravity_parameters_fit"] == 0
