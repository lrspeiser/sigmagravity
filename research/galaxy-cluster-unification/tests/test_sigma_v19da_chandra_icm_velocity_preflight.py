from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "sigma_v19da_chandra_icm_velocity_preregistration.json"
RUNNER = ROOT / "scripts" / "check_sigma_v19da_chandra_icm_velocity_preflight.py"
OUTPUT = ROOT / "results" / "sigma_v19da_chandra_icm_velocity_preflight"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_runner():
    spec = importlib.util.spec_from_file_location("sigma_v19da_preflight", RUNNER)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def config() -> dict:
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def test_protocol_is_source_only_and_discloses_nonpristine_transfer() -> None:
    payload = config()
    assert payload["freeze_state"] == "frozen_before_any_v19da_pha_or_redshift_payload_access"
    assert payload["clusters"]["BULLET"]["role"] == "development_reproduction"
    assert payload["clusters"]["ABELL2146"]["role"] == "internally_sealed_nonpristine_transfer"
    assert "not a pristine" in payload["evidence_design"]["ABELL2146"]
    assert payload["authorization"]["open_abell2146_before_bullet_development_pass"] is False
    assert payload["authorization"]["open_lensing_halo_or_gravity_payload"] is False
    assert payload["authorization"]["derive_or_change_action_or_gravity_constants"] is False


def test_region_rule_is_identical_and_target_blind() -> None:
    payload = config()
    regionization = payload["regionization"]
    assert regionization["same_rule_and_targets_for_both_clusters"] is True
    assert regionization["net_count_targets_0p5_7_keV"] == {
        "primary_8000": 8000.0,
        "robustness_10000": 10000.0,
    }
    forbidden = " ".join(regionization["forbidden_region_information"]).lower()
    for token in ("redshift", "lensing", "halo", "gravity", "temperature"):
        assert token in forbidden


def test_runner_opens_only_the_spatial_binmap_fits() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    assert source.count("fits.getdata(") == 1
    assert "fits.getdata(binmap_path)" in source
    for forbidden_call in ("load_pha(", "unpack_pha(", "load_arf(", "load_rmf("):
        assert forbidden_call not in source


def test_source_only_build_conserves_every_bin(tmp_path: Path) -> None:
    runner = load_runner()
    report = runner.build(CONFIG, tmp_path, check_external=False)
    expected = {
        "BULLET": {"primary_8000": 43, "robustness_10000": 35},
        "ABELL2146": {"primary_8000": 16, "robustness_10000": 12},
    }
    for cluster, branches in expected.items():
        for branch, count in branches.items():
            observed = report["region_summary"][cluster]["branches"][branch]
            assert observed["regions"] == count
            assert observed["all_bins_used_once"] is True
            assert observed["all_regions_connected"] is True
            assert observed["minimum_net_counts"] >= observed["target_net_counts"]
    groups = pd.read_csv(tmp_path / "frozen_region_groups.csv")
    assert len(groups) == sum(sum(item.values()) for item in expected.values())
    assert groups["connected_by_construction"].all()
    assert groups["meets_target"].all()
    assert report["access_audit"]["source_or_background_pha_payload_opened"] is False
    assert report["access_audit"]["temperature_abundance_or_redshift_fit_opened"] is False


def test_committed_preflight_is_current_and_passes() -> None:
    payload = config()
    report = json.loads((OUTPUT / "report.json").read_text(encoding="utf-8"))
    assert report["decision"] == "passed_target_sealed_chandra_velocity_preflight"
    assert report["config_sha256"] == sha256(CONFIG)
    assert report["runner_sha256"] == sha256(RUNNER)
    assert payload["implementation"]["preflight_sha256"] == sha256(RUNNER)
    assert all(report["gates"].values())
    assert report["external_archive_audit"]["products_checked_by_metadata_only"] == 4 * 5082
    assert report["external_archive_audit"]["all_products_present_with_frozen_sizes"] is True
