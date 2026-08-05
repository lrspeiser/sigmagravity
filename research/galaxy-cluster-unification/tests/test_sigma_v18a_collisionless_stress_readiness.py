from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "sigma_v18a_collisionless_stress_readiness.json"
REPORT = ROOT / "results" / "sigma_v18a_collisionless_stress_readiness" / "report.json"
SCRIPT = ROOT / "scripts" / "audit_sigma_v18a_collisionless_stress_readiness.py"


def load_module():
    spec = importlib.util.spec_from_file_location("sigma_v18a_readiness", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_inherited_gate_is_not_relaxed() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    parent = json.loads((ROOT / config["parent_gate"]).read_text(encoding="utf-8"))
    inherited = parent["stage_B_collisionless_member_stress"]["required_for_each_cluster"]
    assert inherited["minimum_secure_members_inside_1_8_Mpc"] == 50
    assert config["audit_rules"]["minimum_unique_secure_members"] == 50
    assert config["audit_rules"]["member_projected_aperture_kpc"] == 1800.0
    assert config["audit_rules"]["minimum_is_inherited_unchanged"] is True
    assert config["audit_rules"]["formula_or_kernel_selection_authorized"] is False
    assert config["audit_rules"]["lensing_target_access_authorized"] is False


def test_catalog_deduplication_is_one_to_one() -> None:
    module = load_module()
    left = [
        {"RAJ2000": "10.0", "DEJ2000": "-50.0", "z": "0.3", "Galaxy": "a"},
        {"RAJ2000": "11.0", "DEJ2000": "-50.0", "z": "0.3", "Galaxy": "b"},
    ]
    right = [
        {"RAJ2000": "10.0", "DEJ2000": "-50.0", "z": "0.3", "Gal": "x"},
    ]
    matches = module.one_to_one_matches(left, right, 1.0)
    assert len(matches) == 1
    assert matches[0]["ruel_index"] == 0
    assert matches[0]["bayliss_index"] == 0


def test_report_proves_public_samples_are_duplicates_and_gate_fails() -> None:
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    assert report["status"] == (
        "completed Sigma v18A AS295 collisionless-stress readiness audit"
    )
    assert report["catalogs"]["ruel_2014"]["spectra_rows"] == 39
    assert report["catalogs"]["ruel_2014"]["published_member_count"] == 30
    assert report["catalogs"]["bayliss_2016"]["spectra_rows"] == 38
    assert report["catalogs"]["bayliss_2016"]["published_member_count"] == 29
    assert report["deduplication"]["matched_spectra"] == 38
    assert report["deduplication"]["combined_unique_spectra"] == 39
    assert report["deduplication"]["combined_unique_fixed_window_members"] == 30
    assert report["deduplication"]["member_projected_aperture_kpc"] == 1800.0
    assert report["frozen_member_requirement"] == 50
    assert report["member_shortfall"] == 20
    assert report["stage_b_source_construction_authorized"] is False
    assert report["formula_or_kernel_selection_authorized"] is False
    assert report["lensing_target_opened"] is False


def test_report_hashes_all_inputs() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    assert report["input_hashes"]["config"] == digest(CONFIG)
    assert report["input_hashes"]["parent_gate"] == digest(ROOT / config["parent_gate"])
    for name, source in config["public_inputs"].items():
        assert report["input_hashes"][name] == digest(ROOT / source["raw_path"])
