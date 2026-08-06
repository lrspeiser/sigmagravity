import hashlib
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/sigma_v19cy_a2319_response_aware_spectral.json"


def load() -> dict:
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def test_protocol_is_frozen_and_sealed_targets_remain_forbidden():
    config = load()
    assert config["protocol_version"].endswith("1.0.0")
    assert "frozen before generating" in config["status"]
    authorization = config["authorization"]
    assert authorization["read_A2319_development_energy_columns"] is True
    assert authorization["access_A3667_validation"] is False
    assert authorization["access_A754_holdout"] is False
    assert authorization["open_lensing_halo_or_gravity_targets"] is False
    assert authorization["change_gravity_formula_or_parameters"] is False


def test_all_parent_hashes_match_current_terminal_reports():
    for parent in load()["parents"].values():
        assert sha256(ROOT / parent["path"]) == parent["sha256"]


def test_branch_partition_and_corrected_gti_contract_are_exact():
    config = load()
    branches = config["branches"]
    assert [branch["name"] for branch in branches] == [
        "000101_open_0_cross_obsid",
        "000101_open_1_cross_obsid",
        "000102_open_0_cross_obsid",
    ]
    assert sum(len(branch["regions"]) for branch in branches) == 10
    assert sum(branch["clipped_parent_gti_rows"] for branch in branches) == 39
    assert sum(branch["clipped_parent_gti_exposure_seconds"] for branch in branches) == pytest.approx(
        102387.83218416572
    )
    assert config["gti_protocol"]["exposure_definition"] == (
        "sum(STOP-START) over the final GTI"
    )
    for branch in branches:
        assert sha256(ROOT / branch["event_path"]) == branch["event_sha256"]


def test_regions_partition_all_science_pixels_without_overlap_per_pointing():
    config = load()
    region_pixels = config["region_pixels"]
    expected = set(range(36)) - {12, 27}
    for regions in (("a", "b", "d"), ("b_prime", "c_prime", "d_prime", "e_prime")):
        lists = [region_pixels[name] for name in regions]
        assert set().union(*map(set, lists)) == expected
        assert sum(map(len, lists)) == len(expected)


def test_response_and_background_settings_match_frozen_science_scope():
    config = load()
    assert config["rmf_protocol"]["whichrmf"] == "L"
    assert config["rmf_protocol"]["resolist"] == "0"
    assert config["nxb_protocol"]["sortcol"] == "CORTIME"
    assert config["nxb_protocol"]["sortbin"] == "6,8,10,12,99"
    assert config["nxb_protocol"]["timefirst_days"] == 300
    assert config["nxb_protocol"]["timelast_days"] == 300
    assert config["attitude_and_arf_protocol"]["xaexpmap"] == {
        "instrume": "RESOLVE",
        "badimgfile": "NONE",
        "outmaptype": "EXPOSURE",
        "delta_arcmin": 0.25,
        "numphi": 4,
        "maskcalsrc": True,
        "pixel_gti": "observation px1000 exposure GTI",
    }
    assert config["attitude_and_arf_protocol"]["xaarfgen"]["seed"] == 7
    assert config["attitude_and_arf_protocol"]["xaarfgen"]["numphoton"] == 600000


def test_fit_is_one_response_aware_physical_model_with_two_robustness_checks():
    config = load()
    fit = config["fit_protocol"]
    assert fit["primary_model"].startswith("tbabs*bapec")
    assert fit["primary_band_keV"] == [3.0, 9.5]
    assert fit["atomdb"]["version"] == "3.0.9"
    assert fit["abundance_table"] == "lodd"
    assert fit["nh_1e22_cm2_fixed"] == pytest.approx(0.112)
    assert [item["name"] for item in fit["robustness_models"]] == [
        "narrow_fe_k",
        "two_temperature_shared_velocity",
    ]


def test_terminal_gate_requires_all_regions_but_does_not_claim_gravity_validation():
    config = load()
    gate = config["terminal_gate"]
    assert gate["required_branch_region_products"] == 10
    assert gate["require_all_seven_primary_fits_converged"] is True
    assert gate["minimum_regions_meeting_velocity_interval_gate"] == 5
    assert gate["minimum_robust_regions"] == 5
    assert "does not validate Sigma gravity" in gate["decision"]
