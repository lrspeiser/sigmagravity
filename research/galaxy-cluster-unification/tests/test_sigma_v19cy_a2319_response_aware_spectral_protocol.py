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
    assert config["protocol_version"].endswith("1.0.5")
    assert "NXB constraints were refrozen" in config["status"]
    assert ";" in config["runtime"]["pfiles"]
    assert config["closed_failure_history"]["response_or_background_generated"] is False
    amendment = config["pre_response_interface_amendment"]
    assert amendment["response_or_background_generated"] is False
    assert amendment["validation_or_holdout_accessed"] is False
    nxb_amendment = config["pre_nxb_interface_amendment"]
    assert nxb_amendment["nxb_or_rmf_or_arf_generated"] is False
    assert nxb_amendment["source_count_probe"]["selected_events"] == 4941
    assert nxb_amendment["source_count_probe"]["pha_counts"] == 4941
    fit_amendment = config["pre_fit_statistical_amendment"]
    assert fit_amendment["source_spectrum_contract"]["statistic"] == "cstat"
    assert fit_amendment["nxb_spectrum_contract"]["statistic"] == "chi standard"
    assert fit_amendment["source_energy_distribution_summarized_or_fit"] is False
    assert fit_amendment["validation_or_holdout_accessed"] is False
    grouping_amendment = config["pre_fit_grouping_amendment"]
    assert grouping_amendment["previous_version"].endswith("1.0.4")
    assert grouping_amendment["arf_generation_config_sha256"] == (
        "fb164dc7eb0b9fecedc0dfddf3f6c4e625d08e9470893a5c34f635811a9a9c27"
    )
    arf_report = ROOT / (
        "results/sigma_v19cy_direct_icm_velocity_evidence/"
        "development_response_arfs.json"
    )
    assert sha256(arf_report) == grouping_amendment["arf_report_sha256"]
    assert grouping_amendment["nxb_grouping"]["grouptype"] == "optsnmin"
    assert grouping_amendment["nxb_grouping"]["groupscale"] == pytest.approx(3.0)
    assert grouping_amendment["ten_product_preflight"]["products_checked"] == 10
    assert grouping_amendment["source_energy_distribution_summarized_or_fit"] is False
    assert grouping_amendment["velocity_fit_performed"] is False
    assert grouping_amendment["validation_or_holdout_accessed"] is False
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
    assert "DATE-OBS" in config["rmf_protocol"]["caldb_time"]
    assert config["nxb_protocol"]["sortcol"] == "CORTIME"
    assert config["nxb_protocol"]["sortbin"] == "6,8,10,12,99"
    assert config["nxb_protocol"]["timefirst_days"] == 300
    assert config["nxb_protocol"]["timelast_days"] == 300
    assert config["nxb_protocol"]["timefirst_parameter"] == "-300"
    assert config["nxb_protocol"]["timelast_parameter"] == "+300"
    assert "detector-region" in config["nxb_protocol"]["regfile"]
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
    assert config["attitude_and_arf_protocol"]["xaarfgen"]["source_ra_deg"] == pytest.approx(290.299)
    assert config["attitude_and_arf_protocol"]["xaarfgen"]["source_dec_deg"] == pytest.approx(43.9345)
    for executable in ("xrtraytrace", "heasim", "xaxmaarfgen", "aharfgen"):
        assert len(config["runtime"]["executables_sha256"][executable]) == 64


def test_fit_is_one_response_aware_physical_model_with_two_robustness_checks():
    config = load()
    fit = config["fit_protocol"]
    assert fit["primary_model"].startswith("tbabs*bapec")
    assert fit["primary_band_keV"] == [3.0, 9.5]
    assert fit["statistic"] == "mixed simultaneous likelihood; no background subtraction"
    assert fit["source_statistic"] == "cstat"
    assert fit["nxb_statistic"] == "chi standard"
    assert fit["nxb_constraint_band_keV"] == [1.0, 17.0]
    assert fit["data_group_protocol"].startswith(
        "For every branch-region product, load ungrouped source COUNTS"
    )
    assert config["nxb_protocol"]["grouping"] == {
        "source": "none",
        "nxb_grouptype": "optsnmin",
        "nxb_groupscale": 3.0,
        "nxb_grid_channels": 60000,
    }
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
    assert gate["require_all_ten_nxb_grouping_commands_exit_zero"] is True
    assert gate["require_nxb_grouping_zero_variance_groups_in_band"] == 0
    assert gate["minimum_nxb_group_signal_to_noise_in_band"] == pytest.approx(3.0)
    assert gate["require_all_seven_primary_fits_converged"] is True
    assert gate["minimum_regions_meeting_velocity_interval_gate"] == 5
    assert gate["minimum_robust_regions"] == 5
    assert "does not validate Sigma gravity" in gate["decision"]
