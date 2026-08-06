import hashlib
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/sigma_v19cy_a2319_response_aware_spectral.json"
REPORT = ROOT / (
    "results/sigma_v19cy_direct_icm_velocity_evidence/"
    "development_response_aware_spectral.json"
)
INDEX = ROOT / (
    "results/sigma_v19cy_direct_icm_velocity_evidence/"
    "development_response_aware_spectral_artifacts.json"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_terminal_result_fails_only_after_all_development_fits_are_reported():
    report = load(REPORT)
    assert report["status"] == "response_aware_spectral_terminal_gate_failed"
    assert report["terminal_gate_passed"] is False
    assert report["signed_gas_current_constructed"] is False
    assert report["validation_or_holdout_accessed"] is False
    assert report["config_sha256"] == sha256(CONFIG)
    assert len(report["nxb_prefits"]) == 7
    assert len(report["fits"]) == 21
    assert sum(row["variant"] == "primary" for row in report["fits"]) == 7
    assert sum(row["converged"] for row in report["fits"]) == 20
    assert all(row["converged"] for row in report["nxb_prefits"])
    assert all(not row["source_spectra_loaded"] for row in report["nxb_prefits"])


def test_terminal_gates_preserve_the_public_nxb_bound_failure():
    gates = load(REPORT)["gates"]
    assert gates == {
        "all_seven_nxb_only_prefits_converged": True,
        "all_seven_primary_fits_converged": True,
        "all_ten_nxb_grouping_gates_passed": True,
        "arf_gate_passed": True,
        "at_least_five_primary_velocity_halfwidths_at_most_200_km_s": True,
        "at_least_five_regions_pass_both_robustness_models": True,
        "no_free_parameter_at_hard_bound_in_any_fit": False,
        "response_component_gate_passed": True,
    }


def test_primary_velocities_and_diagnostic_summary_are_frozen():
    report = load(REPORT)
    primary = {
        row["region"]: row
        for row in report["fits"]
        if row["variant"] == "primary"
    }
    expected = {
        "a": -78.48013715398505,
        "b": -89.06895584521452,
        "d": 0.08312620746830923,
        "b_prime": -224.71173610074098,
        "c_prime": -86.02216126819593,
        "d_prime": 55.806177319627736,
        "e_prime": 16.329317235656426,
    }
    for region, velocity in expected.items():
        assert primary[region]["velocity_km_s"] == pytest.approx(velocity)
        assert primary[region]["velocity_interval_halfwidth_km_s"] <= 200.0
    aggregate = report["published_no_ssm_comparison"]["aggregate"]
    assert aggregate[
        "inverse_combined_variance_weighted_rms_difference_km_s"
    ] == pytest.approx(61.34077319827083)
    assert aggregate["pearson_velocity_correlation"] == pytest.approx(
        0.695085987034409
    )
    assert aggregate["sign_agreement_fraction"] == pytest.approx(5 / 7)


def test_artifact_index_hashes_every_installed_product():
    index = load(INDEX)
    assert index["terminal_report_sha256"] == sha256(REPORT)
    assert index["config_sha256"] == sha256(CONFIG)
    assert index["artifact_count"] == 124
    assert index["total_bytes"] == 11_758_779
    assert index["validation_or_holdout_accessed"] is False
    assert len(index["artifacts"]) == index["artifact_count"]
    assert sum(row["bytes"] for row in index["artifacts"]) == index["total_bytes"]
    for row in index["artifacts"]:
        path = ROOT / row["path"]
        assert path.stat().st_size == row["bytes"]
        assert sha256(path) == row["sha256"]
