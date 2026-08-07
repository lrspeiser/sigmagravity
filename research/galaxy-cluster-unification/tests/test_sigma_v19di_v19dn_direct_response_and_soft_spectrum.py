from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from astropy.io import fits

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from sigma_v19dk_fits_canonical import canonicalize_fits


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(relative: str) -> dict:
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def test_all_frozen_runner_hashes_are_exact() -> None:
    stems = (
        "sigma_v19di_direct_ogip_writer_preflight",
        "sigma_v19dj_direct_response_commissioning",
        "sigma_v19dk_fits_canonicalization_preflight",
        "sigma_v19dk2_grouped_canonicalization_preflight",
        "sigma_v19dl_canonicalized_direct_response_commissioning",
        "sigma_v19dm_minimal_thermal_mixture_diagnostic",
        "sigma_v19dm2_statistic_parity_remediation",
        "sigma_v19dn_integrated_residual_localization",
        "sigma_v19do_observation_resolved_soft_background_audit",
        "sigma_v19do2_backscal_ratio_remediation",
    )
    for stem in stems:
        config = load(f"configs/{stem}.json")
        assert config["freeze_state"].startswith("frozen_")
        assert "provisional" not in config["freeze_state"]
        runner = ROOT / config["implementation"]["runner"]
        assert sha256(runner) == config["implementation"]["runner_sha256"]


def test_canonicalizer_removes_volatile_cards_and_is_byte_deterministic(
    tmp_path: Path,
) -> None:
    values = np.array([1.5, 2.5, 4.0], dtype=np.float32)
    paths = [tmp_path / "first.fits", tmp_path / "second.fits"]
    for index, path in enumerate(paths):
        primary = fits.PrimaryHDU()
        primary.header["DATE"] = f"2026-08-0{index + 1}T00:00:00"
        primary.header.add_history(f"volatile execution {index}")
        table = fits.BinTableHDU.from_columns(
            [fits.Column(name="VALUE", format="E", array=values)]
        )
        table.header["DATE"] = f"2026-08-0{index + 1}T00:00:01"
        table.header.add_history(f"volatile table {index}")
        fits.HDUList([primary, table]).writeto(path, checksum=True)
        canonicalize_fits(path, "SIGMA V19DK stable test history")

    assert sha256(paths[0]) == sha256(paths[1])
    with fits.open(paths[0], checksum=True) as hdus:
        assert np.array_equal(hdus[1].data["VALUE"], values)
        for hdu in hdus:
            assert hdu.verify_checksum() == 1
            assert hdu.verify_datasum() == 1
            assert "2026-08-0" not in str(hdu.header)


def test_direct_writer_and_canonicalization_controls_pass() -> None:
    writer = load("results/sigma_v19di_direct_ogip_writer_preflight/report.json")
    assert writer["aggregate_pass"] is True
    assert writer["spectrum_fitted"] is False
    canonical = load(
        "results/sigma_v19dk2_grouped_canonicalization_preflight/report.json"
    )
    assert canonical["aggregate_pass"] is True
    assert canonical["full_response_commissioning_successor_authorized"] is True
    for control in canonical["controls"]:
        assert control["passed"] is True
        assert all(control["byte_identical_across_independent_runs"].values())
        assert all(run["passed"] for run in control["runs"])


def test_full_commissioning_fails_only_the_integrated_spectral_gate() -> None:
    report = load(
        "results/sigma_v19dl_canonicalized_direct_response_commissioning/"
        "report.json"
    )
    expected_true = {
        "v19w5_unified_archive_and_every_product_hash_exact",
        "combination_uses_every_registered_cell_exactly_once",
        "combined_source_background_arf_and_rmf_exist_and_links_are_exact",
        "every_cell_event_energy_counts_equal_manifest",
        "combined_full_pha_source_counts_conserved_exactly",
        "both_regional_fits_pass",
        "every_snapshot_canonicalized_with_valid_checksums",
    }
    assert all(report["gates"][key] for key in expected_true)
    assert report["gates"]["both_integrated_fits_pass"] is False
    assert report["validated_response_cells"] == 5082
    by_cluster = {row["cluster"]: row for row in report["integrated_fits"]}
    assert by_cluster["ABELL2146"]["gates"]["all_passed"] is True
    assert by_cluster["BULLET"]["gates"]["all_passed"] is False
    assert np.isclose(
        by_cluster["BULLET"]["fit"]["reduced_statistic"], 2.793669223725117
    )
    assert report["full_494_region_combination_and_fit_authorized"] is False


def test_statistic_remediation_changes_no_scientific_choice() -> None:
    original = load("configs/sigma_v19dm_minimal_thermal_mixture_diagnostic.json")
    remediation = load("configs/sigma_v19dm2_statistic_parity_remediation.json")
    for key in (
        "model",
        "starts",
        "one_temperature_free_parameters",
        "two_temperature_free_parameters",
        "minimum_temperature_ratio",
        "minimum_normalization_fraction",
        "maximum_delta_bic_for_admission",
    ):
        assert remediation["mixture"][key] == original["mixture"][key]
    assert remediation["remediation"]["expected_statistic"] == "chi2xspecvar"
    invalid = load(
        "results/sigma_v19dm_minimal_thermal_mixture_diagnostic/report.json"
    )
    valid = load(
        "results/sigma_v19dm2_statistic_parity_remediation/report.json"
    )
    assert invalid["aggregate_pass"] is False
    assert valid["v19dm_execution_result_scientifically_discarded"] is True
    assert valid["aggregate_pass"] is False
    assert valid["minimal_adequate_full_regional_successor_authorized"] is False


def test_valid_two_temperature_result_rejects_bullet_repair() -> None:
    report = load(
        "results/sigma_v19dm2_statistic_parity_remediation/report.json"
    )
    rows = {row["cluster"]: row for row in report["integrated_model_selection"]}
    assert rows["ABELL2146"]["selection"]["model"] == "one_temperature"
    assert rows["BULLET"]["selection"]["model"] == "none"
    bullet = rows["BULLET"]["two_temperature"]
    assert bullet["statistic_name"] == "chi2xspecvar"
    assert np.isclose(bullet["reduced_statistic"], 2.802293013626455)
    assert np.isclose(bullet["delta_bic_vs_one_temperature"], 10.416390803080048)
    assert bullet["admitted_when_one_temperature_fails"] is False


def test_residual_localization_is_soft_dominated_and_remains_diagnostic() -> None:
    report = load(
        "results/sigma_v19dn_integrated_residual_localization/report.json"
    )
    assert report["status"] == "integrated_residual_localization_completed"
    rows = {
        (row["cluster"], row["band_id"]): row["fit"]
        for row in report["band_fits"]
    }
    assert np.isclose(
        rows[("BULLET", "soft_only_0p5_2")]["fit"]["reduced_statistic"],
        4.162264374664096,
    )
    assert np.isclose(
        rows[("BULLET", "hard_only_2_7")]["fit"]["reduced_statistic"],
        1.2200300017923373,
    )
    assert rows[("BULLET", "soft_only_0p5_2")]["gates"][
        "all_free_parameters_strictly_inside_bounds"
    ] is False
    assert report["full_regional_successor_authorized"] is False
    assert report["all_494_regions_run"] is False
    assert report["thermal_stress_or_baroclinicity_constructed"] is False
    assert report["lensing_halo_action_gravity_or_holdout_payload_opened"] is False
    assert report["gravity_formula_or_parameter_changed"] is False


def test_invalid_backscal_equality_run_is_retained_and_discarded() -> None:
    report = load(
        "results/sigma_v19do_observation_resolved_soft_background_audit/report.json"
    )
    assert report["status"] == (
        "observation_resolved_soft_background_audit_execution_failed"
    )
    assert report["execution_exception"].startswith(
        "RuntimeError: V19DO BACKSCAL mismatch:"
    )
    remediation = load("configs/sigma_v19do2_backscal_ratio_remediation.json")
    assert remediation["remediation"]["retained_scale"] == (
        "source_EXPOSURE/background_EXPOSURE * "
        "source_BACKSCAL/background_BACKSCAL * "
        "source_AREASCAL/background_AREASCAL"
    )


def test_backscal_ratio_audit_covers_every_cell_and_reconstructs_counts() -> None:
    report = load("results/sigma_v19do2_backscal_ratio_remediation/report.json")
    assert report["status"] == (
        "backscal_ratio_observation_soft_background_audit_completed"
    )
    assert report["aggregate_pass"] is True
    assert all(report["gates"].values())
    assert report["cell_audit"]["rows"] == 5082
    cell_audit = ROOT / report["cell_audit"]["path"]
    assert sha256(cell_audit) == report["cell_audit"]["sha256"]
    assert report["integrated_count_checks"]["BULLET"]["exact"] is True
    assert report["integrated_count_checks"]["ABELL2146"]["exact"] is True
    assert len({row["sha256"] for row in report["response_ebounds_controls"]}) == 1


def test_soft_background_is_small_and_not_the_frozen_heterogeneity_case() -> None:
    report = load("results/sigma_v19do2_backscal_ratio_remediation/report.json")
    bullet = report["cluster_summary"]["BULLET"]
    abell = report["cluster_summary"]["ABELL2146"]
    assert np.isclose(
        bullet["soft_0p5_2"]["background_fraction_of_source"],
        0.020373598542600426,
    )
    assert np.isclose(
        abell["soft_0p5_2"]["background_fraction_of_source"],
        0.018918810451318154,
    )
    assert bullet["hard_2_7"]["background_fraction_of_source"] > bullet[
        "soft_0p5_2"
    ]["background_fraction_of_source"]
    for cluster in ("BULLET", "ABELL2146"):
        interpretation = report["interpretations"][cluster]
        assert interpretation["aggregate_soft_background_regime"] == (
            "source_dominated"
        )
        assert interpretation["strong_observation_heterogeneity"] is False
    assert report["next_test"] == (
        "spatially_resolved_joint_plasma_likelihood_with_unmerged_responses"
    )
    assert report["joint_likelihood_or_full_regional_successor_authorized"] is False
    assert report["additional_plasma_component_admitted"] is False
    assert report["thermal_stress_or_baroclinicity_constructed"] is False
    assert report["lensing_halo_action_gravity_or_holdout_payload_opened"] is False
    assert report["gravity_formula_or_parameter_changed"] is False
