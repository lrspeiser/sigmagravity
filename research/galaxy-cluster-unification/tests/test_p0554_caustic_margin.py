import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "p0554_caustic_margin_protocol.json"
RESULTS = ROOT / "results" / "p0554_caustic_margin"


def load_report():
    return json.loads((RESULTS / "report.json").read_text(encoding="utf-8"))


def target_image():
    diagnostics = pd.read_csv(RESULTS / "image_diagnostics.csv")
    return diagnostics[
        diagnostics.system_label.eq("MACS1931")
        & diagnostics.image_id.eq("2c")
    ].copy()


def test_protocol_was_frozen_before_any_multistart_or_margin_score():
    protocol = json.loads(CONFIG.read_text(encoding="utf-8"))
    assert protocol["status"].startswith("frozen_")
    assert "before_any_multistart_or_caustic_margin_score" in protocol["status"]
    assert protocol["evaluation"]["formula_parameters_fit"] == 0
    assert protocol["evaluation"]["geometry_parameters_refit"] == 0
    assert protocol["evaluation"]["old_root_status_used_to_define_margin"] is False


def test_result_coverage_is_complete():
    report = load_report()
    assert report["report_version"] == "P0554-CAUSTIC-MARGIN-RESULTS-0.2.0"
    assert report["status"] == "complete"
    assert report["coverage"] == {
        "variants": 18,
        "raw_clusters": 5,
        "heldout_formula_image_rows": 324,
        "old_one_seed_failures": 7,
        "global_target_system": "MACS1931",
        "global_target_family": 2,
        "global_formula_searches": 18,
        "global_unique_roots_total": 76,
        "SPARC_galaxies": 131,
        "CLASH_systems": 20,
    }
    assert len(pd.read_csv(RESULTS / "image_diagnostics.csv")) == 324
    assert len(pd.read_csv(RESULTS / "global_MACS1931_family2_roots.csv")) == 76
    assert len(pd.read_csv(RESULTS / "global_MACS1931_family2_assignments.csv")) == 54


def test_multistart_recovers_every_old_failure_without_promoting_a_formula():
    report = load_report()
    assert report["multistart"] == {
        "old_failures_recovered": 7,
        "old_failures": 7,
        "all_old_failures_recovered": True,
    }
    failures = target_image().query("not old_root_converged")
    assert len(failures) == 7
    assert failures.local_multistart_root_found.astype(bool).all()
    assert report["verdict"]["no_formula_promoted"] is True


def test_global_search_finds_required_images_and_an_exact_extra_pair_bifurcation():
    report = load_report()
    multiplicity = report["global_multiplicity"]
    assert multiplicity["all_formulas_have_sufficient_unique_roots"] is True
    assert multiplicity["all_observed_family_images_assigned"] is True
    assert multiplicity["predeclared_observed_multiplicity_insufficiency_detected"] is False
    assert multiplicity["descriptive_extra_root_pair_bifurcation_detected"] is True
    assert multiplicity["extra_pair_near_image_2c_matches_old_status"] is True
    assert multiplicity["minimum_unique_roots"] == 3
    assert multiplicity["maximum_unique_roots"] == 5
    assert multiplicity["observed_family_images"] == 3
    assert multiplicity["old_success_unique_root_counts"] == [5]
    assert multiplicity["old_failure_unique_root_counts"] == [3]

    summary = pd.read_csv(RESULTS / "variant_summary.csv")
    status = target_image()[["variant_id", "old_root_converged"]]
    joined = summary.merge(status, on="variant_id", validate="one_to_one")
    successful = joined[joined.old_root_converged.astype(bool)]
    failed = joined[~joined.old_root_converged.astype(bool)]
    assert len(successful) == 11
    assert len(failed) == 7
    assert successful.unique_global_roots.eq(5).all()
    assert failed.unique_global_roots.eq(3).all()
    assert successful.global_roots_within_15_arcsec_of_2c.eq(2).all()
    assert failed.global_roots_within_15_arcsec_of_2c.eq(0).all()


def test_extra_pair_is_near_2c_while_three_root_regime_is_far():
    summary = pd.read_csv(RESULTS / "variant_summary.csv")
    status = target_image()[["variant_id", "old_root_converged"]]
    joined = summary.merge(status, on="variant_id", validate="one_to_one")
    successful = joined[joined.old_root_converged.astype(bool)]
    failed = joined[~joined.old_root_converged.astype(bool)]
    assert successful.nearest_global_root_to_2c_arcsec.max() < 6.0
    assert successful.second_nearest_global_root_to_2c_arcsec.max() < 12.0
    assert successful.nearest_two_root_separation_arcsec.between(1.7, 11.0).all()
    assert failed.nearest_global_root_to_2c_arcsec.min() > 24.0


def test_branch_caustic_margin_separates_regimes_but_observed_jacobian_does_not():
    target = target_image()
    successful = target[target.old_root_converged.astype(bool)]
    failed = target[~target.old_root_converged.astype(bool)]
    assert successful.source_caustic_margin_arcsec.max() < 1.0
    assert failed.source_caustic_margin_arcsec.min() > 4.0
    assert successful.local_nearest_root_distance_arcsec.max() < 6.0
    assert failed.local_nearest_root_distance_arcsec.min() > 24.0
    assert target.observed_minimum_singular_step_span.max() < 1.0e-9

    discrimination = pd.DataFrame(load_report()["discrimination"]).set_index("metric")
    assert discrimination.loc["source_caustic_margin_arcsec", "absolute_AUC"] == 1.0
    assert discrimination.loc["local_nearest_root_distance_arcsec", "absolute_AUC"] == 1.0
    assert np.isclose(
        discrimination.loc["observed_abs_determinant", "absolute_AUC"],
        0.5454545454545454,
    )
    assert discrimination.loc["observed_minimum_singular_value", "absolute_AUC"] < 0.5


def test_corrected_verdict_and_cross_domain_controls_are_preserved():
    report = load_report()
    assert report["verdict"] == {
        "predeclared_observed_multiplicity_failure": False,
        "prior_root_recovery_tracks_a_real_extra_pair_bifurcation": True,
        "one_seed_status_is_not_merely_a_numerical_artifact": True,
        "caustic_margin_perfectly_discriminates_the_extra_pair_regime": True,
        "no_formula_promoted": True,
    }
    controls = pd.DataFrame(report["cross_domain_controls"]).set_index("variant_id")
    assert len(controls) == 18
    assert controls.all_solar_proxies_pass.astype(bool).all()
    assert (
        controls.loc["route_parent", "galaxy_outer_RMSE_km_s"]
        == controls.loc["baseline", "galaxy_outer_RMSE_km_s"]
    )
    assert (
        controls.loc["route_parent", "cluster_RMSE_dex"]
        == controls.loc["baseline", "cluster_RMSE_dex"]
    )
