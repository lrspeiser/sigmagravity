import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/p0554_route_localization_screen_protocol.json"
RESULTS = ROOT / "results/p0554_route_localization_screen"


def load_report():
    return json.loads((RESULTS / "report.json").read_text(encoding="utf-8"))


def test_protocol_was_frozen_before_shape_scores():
    protocol = json.loads(CONFIG.read_text(encoding="utf-8"))
    assert protocol["status"].startswith("frozen_")
    assert "before_any_direction_or_bend_variant_score" in protocol["status"]
    assert len(protocol["variants"]) == 12
    assert protocol["formula"]["new_per_cluster_gravity_parameters"] == 0
    assert protocol["formula"]["geometry_parameters_refit"] == 0


def test_complete_screen_coverage_and_field_invariants():
    report = load_report()
    assert report["coverage"] == {
        "variants": 12,
        "systems": 5,
        "variant_system_scores": 60,
        "images": 77,
        "source_families": 27,
        "route_fields": 55,
    }
    assert len(pd.read_csv(RESULTS / "system_scores.csv")) == 60
    assert len(pd.read_csv(RESULTS / "image_residuals.csv")) == 924
    assert len(pd.read_csv(RESULTS / "field_audits.csv")) == 55
    invariants = report["field_invariants"]
    assert invariants["maximum_route_map_normalization_error"] < 1e-14
    assert invariants["maximum_annular_convergence_error"] < 1e-12
    assert invariants["maximum_normalized_curl_RMS"] < 1e-12


def test_local_soft_200_is_frozen_screen_winner():
    scores = pd.read_csv(RESULTS / "scores.csv").set_index("variant_id")
    best = scores.primary_other_four_equal_system_RMS_arcsec.idxmin()
    assert best == "local_soft_200"
    assert np.isclose(
        scores.loc[best, "primary_improvement_fraction_vs_eta0"],
        0.008238510469249394,
    )
    assert int(scores.loc[best, "primary_systems_improved"]) == 4
    shortlist = pd.read_csv(RESULTS / "shortlist.csv")
    assert shortlist.variant_id.tolist() == ["local_soft_200"]
    assert shortlist.selection_reason.iloc[0] == "lowest_primary_RMS+most_direction_consistent"


def test_screen_shows_spent_macs1931_sign_conflict():
    scores = pd.read_csv(RESULTS / "scores.csv").set_index("variant_id")
    assert scores.loc["local_soft_200", "primary_improvement_fraction_vs_eta0"] > 0
    assert scores.loc["local_soft_200", "all_five_improvement_fraction_vs_eta0"] < 0
    systems = pd.read_csv(RESULTS / "system_scores.csv").pivot_table(
        index="variant_id", columns="system_label", values="heldout_linearized_RMS_arcsec"
    )
    assert systems.loc["local_soft_200", "MACS1931"] > systems.loc["eta_000", "MACS1931"]


def test_cross_domain_values_are_preserved_not_refit():
    controls = load_report()["cross_domain_preservation"]
    assert np.isclose(controls["galaxy_outer_RMSE_km_s"], 12.570912420381397)
    assert np.isclose(controls["CLASH_radial_RMSE_dex"], 0.19641371129844437)
    assert controls["all_solar_proxies_pass"]
    assert "identical for every localization" in controls["interpretation"]


def test_screen_does_not_promote_without_exact_roots():
    verdict = load_report()["verdict"]
    assert verdict["any_nonbaseline_improves_primary_aggregate"]
    assert verdict["any_nonbaseline_improves_at_least_three_primary_systems"]
    assert verdict["shortlist_requires_exact_refit_and_global_roots"]
    assert verdict["no_formula_promoted"]
