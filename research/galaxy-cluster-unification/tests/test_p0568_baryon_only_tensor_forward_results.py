import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
P0568 = ROOT / "results" / "p0568_baryon_only_tensor_forward"
P0568B = ROOT / "results" / "p0568b_tensor_width_refinement"
P0568C = ROOT / "results" / "p0568c_width_coupling_interaction"


def read_report(directory):
    return json.loads((directory / "report.json").read_text(encoding="utf-8"))


def test_p0568_forward_screen_has_the_frozen_coverage():
    report = read_report(P0568)
    assert report["coverage"]["clusters"] == 10
    assert report["coverage"]["development_clusters"] == 7
    assert report["coverage"]["holdout_clusters"] == 3
    assert report["coverage"]["tensor_families"] == 9
    assert report["coverage"]["tensor_candidates"] == 468
    assert report["coverage"]["system_candidate_scores"] == 4770
    assert report["coverage"]["lenstool_uncertainty_scores"] == 1000
    assert report["coverage"]["SPARC_systems"] == 131
    assert report["coverage"]["SPARC_points"] == 968


def test_p0568_selected_tensor_transfers_but_fails_promotion():
    report = read_report(P0568)
    selected = report["selected_tensor"]
    assert selected["family"] == "tidal_low_density"
    assert selected["source_width_kpc"] == 100.0
    assert selected["coupling_t"] == -0.15
    assert selected["holdout_improvement_vs_best_local_fraction"] > 0.0
    assert selected["holdout_improvement_vs_best_local_fraction"] < 0.10
    assert selected["holdout_improvement_vs_central_fraction"] > 0.30
    assert not report["gates"]["cluster_morphology_gate"]
    assert not report["gates"]["SPARC_gate"]
    assert report["gates"]["solar_gate"]
    assert not report["gates"]["overall_promotion"]


def test_p0568_operators_remain_positive_and_nearly_conservative():
    audit = pd.read_csv(P0568 / "numerical_audit.csv")
    assert audit.maximum_tensor_spectral_radius.le(1.0 + 1e-12).all()
    assert audit.correction_integral_fraction.lt(3e-4).all()
    candidates = pd.read_csv(P0568 / "candidate_scores.csv")
    assert np.isfinite(candidates.development_mean_JS).all()
    assert np.isfinite(candidates.holdout_mean_JS).all()


def test_p0568_cross_domain_failure_is_not_hidden_by_solar_screening():
    report = read_report(P0568)
    cross = report["selected_cross_domain"]
    assert cross["SPARC_outer_RMSE_km_s"] > 60.0
    assert not cross["SPARC_pass"]
    assert cross["Cassini_pass"]
    assert cross["Earth_pass"]
    assert cross["Mercury_pass"]


def test_p0568_refinement_resolves_to_an_unstable_transfer_lead():
    refinement = read_report(P0568B)
    assert refinement["selected"]["source_width_kpc"] == 125.0
    assert refinement["selected"]["coupling_t"] == -0.3
    assert refinement["selected"]["improvement_vs_local_development"] > 0.05
    interaction = read_report(P0568C)
    assert interaction["selected"]["source_width_kpc"] == 125.0
    assert interaction["selected"]["coupling_t"] == -0.3
    assert not interaction["stability"]["coupling_at_grid_boundary"]
    assert not interaction["stability"]["transfer_better_than_p0568_original"]
    assert not interaction["verdict"]["stable_universal_tensor_lead"]
    assert not interaction["verdict"]["SPARC_pass"]
    assert interaction["verdict"]["solar_pass"]
    assert not interaction["verdict"]["promoted"]
