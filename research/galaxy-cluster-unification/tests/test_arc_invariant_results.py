import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
COARSE = ROOT / "results" / "arc_invariant_absolute_lensing"
REFINE = ROOT / "results" / "arc_invariant_pareto_refinement"


def load_report(directory: Path) -> dict:
    text = (directory / "report.json").read_text(encoding="utf-8")
    assert "Infinity" not in text
    return json.loads(text)


def test_coarse_absolute_lensing_exposes_baseline_amplitude_failure():
    report = load_report(COARSE)
    assert report["coverage"]["variants"] == 58
    assert report["coverage"]["CLASH_systems"] == 20
    assert report["coverage"]["CLASH_points"] == 84
    baseline = report["baseline"]
    assert np.isclose(baseline["cluster_RMSE_dex"], 0.5985203705875083)
    assert np.isclose(baseline["cluster_median_observed_over_predicted"], 3.681874000343277)
    assert baseline["cluster_RMSE_dex"] > report["references"]["fixed_RAR_cluster_RMSE_dex"]


def test_coarse_sweep_ranks_potential_and_photon_controls_first():
    report = load_report(COARSE)
    impacts = report["parameter_impacts"]
    assert impacts[0]["family"] == "potential_depth"
    assert impacts[0]["cluster_impact_span_dex"] > 0.27
    assert impacts[1]["family"] == "photon_extra_multiplier"
    assert impacts[1]["galaxy_impact_span_km_s"] == 0.0
    assert report["best_zero_slip_cluster"]["all_solar_proxies_pass"] is True


def test_fine_grid_has_complete_cross_domain_coverage():
    report = load_report(REFINE)
    assert report["coverage"] == {
        "variants": 576,
        "SPARC_galaxies": 131,
        "SPARC_outer_points": 968,
        "CLASH_systems": 20,
        "CLASH_points": 84,
        "raw_shortlist": 4,
        "raw_optimization_starts_each": 16,
    }
    scores = pd.read_csv(REFINE / "variant_scores.csv")
    assert scores.candidate_id.nunique() == 576
    assert scores.all_solar_proxies_pass.all()


def test_fine_grid_freezes_zero_slip_and_photon_tradeoff():
    report = load_report(REFINE)
    photon = report["best_cluster_within_galaxy_limit"]
    zero = report["best_zero_slip_within_galaxy_limit"]
    assert photon["candidate_id"] == "P0070"
    assert photon["photon_extra_multiplier"] == 2.25
    assert np.isclose(photon["cluster_RMSE_dex"], 0.11117016509702089)
    assert np.isclose(photon["cross_galaxy_outer_RMSE_km_s"], 15.500950677047644)
    assert zero["candidate_id"] == "P0420"
    assert zero["photon_extra_multiplier"] == 1.0
    assert np.isclose(zero["cluster_RMSE_dex"], 0.16664256866398205)
    assert zero["all_solar_proxies_pass"] is True


def test_raw_images_reject_profile_level_winners():
    report = load_report(REFINE)
    raw = {row["candidate_id"]: row for row in report["raw_RXJ2129_scores"]}
    assert raw["P0070"]["heldout_all_roots_converged"] is True
    assert np.isclose(raw["P0070"]["heldout_RMS_arcsec"], 1.3236818086601871)
    assert raw["P0420"]["heldout_all_roots_converged"] is False
    assert raw["P0420"]["heldout_RMS_arcsec"] is None
    assert report["best_finite_raw_score"]["candidate_id"] == "P0554"
    assert np.isclose(report["best_finite_raw_score"]["heldout_RMS_arcsec"], 1.2449035439758638)
    assert report["best_finite_raw_score"]["heldout_RMS_arcsec"] > 0.5
    assert report["best_finite_raw_score"]["heldout_RMS_arcsec"] > report["references"]["previous_locked_raw_candidate_heldout_RMS_arcsec"]


def test_best_raw_compromise_has_no_hidden_galaxy_type_victory():
    morphology = pd.read_csv(REFINE / "selected_morphology_scores.csv")
    selected = morphology[morphology.candidate_id.eq("P0554")]
    assert len(selected) == 10
    assert (selected.arc_over_RAR > 1.0).all()
    assert selected.loc[selected.bin.eq("dwarf_mass"), "arc_over_RAR"].iloc[0] > 1.45
