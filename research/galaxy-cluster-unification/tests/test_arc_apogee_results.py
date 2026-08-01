import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
CROSS = ROOT / "results" / "arc_apogee_cross_domain"
REFINE = ROOT / "results" / "arc_apogee_boundary_refinement"


def test_cross_domain_report_has_complete_sweeps():
    report = json.loads((CROSS / "report.json").read_text(encoding="utf-8"))
    assert report["coverage"] == {
        "galaxy_variants": 540,
        "SPARC_galaxies": 131,
        "SPARC_outer_points": 968,
        "cluster_variants": 180,
        "cluster_systems": 10,
        "cluster_method_scores": 3600,
    }
    galaxy = pd.read_csv(CROSS / "galaxy_variant_scores.csv")
    cluster = pd.read_csv(CROSS / "cluster_variant_scores.csv")
    assert galaxy.candidate_id.nunique() == 540
    assert cluster.candidate_id.nunique() == 180
    assert len(cluster) == 3600


def test_reference_and_primary_scores_are_frozen():
    report = json.loads((CROSS / "report.json").read_text(encoding="utf-8"))
    refs = report["references_same_RAR_nuisances"]
    assert np.isclose(refs["Newtonian_same_nuisance"]["RMSE_km_s"], 72.39921475798786)
    assert np.isclose(refs["RAR_same_nuisance"]["RMSE_km_s"], 10.348465773189677)
    assert np.isclose(refs["simple_MOND_same_nuisance"]["RMSE_km_s"], 10.439784073545031)
    primary = report["galaxy_selection"]["primary"]
    best = report["galaxy_selection"]["best_overall"]
    assert primary["candidate_id"] == "A0112"
    assert primary["cross_galaxy_outer_RMSE_km_s"] > 30.0
    assert best["candidate_id"] == "A0192"
    assert np.isclose(best["cross_galaxy_outer_RMSE_km_s"], 13.197860196706849)
    assert best["all_solar_proxies_pass"] is True


def test_extent_gate_has_opposite_domain_effects():
    report = json.loads((CROSS / "report.json").read_text(encoding="utf-8"))
    impacts = {
        (row["domain"], row["parameter"]): row
        for row in report["most_impactful_parameters"]
    }
    galaxy_gate = impacts[("SPARC", "gate_mode")]
    cluster_gate = impacts[("cluster_shape", "gate_mode")]
    assert galaxy_gate["best_level"] == "none"
    assert galaxy_gate["impact_span"] > 18.0
    assert cluster_gate["best_level"] == "cluster_logistic_soft"
    assert cluster_gate["impact_span"] > 0.029


def test_boundary_refinement_selects_stable_solar_safe_universal_law():
    report = json.loads((REFINE / "report.json").read_text(encoding="utf-8"))
    scores = pd.read_csv(REFINE / "scores.csv")
    assert scores.candidate_id.nunique() == 1440
    best = report["best_variant"]
    assert best["candidate_id"] == "R1322"
    assert best["scale_mix_mu"] == 1.0
    assert best["alpha"] == 0.75
    assert best["apogee_ratio"] == 100.0
    assert best["screen_exponent"] == 1.0
    assert 1.44 < best["fold_q_min"] < best["universal_q"] < best["fold_q_max"] < 1.48
    assert np.isclose(best["cross_galaxy_outer_RMSE_km_s"], 12.966366834672476)
    assert np.isclose(report["best_arc_to_RAR_RMSE_ratio"], 1.2529748002129104)
    assert best["all_solar_proxies_pass"] is True
    assert abs(best["Mercury_precession_mas_per_century"]) < 3.1


def test_square_root_mass_radius_improvement_is_monotonic():
    report = json.loads((REFINE / "report.json").read_text(encoding="utf-8"))
    rows = sorted(report["best_variant_by_scale_mix"], key=lambda row: row["scale_mix_mu"])
    assert [row["scale_mix_mu"] for row in rows] == [0.0, 0.25, 0.5, 0.75, 1.0]
    rmse = [row["cross_galaxy_outer_RMSE_km_s"] for row in rows]
    assert all(left > right for left, right in zip(rmse, rmse[1:]))
    assert np.allclose(rmse, [23.5239488, 21.3535233, 18.6545555, 15.4210101, 12.9663668])


def test_selected_law_is_honestly_worse_than_rar_in_every_morphology_bin():
    report = json.loads((REFINE / "report.json").read_text(encoding="utf-8"))
    morphology = report["morphology_for_best"]
    assert len(morphology) == 10
    assert all(row["arc_over_RAR"] > 1.0 for row in morphology)
    by_bin = {row["bin"]: row for row in morphology}
    assert by_bin["late_type"]["arc_over_RAR"] < by_bin["gas_poor"]["arc_over_RAR"]
