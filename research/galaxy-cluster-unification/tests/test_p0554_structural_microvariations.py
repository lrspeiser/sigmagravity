import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "p0554_structural_microvariations"
sys.path.insert(0, str(ROOT / "scripts"))

from run_p0554_multiscale_elasticity import build_variants  # noqa: E402


def load_json(path):
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def test_structural_protocol_is_frozen_and_parent_preserving():
    protocol = load_json("configs/p0554_structural_microvariations_protocol.json")
    assert protocol["status"] == "frozen_before_any_structural_variant_score"
    assert len(protocol["parameter_coordinates"]) == 9
    assert protocol["evaluation"]["formula_parameters_fit"] == 0
    assert protocol["evaluation"]["lens_geometry_parameters_fit"] == 0
    variants = build_variants(protocol)
    assert len(variants) == 1 + 9 * 4 * 2
    assert len({variant.variant_id for variant in variants}) == 73


def test_outputs_cover_every_formula_and_real_data_domain():
    report = load_json("results/p0554_structural_microvariations/report.json")
    assert report["coverage"] == {
        "variants": 73,
        "structural_parameters": 9,
        "coordinate_steps": 4,
        "SPARC_galaxies": 131,
        "SPARC_outer_points": 968,
        "CLASH_systems": 20,
        "CLASH_points": 84,
        "raw_clusters": 5,
        "raw_heldout_images": 18,
    }
    scores = pd.read_csv(RESULTS / "variant_scores.csv")
    raw = pd.read_csv(RESULTS / "raw_system_scores.csv")
    central = pd.read_csv(RESULTS / "central_differences.csv")
    assert scores.variant_id.nunique() == 73
    assert len(raw) == 73 * 5
    assert len(central) == 9 * 4


def test_structural_parent_reproduces_the_frozen_p0554_scores_exactly():
    report = load_json("results/p0554_structural_microvariations/report.json")
    local = load_json("results/p0554_local_cross_domain_sensitivity/report.json")
    parent = report["parent_reproduction"]
    baseline = local["baseline"]
    assert parent["galaxy_outer_RMSE_km_s"] == baseline["galaxy_outer_RMSE_km_s"]
    assert parent["cluster_RMSE_dex"] == baseline["cluster_RMSE_dex"]
    assert (
        parent["Mercury_precession_mas_per_century"]
        == baseline["Mercury_precession_mas_per_century"]
    )
    assert parent["raw_roots"] == 17


def test_stable_structural_domain_rankings_are_frozen():
    report = load_json("results/p0554_structural_microvariations/report.json")
    top = report["top_stable_structural_parameter_by_domain"]
    assert top["galaxy"]["parameter"] == "response_addition_softness"
    assert top["cluster"]["parameter"] == "lensing_addition_softness"
    assert top["RXJ2129"]["parameter"] == "screen_softness"
    assert top["four_cluster"]["parameter"] == "lensing_addition_softness"
    assert top["Mercury"]["parameter"] == "response_addition_softness"
    assert top["galaxy"]["better_direction"] == "plus"
    assert top["cluster"]["better_direction"] == "minus"


def test_addition_laws_are_powerful_but_not_universal():
    summary = pd.read_csv(RESULTS / "parameter_summary.csv").set_index("parameter")
    response = summary.loc["response_addition_softness"]
    lensing = summary.loc["lensing_addition_softness"]
    assert response.galaxy_better_direction == "plus"
    assert response.cluster_better_direction == "minus"
    assert response.Mercury_better_direction == "plus"
    assert lensing.cluster_better_direction == "minus"
    assert lensing.RXJ2129_better_direction == "minus"
    assert lensing.four_cluster_better_direction == "plus"
    assert lensing.galaxy_median_abs_slope == 0.0
    assert lensing.Mercury_median_abs_slope == 0.0


def test_root_bifurcations_and_solar_crossing_are_not_accuracy_claims():
    report = load_json("results/p0554_structural_microvariations/report.json")
    topology = report["root_topology"]
    assert topology["baseline_roots"] == 17
    assert topology["minimum_roots"] == 14
    assert topology["maximum_roots"] == 18
    assert set(topology["parameters_bifurcating_at_smallest_step"]) == {
        "response_addition_softness",
        "lensing_addition_softness",
        "potential_scale_coupling",
        "potential_softness",
    }
    assert topology["parameters_never_changing_roots"] == []
    assert report["solar_boundary_crossings"] == [
        {
            "parameter": "response_addition_softness",
            "smallest_solar_boundary_crossing_u": 0.25,
        }
    ]
    assert report["verdict"]["candidate_selected"] is False


def test_screen_shape_is_only_material_same_direction_structure():
    report = load_json("results/p0554_structural_microvariations/report.json")
    assert report["stable_same_direction_nonSolar"] == ["screen_softness"]
    scores = pd.read_csv(RESULTS / "variant_scores.csv").set_index("variant_id")
    parent = scores.loc["baseline"]
    lower = scores.loc["screen_softness_minus_u010"]
    assert lower.galaxy_outer_RMSE_km_s < parent.galaxy_outer_RMSE_km_s
    assert lower.cluster_RMSE_dex < parent.cluster_RMSE_dex
    assert lower.all_solar_proxies_pass == np.True_
