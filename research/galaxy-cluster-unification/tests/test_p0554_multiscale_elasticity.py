import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "p0554_multiscale_elasticity"
sys.path.insert(0, str(ROOT / "scripts"))

from run_p0554_multiscale_elasticity import (  # noqa: E402
    build_variants,
    normalized_difference,
)


def load_json(path):
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def test_protocol_is_frozen_and_builds_unique_central_pairs():
    protocol = load_json("configs/p0554_multiscale_elasticity_protocol.json")
    assert protocol["status"] == "frozen_before_any_multiscale_variant_score"
    assert protocol["coordinate_steps"] == [0.1, 0.25, 0.5, 1.0]
    assert len(protocol["parameter_coordinates"]) == 11
    assert protocol["evaluation"]["formula_parameters_fit"] == 0
    assert protocol["evaluation"]["lens_geometry_parameters_fit"] == 0
    variants = build_variants(protocol)
    assert len(variants) == 1 + 11 * 4 * 2
    assert len({variant.variant_id for variant in variants}) == len(variants)
    by_id = {variant.variant_id: variant for variant in variants}
    assert np.isclose(by_id["alpha_minus_u010"].changed_value, 0.7425)
    assert np.isclose(by_id["mass_radius_delta_plus_u010"].changed_value, 0.005)


def test_central_difference_uses_declared_normalization():
    slope, curvature = normalized_difference(8.0, 10.0, 14.0, 0.5, 10.0)
    assert np.isclose(slope, 0.6)
    assert np.isclose(curvature, 0.8)


def test_multiscale_outputs_have_complete_coverage():
    report = load_json("results/p0554_multiscale_elasticity/report.json")
    assert report["coverage"] == {
        "variants": 89,
        "parameters": 11,
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
    assert scores.variant_id.nunique() == 89
    assert len(raw) == 89 * 5
    assert len(central) == 11 * 4


def test_stable_domain_rankings_and_solar_boundary_are_frozen():
    report = load_json("results/p0554_multiscale_elasticity/report.json")
    top = report["top_stable_parameter_by_median_multiscale_slope"]
    assert top["galaxy"]["parameter"] == "alpha"
    assert top["cluster"]["parameter"] == "mass_radius_delta"
    assert top["RXJ2129"]["parameter"] == "secondary_path_ratio_power"
    assert top["four_cluster"]["parameter"] == "mass_radius_delta"
    assert top["Mercury"]["parameter"] == "screen_exponent"
    assert top["Mercury"]["better_direction"] == "plus"
    crossings = {
        row["parameter"]: row["smallest_solar_boundary_crossing_u"]
        for row in report["solar_boundary_crossings"]
    }
    assert crossings == {
        "mass_radius_delta": 1.0,
        "screen_exponent": 0.25,
        "alpha": 1.0,
    }


def test_raw_bifurcations_are_separate_from_continuous_rms_slopes():
    report = load_json("results/p0554_multiscale_elasticity/report.json")
    topology = report["root_topology"]
    assert topology["baseline_roots"] == 17
    assert topology["minimum_roots"] == 13
    assert topology["maximum_roots"] == 18
    assert set(topology["parameters_bifurcating_at_smallest_step"]) == {
        "mass_radius_delta",
        "invariant_scale",
        "photon_extra_multiplier",
        "invariant_power",
    }
    central = pd.read_csv(RESULTS / "central_differences.csv")
    mass = central[
        central.parameter.eq("mass_radius_delta") & central.coordinate_u.eq(0.1)
    ].iloc[0]
    assert mass.RXJ2129_minus_roots + mass.four_cluster_minus_roots == 16
    assert mass.RXJ2129_plus_roots + mass.four_cluster_plus_roots == 18
    assert mass.RXJ2129_common_complete_systems == 1


def test_transition_radius_has_opposite_raw_directions_and_screen_scale_is_only_alignment():
    report = load_json("results/p0554_multiscale_elasticity/report.json")
    assert report["stable_same_direction_nonSolar"] == ["screen_scale"]
    summary = pd.read_csv(RESULTS / "parameter_summary.csv").set_index("parameter")
    mass = summary.loc["mass_radius_delta"]
    assert mass.RXJ2129_better_direction == "plus"
    assert mass.four_cluster_better_direction == "minus"
    assert mass.stable_nonSolar_directions_agree == np.False_
    assert report["verdict"]["candidate_selected"] is False
