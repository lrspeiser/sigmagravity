import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results/p0613_bounded_endpoint_cross_domain"


def test_factorial_uses_three_universal_coordinates_and_real_cross_domain_data():
    report = json.loads((OUTPUT / "report.json").read_text(encoding="utf-8"))
    variants = pd.read_csv(OUTPUT / "variant_scores.csv")
    assert report["coverage"]["universal_variants"] == 27
    assert report["coverage"]["raw_clusters"] == 4
    assert report["coverage"]["heldout_images"] == 11
    assert report["coverage"]["SPARC_galaxies"] == 131
    assert len(variants) == 27
    assert report["formula"]["per_object_gravity_parameters"] == 0


def test_every_variant_has_a_valid_galaxy_score_and_bounded_root_count():
    variants = pd.read_csv(OUTPUT / "variant_scores.csv")
    assert np.isfinite(variants.SPARC_outer_RMSE_km_s).all()
    assert variants.heldout_converged_roots.between(0, 11).all()
    assert variants.complete_systems.between(0, 4).all()


def test_saturation_is_an_exact_galaxy_null_but_not_cluster_null():
    variants = pd.read_csv(OUTPUT / "variant_scores.csv")
    for _, block in variants.groupby(["width_over_R80", "route_fraction_multiplier"]):
        assert block.SPARC_outer_RMSE_km_s.nunique() == 1
    cap_impact = pd.read_csv(OUTPUT / "parameter_impacts.csv").set_index("parameter").loc[
        "contrast_cap"
    ]
    assert abs(cap_impact.SPARC_RMSE_span_km_s) < 1.0e-12
    assert cap_impact.mean_root_count_span == 0.0
    assert cap_impact.maximum_system_root_pattern_span == 0.0
    assert cap_impact.safe_variant_fraction_span > 0.0
    interactions = pd.read_csv(OUTPUT / "interaction_effects.csv")
    cap_interactions = interactions[
        interactions.left_parameter.eq("contrast_cap")
        | interactions.right_parameter.eq("contrast_cap")
    ]
    assert (cap_interactions.root_count_interaction_RMS > 0.0).all()


def test_solar_null_passes_but_galaxy_parity_decides_advance():
    report = json.loads((OUTPUT / "report.json").read_text(encoding="utf-8"))
    assert report["solar"]["maximum_fractional_change"] == 0.0
    assert report["solar"]["Mercury_precession_mas_per_century"] == 0.0
    assert report["gates"]["solar_fractional_change_pass"] is True
    assert report["gates"]["Mercury_precession_pass"] is True
    assert report["gates"]["cross_domain_advance_pass"] is False
    assert report["interpretation"]["formula_promoted"] is False
