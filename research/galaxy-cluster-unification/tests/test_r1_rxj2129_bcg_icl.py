import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def test_bcg_icl_result_records_decisive_nonidentifiability() -> None:
    report = json.loads(
        (ROOT / "results/r1_rxj2129_bcg_icl/report.json").read_text(
            encoding="utf-8"
        )
    )
    assert report["gravity_or_lens_residual_read"] is False
    assert report["radial_cross_validation"]["gate_pass"] is True
    improvements = report["radial_cross_validation"][
        "two_component_improvement_fraction"
    ]
    assert improvements["F125W"] >= 0.20
    assert improvements["F814W"] >= 0.20
    assert report["structural_gate"]["outer_to_inner_radius_ratio_pass"] is True
    assert report["structural_gate"]["f125w_outer_light_fraction_pass"] is False
    assert report["bcg_icl_nonidentifiability_explicit"] is True
    assert report["component_identifiability_gate_pass"] is False
    assert report["stellar_mass_mapping_authorized"] is False
    assert report["sensitivity_grid_status"].startswith("not_run_because")


def test_bcg_icl_fit_ledger_contains_baseline_and_two_radial_folds() -> None:
    variants = pd.read_csv(
        ROOT / "data/derived/r1_rxj2129_bcg_icl_model_variants.csv"
    )
    assert len(variants) == 6
    assert set(variants["model"]) == {"one_component", "two_component"}
    assert set(variants["variant"]) == {
        "baseline_full",
        "radial_cv_fold_0",
        "radial_cv_fold_1",
    }
    two = variants[variants["model"] == "two_component"]
    assert (two["outer_to_inner_re_ratio"] >= 2.0).all()
