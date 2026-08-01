import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load(path: str) -> dict:
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def test_variable_exponent_protocol_was_frozen_before_scoring():
    protocol = load("configs/unbounded_running_variable_exponent_protocol.json")
    assert protocol["status"] == "frozen_before_variable_exponent_scores"
    assert set(protocol["models"]) == {
        "curvature_variable_mass_power",
        "curvature_variable_density_power",
        "curvature_variable_shape_power",
    }
    assert protocol["selection"]["no_per_object_gravity_parameters"]


def test_variable_exponent_result_records_the_cross_domain_failure():
    report = load("results/unbounded_running_variable_exponent/report.json")
    assert not report["verdict"]["any_universal_survivor"]
    models = report["models"]
    assert models["curvature_variable_mass_power"]["bridge_metrics"][
        "equal_domain_RMSE_dex"
    ] < report["references"]["bridge"]["prior_Sigma"]["equal_domain_RMSE_dex"]
    assert models["curvature_variable_mass_power"]["SPARC_metrics"][
        "outer_holdout"
    ]["RMSE_km_s"] > report["references"]["SPARC"]["NFW_outer_RMSE_km_s"]
    assert report["selection"]["post_transfer_consistency_ranking"][0] == (
        "curvature_variable_shape_power"
    )
    assert not models["curvature_variable_shape_power"]["raw_lensing"]["heldout"][
        "all_roots_converged"
    ]
