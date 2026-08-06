import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = ROOT / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import test_sigma_v19cy_a2319_endpoint_gain_candidates as endpoint


def test_frozen_endpoint_scope_preserves_successful_interior_evidence() -> None:
    config, parent, _, _, _ = endpoint.validate_inputs()
    folds = {fold["segment"]: fold for fold in parent["folds"]}
    assert folds[1]["comparison"]["passed"]
    assert folds[2]["comparison"]["passed"]
    assert not folds[0]["comparison"]["passed"]
    assert not folds[3]["comparison"]["passed"]
    assert [row["name"] for row in config["candidates"]] == [
        "nearest_anchor_constant",
        "nearest_anchor_linear",
    ]
    assert not config["inputs"]["science_event_allowed"]
    for key in (
        "read_cluster_sky_event_rows",
        "apply_gain_to_cluster_sky_events",
        "fit_cluster_velocity",
        "access_validation_or_holdout_assets",
        "open_lensing_halo_or_gravity_targets",
        "change_gravity_formula_or_parameters",
        "derive_or_select_action",
    ):
        assert not config["authorization"][key]


def _manufactured_histories() -> tuple[np.ndarray, np.ndarray]:
    dtype = [("TIME", "f8"), ("PIXEL", "i2"), ("TEMP_FIT", "f8")]
    common = np.zeros(5, dtype=dtype)
    common["TIME"] = np.arange(5, dtype=float) * 100.0
    common["PIXEL"] = 12
    common["TEMP_FIT"] = 0.1 + 2e-6 * common["TIME"]
    fe55 = np.zeros(36 * 3, dtype=dtype)
    row = 0
    for time in common["TIME"][1:4]:
        common_value = 0.1 + 2e-6 * time
        for pixel in range(36):
            fe55["TIME"][row] = time
            fe55["PIXEL"][row] = pixel
            fe55["TEMP_FIT"][row] = common_value + pixel * 1e-4 + 3e-8 * time
            row += 1
    return fe55, common


def test_nearest_anchor_constant_and_linear_models_are_distinct() -> None:
    fe55, common = _manufactured_histories()
    constant = endpoint.fit_nearest_models("nearest_anchor_constant", fe55, common)
    linear = endpoint.fit_nearest_models("nearest_anchor_linear", fe55, common)
    assert constant[7]["slope_per_second"] == 0.0
    assert abs(float(linear[7]["slope_per_second"]) - 3e-8) < 1e-15
    assert constant[12]["differential_at_center"] == 0.0
    assert linear[12]["differential_at_center"] == 0.0


def test_selection_requires_one_rule_to_pass_both_endpoints() -> None:
    order = ["nearest_anchor_constant", "nearest_anchor_linear"]
    split = {
        "nearest_anchor_constant": [
            {"comparison": {"passed": True}},
            {"comparison": {"passed": False}},
        ],
        "nearest_anchor_linear": [
            {"comparison": {"passed": False}},
            {"comparison": {"passed": True}},
        ],
    }
    selection = endpoint.select_candidate(split, order)
    assert not selection["passed"]
    assert selection["selected"] is None
    both = {
        candidate: [{"comparison": {"passed": True}} for _ in range(2)]
        for candidate in order
    }
    selection = endpoint.select_candidate(both, order)
    assert selection["passed"]
    assert selection["selected"] == "nearest_anchor_constant"
