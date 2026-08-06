import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = ROOT / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import apply_sigma_v19cy_a2319_bracketed_science_calibration as bracketed


def test_frozen_reduced_scope_contains_only_undisturbed_brackets() -> None:
    config, topology, _ = bracketed.validate_inputs()
    included = [row["name"] for row in config["included_branches"]]
    assert included == [
        "000101_open_0_cross_obsid",
        "000101_open_1_cross_obsid",
        "000102_open_0_cross_obsid",
    ]
    excluded = {row["name"] for row in config["excluded_branches"]}
    assert excluded == {
        "000102_open_1_cross_obsid",
        "000103_pre_adr_forward",
        "000103_post_adr_backward",
        "000103_final_forward",
    }
    branches = {row["name"]: row for row in topology["branches"]}
    assert all(name in branches for name in set(included) | excluded)
    assert config["terminal_gate"]["required_branch_outputs"] == 3
    assert config["inputs"]["required_object_by_obsid"] == {
        "000101000": "Abell2319",
        "000102000": "Abell2319_Cor1",
    }
    assert config["recorded_failures"][0]["sky_event_row_read"] is False
    assert not config["authorization"]["inspect_or_fit_cluster_energy_distribution"]
    assert not config["authorization"]["fit_cluster_velocity"]
    assert not config["authorization"]["access_validation_or_holdout_assets"]


def test_bracketed_model_recovers_manufactured_differential_trend() -> None:
    dtype = [("TIME", "f8"), ("PIXEL", "i2"), ("TEMP_FIT", "f8")]
    fe55 = {}
    pxcal = {}
    for obsid, base in (("left", 0.0), ("right", 1000.0)):
        common = np.zeros(5, dtype=dtype)
        common["TIME"] = base + np.arange(5, dtype=float) * 100.0
        common["PIXEL"] = 12
        common["TEMP_FIT"] = 0.1 + 2e-6 * common["TIME"]
        pxcal[obsid] = common
        rows = np.zeros(36 * 3, dtype=dtype)
        row = 0
        for time in common["TIME"][1:4]:
            common_value = 0.1 + 2e-6 * time
            for pixel in range(36):
                rows["TIME"][row] = time
                rows["PIXEL"][row] = pixel
                rows["TEMP_FIT"][row] = (
                    common_value + pixel * 1e-4 + 4e-8 * time
                )
                row += 1
        fe55[obsid] = rows
    models = bracketed.fit_bracketed_models(["left", "right"], fe55, pxcal)
    assert abs(float(models[7]["slope_per_second"]) - 4e-8) < 1e-15
    predicted = float(models[7]["differential_at_center"]) + float(
        models[7]["slope_per_second"]
    ) * (700.0 - float(models[7]["time_center"]))
    assert abs(predicted - (7e-4 + 4e-8 * 700.0)) < 1e-12
    assert models[12]["differential_at_center"] == 0.0
