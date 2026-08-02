from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_p0724_grid_box_vertical_sensitivity import (
    paired_prediction_sensitivity,
    rank_orders,
    sensitivity_summaries,
)


def fixture_config() -> dict:
    return {
        "baselineScenario": "baseline",
        "systems": ["G1"],
        "models": [{"id": "M1"}],
        "scenarios": [
            {"id": "baseline", "role": "reference"},
            {"id": "variant", "role": "sensitivity"},
        ],
        "stabilityGates": {
            "maximumScenarioMedianNormalizedPredictionRmse": 0.10,
            "maximumScenarioP90NormalizedPredictionRmse": 0.10,
            "maximumModelAggregateFitRmseRelativeChange": 0.20,
        },
    }


def test_paired_prediction_sensitivity_uses_baseline_speed_scale() -> None:
    rows = [
        {
            "scenario": scenario,
            "model": "M1",
            "system_id": "G1",
            "point_index": index,
            "predicted_speed_m_s": speed,
        }
        for scenario, speeds in (
            ("baseline", [1000.0, 2000.0]),
            ("variant", [1100.0, 1900.0]),
        )
        for index, speed in enumerate(speeds)
    ]
    result = paired_prediction_sensitivity(fixture_config(), rows)
    assert len(result) == 1
    assert result[0]["paired_points"] == 2
    assert np.isclose(result[0]["prediction_delta_rmse_km_s"], 0.1)
    assert np.isclose(
        result[0]["normalized_prediction_rmse"],
        0.1 / np.sqrt((1.0**2 + 2.0**2) / 2.0),
    )


def test_sensitivity_summary_applies_frozen_gates_and_ranks() -> None:
    config = fixture_config()
    paired = [
        {
            "scenario": "variant",
            "model": "M1",
            "system_id": "G1",
            "normalized_prediction_rmse": 0.05,
        }
    ]
    summaries = [
        {"scenario": "baseline", "model": "M1", "equalGalaxyRmseKmS": 10.0},
        {"scenario": "variant", "model": "M1", "equalGalaxyRmseKmS": 11.0},
    ]
    result = sensitivity_summaries(config, paired, summaries)
    assert result[0]["status"] == "stable"
    assert np.isclose(result[0]["maximumModelAggregateFitRmseRelativeChange"], 0.1)
    assert rank_orders(config, summaries) == {"baseline": ["M1"], "variant": ["M1"]}
