from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_p0725_aqual_solver_robustness import physics_core
from run_p0728_complete_fine_grid_aqual import (
    aggregate_scores,
    normalized_prediction_change,
    selected_manifest,
)


def test_frozen_selected_solver_preserves_physics_and_p0727_selection() -> None:
    config = json.loads(
        (ROOT / "configs" / "p0728_complete_fine_grid_aqual.json").read_text(
            encoding="utf-8"
        )
    )
    base = json.loads((ROOT / config["baseManifest"]).read_text(encoding="utf-8"))
    p0727 = json.loads((ROOT / config["p0727Report"]).read_text(encoding="utf-8"))
    manifest = selected_manifest(base, config)
    selected = next(
        item
        for item in p0727["manifests"]
        if item["variant"] == config["selectedSolverVariant"]
    )
    assert p0727["selectedUniversalSolverVariant"] == config["selectedSolverVariant"]
    assert selected["solver"] == config["solver"]
    assert physics_core(manifest) == physics_core(base)
    assert manifest["parameterPolicy"]["perObjectParameters"] == []
    assert len(config["systems"]) == 4


def test_scoring_and_normalized_change_are_deterministic() -> None:
    aggregate = aggregate_scores(
        [
            {
                "rmseMPerS": 10.0,
                "residualsMPerS": [0.0, 10.0],
                "uncertaintiesMPerS": [2.0, 2.0],
            },
            {
                "rmseMPerS": 20.0,
                "residualsMPerS": [20.0],
                "uncertaintiesMPerS": [4.0],
            },
        ]
    )
    assert aggregate["systems"] == 2
    assert aggregate["validObservationPoints"] == 3
    assert aggregate["equalGalaxyRmseKmS"] == pytest.approx((250.0**0.5) / 1000.0)
    assert aggregate["pointWeightedRmseKmS"] == pytest.approx((500.0 / 3.0) ** 0.5 / 1000.0)
    assert normalized_prediction_change({0: 2.0, 1: 4.0}, {0: 3.0, 1: 5.0}) == pytest.approx(
        (2.0 / 20.0) ** 0.5
    )
