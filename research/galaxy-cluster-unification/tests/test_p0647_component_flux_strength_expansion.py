from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "p0647_component_flux_strength_expansion"


def report() -> dict:
    return json.loads((RESULTS / "report.json").read_text(encoding="utf-8"))


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_boundary_result_is_rejected_by_the_frozen_gates():
    result = report()
    assert result["status"] == "fail"
    assert result["all_progression_gates_pass"] is False
    assert sum(result["gate_results"].values()) == 10
    assert result["gate_results"]["lambda_interior"] is False
    assert result["gate_results"]["strict_local_minimum"] is False
    assert all(
        value
        for name, value in result["gate_results"].items()
        if name not in {"lambda_interior", "strict_local_minimum"}
    )


def test_selected_boundary_gain_is_not_hidden_or_promoted():
    selection = report()["selection"]
    assert selection["selected_lambda"] == 12.5
    assert selection["CV_improvement_fraction_vs_lambda0"] > 0.16
    assert selection["CV_improvement_fraction_vs_P0646_isotropic"] > 0.15
    assert selection["P0601_spent_heldout_used_for_selection"] is False
    audit = report()["local_minimum_audit"]
    assert audit["interior"] is False
    assert audit["strict_local_minimum"] is False


def test_exact_grid_records_root_topology_instability():
    scores = pd.read_csv(RESULTS / "lambda_scores.csv").set_index("lambda")
    assert len(scores) == 6
    assert int(scores.loc[3.5, "CV_roots"]) == 14
    assert int(scores.loc[5.0, "CV_roots"]) == 13
    assert np.isinf(scores.loc[3.5, "pooled_CV_RMS_arcsec"])
    assert np.isinf(scores.loc[5.0, "pooled_CV_RMS_arcsec"])
    assert scores.loc[[6.5, 8.0, 10.0, 12.5], "all_CV_roots"].all()


def test_descriptive_full_refit_and_solar_proxy_are_preserved():
    result = report()
    assert result["full_refit"]["training_roots"] == 15
    assert result["full_refit"]["spent_heldout_roots"] == 7
    assert result["full_refit"]["spent_heldout_worsening_fraction_vs_P0599"] < 0.10
    assert result["solar_proxy"]["selected_lambda_1au_coefficient"] <= result["solar_proxy"]["limit"]
    assert result["solar_proxy"]["is_a_full_PPN_or_Cassini_test"] is False


def test_blind_outcomes_remain_sealed():
    result = report()
    assert result["sealed_P0633_kinematics_opened"] is False
    assert result["sealed_P0640_lensing_constraints_opened"] is False
    assert result["coverage"]["per_object_spatial_gravity_parameters"] == 0


def test_protocol_and_source_hashes_make_result_reproducible():
    result = report()
    assert result["protocol_sha256"] == digest(
        ROOT / "configs/p0647_component_flux_strength_expansion.json"
    )
    assert result["source_sha256"] == digest(
        ROOT / "scripts/run_p0647_component_flux_strength_expansion.py"
    )
    assert (RESULTS / "strength_expansion.png").stat().st_size > 20000
