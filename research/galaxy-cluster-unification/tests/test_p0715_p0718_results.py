from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def report(stage: str) -> dict:
    return json.loads((ROOT / "results" / stage / "report.json").read_text(encoding="utf-8"))


def test_p0715_coordinate_safe_engine_is_stable_and_archive_conformant() -> None:
    outcome = report("p0715_sky_lensing_engine_validation")
    assert outcome["status"] == "pass"
    assert outcome["formula_rescored"] is False
    assert outcome["families"] == 18
    assert outcome["root_count_conformance_to_P0714"] is True
    assert outcome["maximum_source_position_difference_arcsec"] < 1.0e-9
    assert outcome["maximum_RMS_difference_arcsec"] < 1.0e-6
    assert outcome["all_model_root_count_stability_fraction"] == 1.0
    assert outcome["glafic_root_count_stability_fraction"] == 1.0
    assert outcome["coordinate_contract"] == {
        "array_axis_0": "north",
        "array_axis_1": "east",
        "vector_component_0": "alpha_east",
        "vector_component_1": "alpha_north",
    }

    conformance = pd.read_csv(
        ROOT / "results/p0715_sky_lensing_engine_validation/p0714_conformance.csv"
    )
    assert len(conformance) == 36
    assert conformance.root_count_match.all()
    assert conformance.root_count_stable_81_161_241.all()


def test_p0716_identifies_a_hessian_deficit_without_fitting() -> None:
    outcome = report("p0716_spent_arc_structure_deficit")
    assert outcome["status"] == "completed_spent_structure_diagnostic"
    assert outcome["formula_fitted"] is False
    assert outcome["sample_is_spent"] is True
    overall = outcome["candidate_overall"]
    assert overall["median_kappa_needed"] > 0.30
    assert overall["median_shear_needed"] > 0.13
    assert overall["median_minimum_eigenvalue_gap"] > 0.45
    assert overall["candidate_near_critical_arc_fraction"] == 0.0
    assert overall["halo_near_critical_arc_fraction"] > 0.50
    assert overall["shear_correlation_to_halo"] < 0.0

    deficits = pd.read_csv(
        ROOT / "results/p0716_spent_arc_structure_deficit/arc_structure_deficits.csv"
    )
    assert deficits.cluster.nunique() == 2
    assert deficits.image_id.nunique() >= 40
    assert np.max(np.abs(deficits.eigenvalue_decomposition_residual)) < 1.0e-9


def test_p0717_screened_contrast_transfers_q_but_fails_raw_lensing() -> None:
    outcome = report("p0717_screened_contrast_transfer")
    assert outcome["status"] == "fail_no_formula_passed_all_transfer_gates"
    assert outcome["survivors"] == []
    assert outcome["per_family_gravity_parameters"] == 0
    assert outcome["per_cluster_gravity_parameters_at_test_time"] == 0
    transfer = {row["formula"]: row for row in outcome["parameter_transfer"]}
    assert transfer["AQUAL_contrast_hessian_fit"]["q_disagreement_fraction"] < 0.05
    assert transfer["AQUAL_contrast_hessian_fit"]["solar_gate"] is True
    assert transfer["QUMOND_contrast_source_fit"]["q_transfer_gate"] is True

    gates = pd.read_csv(
        ROOT / "results/p0717_screened_contrast_transfer/formula_rejection_gates.csv"
    )
    assert len(gates) == 4
    assert not gates.all_gates.any()


def test_p0718_componentwise_ordering_improves_roots_but_fails_transfer() -> None:
    outcome = report("p0718_componentwise_summation_transfer")
    assert outcome["status"] == "fail_raw_transfer_gates"
    assert outcome["sample_is_spent"] is True
    assert outcome["q_disagreement_fraction"] < 0.07
    assert outcome["gates"]["q_transfer"] is True
    assert outcome["gates"]["root_convergence"] is False
    assert outcome["gates"]["RMS_ratio"] is False
    assert outcome["all_gates"] is False
    scores = pd.DataFrame(outcome["scores"]).set_index("test_cluster")
    assert scores.loc["PLCKG287", "root_convergence_fraction"] > 0.85
    assert scores.loc["AS295", "root_convergence_fraction"] < 0.60
    assert (scores.median_RMS_ratio_to_halo > 10.0).all()

    sensitivity = pd.read_csv(
        ROOT / "results/p0718_componentwise_summation_transfer/member_input_sensitivities.csv"
    )
    assert sensitivity.variant.nunique() == 5
    assert len(sensitivity) == 10
    resolution = pd.read_csv(
        ROOT / "results/p0718_componentwise_summation_transfer/resolution_audit.csv"
    )
    assert (resolution.median_high_minus_low_arc_deflection_arcsec > 0.5).all()
