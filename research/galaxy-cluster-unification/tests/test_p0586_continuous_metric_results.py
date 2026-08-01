import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]


def report(stage: str) -> dict:
    return json.loads((ROOT / f"results/{stage}/report.json").read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.parametrize(
    "stage",
    [
        "p0586_continuous_baryonic_metric",
        "p0586b_metric_boundary_response",
        "p0586c_signed_metric_response",
        "p0586d_signed_metric_exact",
    ],
)
def test_p0586_series_protocol_hashes_are_locked(stage):
    result = report(stage)
    protocol = ROOT / result["protocol"]["path"]
    assert result["protocol"]["sha256"] == sha256(protocol)


def test_p0586_primary_factorial_selects_direction_without_scalar_boost():
    result = report("p0586_continuous_baryonic_metric")
    assert result["coverage"]["screen_candidates"] == 324
    selected = result["selected"]
    assert selected["minimum_permittivity"] == 1.0
    assert selected["anisotropy_tau"] == 0.6
    assert selected["smoothing_r80_fraction"] == 0.5
    assert selected["selection_improvement_vs_identity_fraction"] == pytest.approx(
        0.0047317058221886255
    )
    assert result["exact_validation"]["improvement_fraction"] < 0.0


def test_p0586_anisotropy_dominates_the_local_parameter_response():
    impacts = {
        row["coordinate"]: row
        for row in report("p0586_continuous_baryonic_metric")["parameter_impacts"]
    }
    assert impacts["anisotropy_tau"]["selected_local_span_arcsec"] > 0.12
    assert impacts["minimum_permittivity"]["selected_local_span_arcsec"] < 0.06
    assert impacts["smoothing_r80_fraction"]["selected_local_span_arcsec"] < 0.05
    assert impacts["gate_power"]["selected_local_span_arcsec"] < 0.004
    assert impacts["a0_m_s2"]["selected_local_span_arcsec"] < 0.0005


def test_p0586_scalar_branch_improves_newton_but_does_not_reach_rar():
    cross = report("p0586_continuous_baryonic_metric")["cross_domain"]
    best = cross["best_galaxy_grid_point"]
    assert best["SPARC_outer_RMSE_km_s"] == pytest.approx(17.099765338910466)
    assert best["Earth_pass"]
    assert best["Mercury_pass"]
    assert cross["Newtonian_outer_RMSE_km_s"] == pytest.approx(72.39921475798786)
    assert cross["fixed_RAR_outer_RMSE_km_s"] == pytest.approx(10.348465773189679)


def test_p0586b_positive_extension_has_no_four_system_candidate_and_loses_a_root():
    result = report("p0586b_metric_boundary_response")
    assert result["all_four_fixed_geometry"]["candidates_improving_all_four"] == 0
    assert result["selected"]["anisotropy_tau"] == 1.2
    assert result["selected"]["systems_improved_fixed_geometry"] == 3
    assert not result["exact_validation"]["selected_all_roots"]


def test_p0586c_negative_broad_metric_has_four_common_screen_candidates():
    result = report("p0586c_signed_metric_response")
    assert result["coverage"]["candidates"] == 108
    assert result["candidates_improving_all_four"] == 4
    best = result["best_common_candidate"]
    assert best["minimum_permittivity"] == 1.0
    assert best["anisotropy_tau"] == -1.2
    assert best["smoothing_r80_fraction"] == 0.8
    assert best["systems_improved"] == 4
    assert not result["gates"]["common_optimum_same_tau_sign"]


def test_p0586c_response_conflict_is_strongest_for_macs1115():
    correlations = report("p0586c_signed_metric_response")["response_correlations"]
    lookup = {
        frozenset((row["left_system"], row["right_system"])): row[
            "Spearman_response_correlation"
        ]
        for row in correlations
    }
    assert lookup[frozenset(("MACS1115", "MACS1931"))] < -0.67
    assert lookup[frozenset(("MACS1115", "MACS0429"))] < -0.57


def test_p0586d_exact_gain_is_monotonic_and_all_roots_survive():
    result = report("p0586d_signed_metric_exact")
    sensitivity = result["sensitivity"]
    rms = [row["heldout_exact_RMS_arcsec"] for row in sensitivity]
    assert all(left > right for left, right in zip(rms, rms[1:]))
    assert all(row["all_training_roots"] for row in sensitivity)
    assert all(row["all_heldout_roots"] for row in sensitivity)
    primary = result["primary_exact"]
    assert primary["improvement_fraction"] == pytest.approx(0.020055134264989283)
    assert not primary["all_systems_improve"]
    assert primary["primary_to_compact_ratio"] > 1.76


def test_p0586d_system_gains_and_affine_failure_are_not_hidden():
    exact = pd.read_csv(ROOT / "results/p0586d_signed_metric_exact/exact_scores.csv")
    systems = exact[exact.row_type.eq("system")].pivot(
        index="system_label", columns="model_id", values="heldout_exact_RMS_arcsec"
    )
    assert systems.loc["MACS0329", "primary_tau_m1p2"] < systems.loc["MACS0329", "zero"]
    assert systems.loc["MACS1931", "primary_tau_m1p2"] < systems.loc["MACS1931", "zero"]
    assert systems.loc["MACS0429", "primary_tau_m1p2"] > systems.loc["MACS0429", "zero"]
    assert systems.loc["MACS1115", "primary_tau_m1p2"] > systems.loc["MACS1115", "zero"]
    result = report("p0586d_signed_metric_exact")
    assert result["numerical"]["maximum_primary_affine_vector_R2"] > 0.99
    assert not result["gates"]["mass_sheet_audit_pass"]
    assert result["gates"]["curl_pass"]
    assert result["gates"]["positive_metric_pass"]
    assert not result["gates"]["formula_promoted"]
