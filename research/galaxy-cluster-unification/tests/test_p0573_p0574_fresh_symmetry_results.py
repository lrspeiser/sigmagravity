import json
from pathlib import Path

import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
P0573 = ROOT / "results" / "p0573_tidal_arrival_fresh_replication"
P0574 = ROOT / "results" / "p0574_symmetry_gated_arrival_microvariation"


def report(path: Path) -> dict:
    return json.loads((path / "report.json").read_text(encoding="utf-8"))


def test_p0573_is_a_zero_refit_three_system_replication():
    result = report(P0573)
    assert result["coverage"]["fresh_clusters"] == 3
    assert result["coverage"]["lenstool_realizations"] == 300
    assert result["coverage"]["glafic_method_controls"] == 3
    assert result["coverage"]["parameters_fit_on_fresh_systems"] == 0


def test_p0573_locked_cluster_signal_replicates_in_both_lens_methods():
    outcome = report(P0573)["result"]
    assert outcome["improvement_vs_local_fraction"] == pytest.approx(0.18369941766377784)
    assert outcome["systems_improved"] == 3
    assert outcome["lenstool_realizations_improved_fraction"] > 0.88
    assert outcome["locked_mean_Pearson"] > outcome["local_mean_Pearson"]
    assert outcome["glafic_improvement_vs_local_fraction"] > 0.28
    assert outcome["glafic_systems_improved"] == 3


def test_p0573_exposes_the_extended_axisymmetric_failure():
    result = report(P0573)
    disk = result["cross_domain"]["extended_axisymmetric_exponential_disk"]
    assert disk["activation_RMS"] > 0.4
    assert result["cross_domain"]["solar_point_source"]["activation_RMS"] == 0.0
    assert not result["gates"]["extended_axisymmetric_disk_null_pass"]
    assert not result["gates"]["raw_lensing_followup_authorized"]


def test_p0574_selects_the_frozen_60_kpc_symmetry_gated_candidate():
    result = report(P0574)
    selected = result["result"]["selected_candidate"]
    assert selected == {
        "candidate_id": "width_60",
        "alpha": 0.5,
        "beta": 1.0,
        "width_kpc": 60.0,
        "f": 0.8,
        "Q0": 0.05,
        "n": 4.0,
    }
    assert result["coverage"]["historical_development_clusters"] == 13
    assert result["coverage"]["candidates"] == 14


def test_p0574_retains_cluster_gain_and_passes_method_controls():
    outcome = report(P0574)["result"]
    assert outcome["validation_improvement_vs_local_fraction"] > 0.15
    assert outcome["validation_systems_improved"] == 3
    assert outcome["retained_fraction_of_P0573_no_gate_gain"] > 0.85
    assert outcome["validation_realizations_improved_fraction"] > 0.84
    assert outcome["glafic_improvement_vs_local_fraction"] > 0.24
    assert outcome["glafic_systems_improved"] == 3


def test_p0574_symmetry_gate_is_exactly_inert_in_axisymmetric_domains():
    result = report(P0574)
    cross = result["cross_domain"]
    assert cross["extended_axisymmetric_disk_Q90"] == 0.0
    assert cross["extended_axisymmetric_disk_effective_route_fraction"] == 0.0
    assert cross["solar_effective_route_fraction"] == 0.0
    assert cross["SPARC_galaxies"] == 175
    assert cross["SPARC_maximum_angular_layer_velocity_change_km_s"] == 0.0
    assert result["gates"]["raw_lensing_followup_authorized"]


def test_p0574_extent_dominates_the_microvariation_impacts():
    impacts = pd.DataFrame(report(P0574)["parameter_impacts"])
    assert impacts.iloc[0].coordinate == "arrival_width"
    assert impacts.iloc[0].relative_span > 0.05
    assert impacts.iloc[-1].coordinate == "Q0"
    assert impacts.iloc[0].relative_span > 100 * impacts.iloc[-1].relative_span
